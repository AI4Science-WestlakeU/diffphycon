import numpy as np
import torch
from torch import nn
from torch_geometric.data import Dataset, Data
import pdb
import sys, os
import argparse
import logging
import datetime
from torch.optim import lr_scheduler
from diffusion_2d import ForceUnet, Unet
from dataset.data_surrogate_models_2d import ForceData, SimulatorData, BoundaryUpdaterData
from tqdm import tqdm
import time

def get_device(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    return device

def load_model(args):
    if args.model_name == "force":
        model = ForceUnet(
            dim = args.image_size,
            out_dim = 1,
            dim_mults = (1, 2, 4, 8),
            channels=args.input_channel
        )
    elif args.model_name == "boundary_updater":
        model = Unet(
            dim = args.image_size,
            out_dim = 3,
            dim_mults = (1, 2, 4, 8),
            channels=args.input_channel
        )
    elif args.model_name == "simulator":
        if args.only_vis_pressure:
            model = Unet(
                dim = args.image_size,
                out_dim = 1,
                dim_mults = (1, 2, 4, 8),
                channels=4
            )
        else:
            if args.pred_mask_offset:
                model = Unet(
                    dim = args.image_size,
                    out_dim = 6,
                    dim_mults = (1, 2, 4, 8),
                    channels=args.input_channel
                )
            else:
                model = Unet(
                    dim = args.image_size,
                    out_dim = 3,
                    dim_mults = (1, 2, 4, 8),
                    channels=args.input_channel
                )
    else:
        raise
        
    if args.resume_training:
        print("args.resume_checkpoint: ", args.resume_checkpoint)
        # assert False
        model.load_state_dict(torch.load(args.resume_checkpoint))
        print("model loaded from: ", args.resume_checkpoint)

    return model

def load_data(args):
    if args.model_name == "force":
        ds_train = ForceData(
            dataset_path=args.dataset_path,
            n_simu = args.n_simu,
            train_split_ratio=args.train_split_ratio,
            is_train=True
        )
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 32)
        ds_test = ForceData(
            dataset_path=args.dataset_path,
            n_simu = args.n_simu,
            train_split_ratio=args.train_split_ratio,
            is_train=False
        )
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    elif args.model_name == "simulator":
        ds_train = SimulatorData(
            dataset_path=args.dataset_path,
            n_simu = args.n_simu,
            train_split_ratio=args.train_split_ratio,
            is_train= True,
            pred_mask_offset=args.pred_mask_offset,
            only_vis_pressure=args.only_vis_pressure
        )
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 32)
        ds_test = SimulatorData(
            dataset_path=args.dataset_path,
            n_simu = args.n_simu,
            train_split_ratio=args.train_split_ratio,
            is_train= False,
            pred_mask_offset=args.pred_mask_offset,
            only_vis_pressure=args.only_vis_pressure
        )
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    elif args.model_name == "boundary_updater":
        ds_train = BoundaryUpdaterData(
            dataset_path=args.dataset_path,
            n_simu = args.n_simu,
            train_split_ratio=args.train_split_ratio,
            incremental=args.incremental,
            is_train= True
        )
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 32)
        ds_test = BoundaryUpdaterData(
            dataset_path=args.dataset_path,
            n_simu = args.n_simu,
            train_split_ratio=args.train_split_ratio,
            incremental=args.incremental,
            is_train= False
        )
        test_loader = torch.utils.data.DataLoader(ds_test, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    else:
        raise
        
    return train_loader, test_loader

def get_scheduler(args, start_epoch, model):
    steps = [3, 6, 10]
    if args.resume_training:
        if args.resume_epoch < steps[0]:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[steps[0] - start_epoch, steps[1] - start_epoch, steps[2] - start_epoch], gamma=0.1)
        if args.resume_epoch < steps[1]:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[steps[1] - start_epoch, steps[2] - start_epoch], gamma=0.1)
        elif args.resume_epoch < steps[2]:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[steps[2] - start_epoch], gamma=0.1)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)
    
    return optimizer, scheduler

def get_normalizer(args):
    import pickle
    normalization_filename = os.path.join(args.dataset_path, "normalization_max_min.pkl")
    normdict = pickle.load(open(normalization_filename, "rb"))
    p_max = normdict["p_max"]
    p_min = normdict["p_min"] 
    
    return p_max, p_min
    

def train(args):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = os.path.join(args.log_dir, "{}_{}.log".format(args.model_name, current_time))
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('args: {}'.format(args))
    train_loader, test_loader = load_data(args)
    print("number of batch in train_loader: ", len(train_loader))
    print("number of batch in test_loader: ", len(test_loader))
    model = load_model(args)
    device = get_device(args)
    model = model.to(device)  
    if args.model_name == "simulator":
        max_val, min_val = get_normalizer(args)
    model.train()
    criterion = torch.nn.MSELoss().to(device)
    ckp_folder = os.path.join(args.checkpoint_dir, current_time)
    os.makedirs(ckp_folder, exist_ok=True)
    # train
    logging.info('Start training...')
    start_epoch = 0 if not args.resume_training else args.resume_epoch
    optimizer, scheduler = get_scheduler(args, start_epoch, model)
    for epoch in range(start_epoch, args.n_epoch):
        losses = []
        logging.info(f'Epoch [{epoch}/{args.n_epoch}], Learning Rate: {optimizer.param_groups[0]["lr"]}')
        t2 = time.time()
        for i, data in tqdm(enumerate(train_loader)):
            t0 = time.time()
            if args.model_name == "force":
                x, y = data
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                losses.append(loss.item()) 
            elif args.model_name == "simulator":
                x, theta_delta, y = data
                x = x.to(device) #[batch, 6, 64, 64]
                theta_delta = theta_delta.to(device).squeeze(-1)
                y = y.to(device) #[batch, 3, 64, 64]
                optimizer.zero_grad()
                y_pred = model(x, theta_delta)
                loss = criterion(y_pred, y)
                losses.append(loss.item()) 
            elif args.model_name == "boundary_updater":
                x, theta, y = data
                x = x.to(device) #[batch, 6, 126, 126]
                theta = theta.to(device).squeeze(-1)
                y = y.to(device) #[batch, 3, 126, 126]
                x_pad = torch.zeros(x.size(0), x.size(1), args.image_size, args.image_size).to(device)
                x_pad[:, :, 1:-1, 1:-1] = x
                x = x_pad
                y_pad = torch.zeros(y.size(0), y.size(1), args.image_size, args.image_size).to(device)
                y_pad[:, :, 1:-1, 1:-1] = y
                y = y_pad
                optimizer.zero_grad()
                y_pred = model(x, theta)
                loss = criterion(y_pred, y)
                losses.append(loss.item()) 
            else:
                assert False
                
            if i % 100 == 0:
                logging.info("training epoch {}, iter: {}, loss: {}".format(epoch, i, loss))
            loss.backward()
            optimizer.step()
            if i > 0 and i % 10000 == 0:
                ckpt_path = os.path.join(ckp_folder, 'epoch_{}_iter_{}.pth'.format(epoch, i))
                torch.save(model.state_dict(), ckpt_path)
        average_loss = torch.mean(torch.tensor(losses))  
        logging.info("training epoch {}, average loss: {}".format(epoch, average_loss))
        
        # test on test set
        logging.info("staring testing...")
        model.eval()
        with torch.no_grad():
            losses = []
            relative_errors = []
            for j, data in enumerate(test_loader):
                if args.model_name == "force":
                    x, y = data
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    losses.append(loss.item()) 
                    relative_errors.append(torch.norm(y - y_pred) / torch.norm(y))
                elif args.model_name == "simulator" or args.model_name == "boundary_updater":
                    x, theta_delta, y = data
                    x = x.to(device) #[batch, 6, 126, 126]
                    theta_delta = theta_delta.to(device).squeeze(-1)
                    y = y.to(device) #[batch, 3, 126, 126]
                    if args.model_name == "boundary_updater":
                        x_pad = torch.zeros(x.size(0), x.size(1), args.image_size, args.image_size).to(device)
                        x_pad[:, :, 1:-1, 1:-1] = x
                        x = x_pad
                        y_pad = torch.zeros(y.size(0), y.size(1), args.image_size, args.image_size).to(device)
                        y_pad[:, :, 1:-1, 1:-1] = y
                        y = y_pad
                    y_pred = model(x, theta_delta)
                    loss = criterion(y_pred, y)
                    losses.append(loss.item()) 
                    relative_errors.append(torch.norm(y - y_pred) / torch.norm(y))
                else:
                    assert False
            average_loss = torch.mean(torch.tensor(losses))  
            average_relative_error = torch.mean(torch.tensor(relative_errors))
            logging.info("testing epoch {}, average loss: {}, average relative error: {}".format(epoch, average_loss, average_relative_error))
        # save model
        ckpt_path = os.path.join(ckp_folder, 'epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), ckpt_path)
        logging.info('model saved: {}'.format(str(ckpt_path)))
        
        scheduler.step()

# will be push into utils
def mask_denoise(tensor, thre_mask=0.5, thre_offset1=0.6, thre_offset2=-0.5):
    if len(tensor.shape) == 2: # only mask
        return torch.where(tensor > thre_mask, torch.tensor(1).to(tensor.device), torch.tensor(0).to(tensor.device))

    else: # mask and offset
        # for mask, set to 1 if > thre_mask, else set to 0
        mask = tensor[:, 0, :, :]
        mask = torch.where(mask > thre_mask, torch.tensor(1).to(tensor.device), torch.tensor(0).to(tensor.device))
        # for offset, set to 1 if > thre_offset1, else set to unchanged
        offset = tensor[:, 1:, :, :]
        offset = torch.where(offset > thre_offset1, torch.tensor(1.).to(tensor.device), offset)
        # for offset, set to thre_offset2 if < thre_offset2, else set to unchanged
        offset = torch.where(offset < thre_offset2, torch.tensor(thre_offset2).to(tensor.device), offset)
        tensor = torch.cat([mask.unsqueeze(1), offset], dim=1)
        return tensor
    

def plot(x, thetas, y, y_gt):
    import matplotlib.pyplot as plt
    
    save_dir = "../plot_boundaries/"
    n = len(y)
    relative_errors = []
    dismatchs = []
    offset_relative_errors = []
    # clamp y[:,1:] to [-0.5, 1]
    y[:,1:] = torch.clamp(y[:,1:], -0.5, 1)
    assert torch.max(y[:,1:]) <= 1
    assert torch.min(y[:,1:]) >= -0.5
    for i in range(n):
        fig, axs = plt.subplots(1, 7, figsize=(32, 4))
        theta = thetas[i]* 180 / np.pi
        # for j in range(3):
        axs[0].imshow(x[i, 0, :, :])
        axs[0].set_title("input mask of time t, theta: {:.2f}".format(theta))
        
        axs[1].imshow(y[i, 0, :, :])
        axs[1].set_title("output mask of time t+1")
        # plt.show()
        # for j in range(3):
        axs[2].imshow(y_gt[i, 0, :, :])
        axs[2].set_title("ground truth mask of time t+1")
        
        im3 = axs[3].imshow(y[i, 1, :, :])
        axs[3].set_title("output offset x of time t+1")
        fig.colorbar(im3, ax=axs[3])
        
        # diff = mask_denoise(y[i,0]) != mask_denoise(y_gt[i,0])
        im4 = axs[4].imshow(y[i, 1, :, :] - y_gt[i, 1, :, :])
        axs[4].set_title("difference offset x of time t+1")
        fig.colorbar(im4, ax=axs[4])
        
        im5 = axs[5].imshow(y[i, 2, :, :])
        axs[5].set_title("output offset y of time t+1")
        fig.colorbar(im5, ax=axs[5])
        
        # diff = mask_denoise(y[i,0]) != mask_denoise(y_gt[i,0])
        im6 = axs[6].imshow(y[i, 2, :, :] - y_gt[i, 2, :, :])
        axs[6].set_title("difference offset y of time t+1")
        fig.colorbar(im6, ax=axs[6])
        
        # diff = mask_denoise(y[i,0]) != mask_denoise(y_gt[i,0])
        # axs[5].imshow(diff)
        # axs[5].set_title("diff between output and ground truth")
        
        
        
        plt.savefig(os.path.join(save_dir, "boundary_updater_new_{}.png").format(i))
        
        relative_error = torch.norm(y[i,0] - y_gt[i,0]) / torch.norm(y_gt[i,0])
        offset_relative_error = torch.norm(y[i,1:] - y_gt[i,1:]) / torch.norm(y_gt[i,1:])
        dismatch = torch.sum(mask_denoise(y[i,0]) != mask_denoise(y_gt[i,0]))
        relative_errors.append(relative_error)
        dismatchs.append(dismatch)
        offset_relative_errors.append(offset_relative_error)
    print("mask relative error: ", np.mean(relative_errors), np.std(relative_errors))
    print("mask dismatch: ", np.mean(dismatchs), np.std(dismatchs))
    print("offset relative error: ", np.mean(offset_relative_errors), np.std(offset_relative_errors))
    
def plot_simulator(x, thetas, y, y_gt):
    import matplotlib.pyplot as plt
    
    save_dir = "../plot_boundaries/"
    n = len(y)
    relative_errors = []
    dismatchs = []
    offset_relative_errors = []
    # clamp y[:,1:] to [-0.5, 1]
    # y[:,1:] = torch.clamp(y[:,1:], -0.5, 1)
    # assert torch.max(y[:,1:]) <= 1
    # assert torch.min(y[:,1:]) >= -0.5
    dim = 0 # vx
    for i in range(n):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        theta = thetas[i]* 180 / np.pi
        # for j in range(3):
        im0 = axs[0].imshow(x[i, dim, :, :])
        axs[0].set_title("input p of time t, theta: {:.2f}".format(theta))
        # set colorbar, range from 0 to 1
        fig.colorbar(im0, ax=axs[0])
        
        im1 = axs[1].imshow(y[i, dim, :, :])
        axs[1].set_title("output p of time t+1")
        fig.colorbar(im1, ax=axs[1])
        # plt.show()
        # for j in range(3):
        im2 = axs[2].imshow(y_gt[i, dim, :, :])
        axs[2].set_title("ground truth p of time t+1")
        fig.colorbar(im2, ax=axs[2])
        plt.savefig(os.path.join(save_dir, "simulator_p_{}.png").format(i))
        
        relative_error = torch.norm(y[i,dim] - y_gt[i,dim]) / torch.norm(y_gt[i,dim])
        relative_errors.append(relative_error)
    

def evaluate(args):
    train_loader, test_loader = load_data(args)
    # checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint))
    args.resume_training = True
    model = load_model(args)
    if args.model_name == "simulator":
        max_val, min_val = get_normalizer(args)
    device = get_device(args)
    model = model.to(device)  
    criterion = torch.nn.MSELoss().to(device)
    model.eval()
    with torch.no_grad():
        losses = []
        relative_errors = []
        for j, data in tqdm(enumerate(test_loader)):
            if args.model_name == "force":
                x, y = data
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                losses.append(loss.item()) 
                relative_errors.append(torch.norm(y - y_pred) / torch.norm(y))
            else:
                x, theta, y = data
                x = x.to(device) #[batch, 6, 126, 126]
                theta = theta.to(device).squeeze(-1)
                y = y.to(device) #[batch, 3, 126, 126]
                x_pad = torch.zeros(x.size(0), x.size(1), args.image_size, args.image_size).to(device)
                x_pad[:, :, 1:-1, 1:-1] = x
                x = x_pad
                y_pad = torch.zeros(y.size(0), y.size(1), args.image_size, args.image_size).to(device)
                y_pad[:, :, 1:-1, 1:-1] = y
                y = y_pad
                y_pred = model(x, theta)
                if args.model_name == "boundary_updater":
                    # plot y_pred of shape [batch, 3, 126, 126]
                    if j == 0:
                        y_pred_array = mask_denoise(y_pred).cpu()
                        x_array = x.cpu()
                        theta_array = theta.cpu()
                        y_gt_array = y.cpu()
                        plot(x_array, theta_array, y_pred_array, y_gt_array)
                    y_pred = mask_denoise(y_pred)
                else:
                    if j == 0:
                        y_pred_array = y_pred.cpu()
                        x_array = x.cpu()
                        theta_array = theta.cpu()
                        y_gt_array = y.cpu()
                        plot_simulator(x_array, theta_array, y_pred_array, y_gt_array)
                loss = criterion(y_pred, y)
                losses.append(loss.item()) 
                relative_errors.append(torch.norm(y - y_pred) / torch.norm(y))
        average_loss = torch.mean(torch.tensor(losses))  
        average_relative_error = torch.mean(torch.tensor(relative_errors))
        print("testing average loss: {}, average relative error: {}".format(average_loss, average_relative_error))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='simulator', help='force | simulator | boundary_updater (default: force)')
    parser.add_argument('--dataset_path', default="/data/user/pde_ctrl/jellyfish/train_data", type=str, help='path to dataset')
    parser.add_argument('--n_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--n_simu', type=int, default=30000)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--train_split_ratio', type=float, default=0.99)
    parser.add_argument('--input_channel', type=int, default=6, help='force: 4 | simulator: 6 | boundary_updater 3 (default: 6)')
    parser.add_argument('--checkpoint_dir', type=str, default='/data/user/pde_ctrl/checkpoints/surrogate_models/simulator_model/')
    parser.add_argument('--log_dir', type=str, default='/data/user/pde_ctrl/logs/surrogate_models/simulator_model/')
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--pred_mask_offset', action='store_true', help="whether predict mask and offset, only used in simulator model")
    parser.add_argument('--only_vis_pressure', action='store_true', help="whether only observe pressure, only used in simulator model")
    parser.add_argument('--incremental', action='store_true', help="whether use incremental dataset, only used in boundary_updater model")
    parser.add_argument('--resume_checkpoint', type=str, default='/data/user/pde_ctrl/checkpoints/surrogate_models/simulator_model/epoch_3.pth')
    parser.add_argument('--resume_epoch', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    train(args)
    # evaluate(args)