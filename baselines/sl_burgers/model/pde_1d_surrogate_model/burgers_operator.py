import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
try:
    from scripts.utils import *
    from scripts.burgers_numeric import *
    from scripts.models import *
except:
    from .scripts.utils import *
    from .scripts.burgers_numeric import *
    from .scripts.models import *
from pde_gen_control.data_burgers_1d import Burgers1DSimple, Burgers
import argparse
import pdb
RESCALER = RESCALER_1e5 = 6.4519
# RESCALER = RESCALER_1e5 = 1
class Printer(object):
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))
from datetime import datetime
def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string
p = Printer(n_digits=6)
def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    parser.add_argument('--experiment', type=str, default='PDE_1d_surrogate model',
                    help='Experiment for which data should create for')
    parser.add_argument('--exp_path', default='/user/project/pde_gen_control', type=str, help='experiment folder')
    parser.add_argument('--date_time', default='2023-12-02_1d_surrogate_model', type=str, help='experiment date')
    parser.add_argument('--num_t', default=10, type=int, help='the number of time interval')
    parser.add_argument('--gpu', default=6, type=int, help='device number')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=500, type=int, help = 'Number of Epochs')
    parser.add_argument('--train_batch_size', default=51, type=int, help = 'size of batch')
    parser.add_argument('--dataset_path', default="/user/project/pde_gen_control/dataset/dataset_control_burgers/free_u_f_1e5", type=str, help = 'size of batch')
    parser.add_argument('--autoregress_steps', default=1, type=int, help = 'use autoregressive n steps prediction loss to train model')
    parser.add_argument('--is_partially_observable', type=int, default=0,
                    help='0: fully observable; 1. partially_observable')
    parser.add_argument('--is_partially_controllable', type=int, default=0,
                    help='0: fully controllable; 1. partially_controllable')
    return parser.parse_args(argv)


class Simu_surrogate_model(object):
    def __init__(self, path,device,s=128,s_ob=128,milestone=500):
        self.path=path
        # pdb.set_trace()
        # self.model_f = Net_f(s).to(device).eval()
        # self.model_u = Net_u(s_ob).to(device).eval()
        self.model_f=Net_f_ELU(s).to(device).eval()
        self.model_u=Net_u_ELU(s_ob).to(device).eval()
        self.model_trans = Net_trans().to(device).eval()
        milestone=milestone
        # pdb.set_trace()
        self.model_f.load_state_dict(torch.load(self.path+f"/model_f-{milestone}.pt"))
        self.model_u.load_state_dict(torch.load(self.path+f"/model_u-{milestone}.pt"))
        self.model_trans.load_state_dict(torch.load(self.path+f"/model_trans-{milestone}.pt"))
        self.s=s
        self.s_ob=s_ob
    @torch.no_grad()
    def simulation(self,ut=None,ft=None,num_t=1):
        '''
        input:
            ut: state ad time t [batch_size,1,ns]
            ft: force at time t [batch_size,1,ns]
        output:
            u_t_plus_1: state at time t+1 [batch_size,1,ns]
        '''
        # pdb.set_trace()
        # print(RESCALER)
        ut=ut.reshape((ut.shape[0],1,self.s_ob))/RESCALER
        ft=ft/RESCALER
        # print(ft.shape)
        f_rec, f_latent = self.model_f(ft[:,[0]])
        u_t_rec, u_t_latent = self.model_u(ut[:,[0]])
        if u_t_latent.shape[-1]<f_latent.shape[-1]:
            f_latent=f_latent[:,:u_t_latent.shape[-1]]+f_latent[:,u_t_latent.shape[-1]:]
        # u_t_dt_rec, u_t_dt_latent = model_u(y_test) #trans, u_t_dt_latent = model_u(y_test)
        nn_u_t_dt_latent = self.model_trans(torch.cat((u_t_latent.view(-1,8,self.s_ob//4), f_latent.view(-1,8,self.s_ob//4)), 1))
        u_t_plus_1 = self.model_u.up(nn_u_t_dt_latent)

        return u_t_plus_1*RESCALER
if __name__=='__main__':
    #argparser
    args = get_args()
    print(args)
    args.exp_path=args.exp_path+"/results/"+args.date_time+"/"
    if args.is_partially_observable==1:
            obsereved_mask=torch.ones((args.train_batch_size*10,128))
            # control_mask[:,:32]=control_mask[:,:32]*0
            # control_mask[:,-32:]=control_mask[:,-32:]*0
            obsereved_mask[:,32:96]=obsereved_mask[:,32:96]*0
    else:
        obsereved_mask=None
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)
    #gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    use_cuda = torch.cuda.is_available()
    print("Is available to use cuda? : ",use_cuda)
    if use_cuda:
        print("-> GPU number {}".format(args.gpu))
    
    #learning parameters
    lr = args.lr
    epochs = args.epochs
    
    # f_data, u_data = torch.load('data/burgers_dataset')
    
    #setting parameters
    Nu0 = 10000
    Nf = 10000
    num_t = 10
    if args.is_partially_observable==1:
        s_ob=64
    else:
        s_ob=128
    if args.is_partially_controllable==1:
        s=128
    else:
        s=128
    
    Ndata = Nu0*num_t
    
    # ##num_interior=128 // num_pts=130=0,dx,...,129*dx // dx=1/129
    # xmin = 0.0; xmax = 1.0
    # delta_x = (xmax-xmin)/(s+1)
    # x = torch.linspace(xmin+delta_x,xmax-delta_x,s)
    
    
    #model
    # model_f = Net_f(s).cuda()
    # model_u = Net_u(s_ob).cuda()

    #ELU activation function
    model_f = Net_f_ELU(s).cuda()
    model_u = Net_u_ELU(s_ob).cuda()

    model_trans = Net_trans().cuda()
    
    logs = dict()

    logs['train_loss']=[]
    logs['train_loss_f_t_rec']=[]
    logs['train_loss_u_t_rec']=[]
    logs['train_loss_trans']=[]
    logs['train_loss_trans_latent']=[]

    logs['test_loss']=[]
    logs['test_loss_f_t_rec']=[]
    logs['test_loss_u_t_rec']=[]
    logs['test_loss_trans']=[]
    logs['test_loss_trans_latent']=[]
    
    # class Burgers_Dataset(Dataset): 
    #     def __init__(self, f_data, u_data):
    #         f_data = f_data.reshape(1,-1,1,s).repeat(Nu0,1,num_t,1).reshape(-1,1,s)
    #         u_t_data = u_data[:,:,:-1].reshape(-1,1,s)
    #         u_t_dt_data = u_data[:,:,1:].reshape(-1,1,s)

    #         self.input_data = torch.cat((u_t_data, f_data), 1)
    #         self.output_data = u_t_dt_data

    #     def __len__(self):
    #         return Ndata

    #     def __getitem__(self, idx): 
    #         pdb.set_trace()
    #         x = torch.FloatTensor(self.input_data[idx]) #[2,128]
    #         y = torch.FloatTensor(self.output_data[idx])
    #         return x, y
    
    
    train_dataset= Burgers1DSimple(
    dataset="burgers",
    input_steps=1,
    output_steps=10,
    time_interval=1,
    is_y_diff=False,
    split="train",
    transform=None,
    pre_transform=None,
    verbose=False,
    root_path =args.dataset_path ,
    device='cuda',
    rescaler=RESCALER
    )
    test_dataset= Burgers1DSimple(
    dataset="burgers",
    input_steps=1,
    output_steps=10,
    time_interval=1,
    is_y_diff=False,
    split="test",
    transform=None,
    pre_transform=None,
    verbose=False,
    root_path =args.dataset_path ,
    device='cuda',
    rescaler=RESCALER,
    )
    index_list=[0,11,1,1,12,2,2,13,3,3,14,4,4,15,5,5,16,6,6,17,7,7,18,8,8,19,9,9,20,10]
    index_np=np.array(index_list)
    index_list=index_np.reshape(10,3)
    sub_index_shuffle=np.arange(10)
    np.random.shuffle(sub_index_shuffle)
    shuffle_index=index_list[sub_index_shuffle]
    shuffle_index=shuffle_index.reshape(-1)##[30]
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.train_batch_size, shuffle=False) #full batch
    optimizer = torch.optim.Adam(list(model_u.parameters()) + list(model_f.parameters()) + list(model_trans.parameters()), lr=lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)
    #following set CosineAnnealingLR ,update lr each 5 epochs
    scheduler=  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00000, last_epoch=-1,)
    # scheduler=  torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.001, step_size_up=35, step_size_down=40, mode="triangular2", gamma=0.9, scale_fn=None, scale_mode="iterations", cycle_momentum=False)
    train_loss_list=[]
    test_loss_list=[]
    for epoch in range(1, epochs+1):
        # train
        # if epoch==1:
        p.print(f"{epoch} epoch_start", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
        model_f.train()
        model_u.train()
        model_trans.train()

        train_loss=AverageMeter()
        train_loss_f_t_rec=AverageMeter()
        train_loss_u_t_rec=AverageMeter()
        train_loss_trans=AverageMeter()
        train_loss_trans_latent=AverageMeter()
        for x_train in train_loader:
            #x_train [512,21,128]
            if args.autoregress_steps==1:
            # if False:
                x_train=x_train[:,shuffle_index,:].clone()
                x_train=x_train.reshape(x_train.shape[0],10,3,128)
                x_train_reshape=x_train.reshape(x_train.shape[0]*10,3,128).clone() #[batch_size*10,3,128]
                # x_train, y_train = x_train.cuda(), y_train.cuda() #x:[512, 2, 128] y:[512, 1, 128]
                x_train=x_train_reshape[:,:2,:].cuda()
                y_train=x_train_reshape[:,2:3,:].cuda()
                if s_ob==64:
                    ut_current=torch.cat([x_train[:,[0],:32],x_train[:,[0],96:]],dim=2).cuda()
                    ut_next=torch.cat([y_train[:,[0],:32],y_train[:,[0],96:]],dim=2).cuda()
                else:
                    ut_current=x_train[:,[0],:].cuda()
                    ut_next=y_train[:,[0],:].cuda()
                if s==64:
                    ft_current=torch.cat([x_train[:,[1],:32],x_train[:,[1],96:]],dim=2).cuda()
                else:
                    ft_current=x_train[:,[1],:].cuda()

                optimizer.zero_grad()
                # pdb.set_trace()
                f_rec, f_latent = model_f(ft_current) #f_rec [510, 1, 128] f_latent [510,256]
                u_t_rec, u_t_latent = model_u(ut_current) #
                u_t_dt_rec, u_t_dt_latent = model_u(ut_next) #trans, u_t_dt_latent = model_u(y_test)
                if u_t_latent.shape[-1]<f_latent.shape[-1]:
                    # pdb.set_trace()
                    f_latent=f_latent[:,:u_t_latent.shape[-1]]+f_latent[:,u_t_latent.shape[-1]:]
                nn_u_t_dt_latent = model_trans(torch.cat((u_t_latent.view(-1,8,s_ob//4), f_latent.view(-1,8,s_ob//4)), 1))

                trans = model_u.up(nn_u_t_dt_latent)
                # if s_ob==64:
                #     obsereved_mask_reshape=obsereved_mask.reshape(obsereved_mask.shape[0],1,-1).cuda()
                #     loss1 = rel_error2(f_rec, ft_current).mean()
                #     loss2 = rel_error(u_t_rec*obsereved_mask_reshape[:x_train.shape[0]], x_train[:,[0],:]*obsereved_mask_reshape[:x_train.shape[0]]).mean()
                #     loss3 = rel_error(trans*obsereved_mask_reshape[:x_train.shape[0]], y_train*obsereved_mask_reshape[:x_train.shape[0]]).mean()
                #     loss5 = rel_error(nn_u_t_dt_latent.view(-1,8*s//4), u_t_dt_latent).mean()
                # else:
                loss1 = rel_error2(f_rec, ft_current).mean()
                loss2 = rel_error(u_t_rec,ut_current).mean()
                loss3 = rel_error(trans, ut_next).mean()
                # pdb.set_trace()
                loss5 = rel_error(nn_u_t_dt_latent.view(-1,8*s_ob//4), u_t_dt_latent).mean()
                loss = loss1 + 0.5*loss2 + 2.5*loss3 + loss5
                # pdb.set_trace()
                loss.backward()

                optimizer.step()
            else:
                # index_start=epoch%8
                #x_train [batch_size,21,128]
                x_train=x_train.clone()
                ut_current=x_train[:,[0],:].cuda()
                ut_next=x_train[:,1:11,:].cuda()
                ft_current=x_train[:,11:,:].cuda()
                if s_ob==64:
                    ut_current=torch.cat([ut_current[:,:,:32],ut_current[:,:,96:]],dim=2).cuda()
                    ut_next=torch.cat([ut_next[:,:,:32],ut_next[:,:,96:]],dim=2).cuda()
                if s==64:
                    ft_current=torch.cat([ft_current[:,:,:32],ft_current[:,:,96:]],dim=2).cuda()

                optimizer.zero_grad()
                # pdb.set_trace()
                loss_sum=0
                for index_start in range(8):
                    for t in range(args.autoregress_steps):
                        t=t+index_start
                        if t==index_start:
                            f_rec, f_latent = model_f(ft_current[:,[t]]) #f_rec [510, 1, 128] f_latent [510,256]
                            u_t_rec, u_t_latent = model_u(ut_current) #
                            u_t_dt_rec, u_t_dt_latent = model_u(ut_next[:,[t]]) #trans, u_t_dt_latent = model_u(y_test)
                        else:
                            f_rec, f_latent = model_f(ft_current[:,[t]]) #f_rec [510, 1, 128] f_latent [510,256]
                            ut_current=trans.clone()
                            # ut_current=ut_next[:,[t-1]].clone()
                            u_t_rec, u_t_latent = model_u(ut_current) #
                            u_t_dt_rec, u_t_dt_latent = model_u(ut_next[:,[t]]) #trans, u_t_dt_latent = model_u(y_test)
                        if u_t_latent.shape[-1]<f_latent.shape[-1]:
                            # pdb.set_trace()
                            f_latent=f_latent[:,:u_t_latent.shape[-1]]+f_latent[:,u_t_latent.shape[-1]:]
                        nn_u_t_dt_latent = model_trans(torch.cat((u_t_latent.view(-1,8,s_ob//4), f_latent.view(-1,8,s_ob//4)), 1))

                        trans = model_u.up(nn_u_t_dt_latent)
                        # if s_ob==64:
                        #     obsereved_mask_reshape=obsereved_mask.reshape(obsereved_mask.shape[0],1,-1).cuda()
                        #     loss1 = rel_error2(f_rec, ft_current).mean()
                        #     loss2 = rel_error(u_t_rec*obsereved_mask_reshape[:x_train.shape[0]], x_train[:,[0],:]*obsereved_mask_reshape[:x_train.shape[0]]).mean()
                        #     loss3 = rel_error(trans*obsereved_mask_reshape[:x_train.shape[0]], y_train*obsereved_mask_reshape[:x_train.shape[0]]).mean()
                        #     loss5 = rel_error(nn_u_t_dt_latent.view(-1,8*s//4), u_t_dt_latent).mean()
                        # else:
                        
                        loss1 = rel_error2(f_rec, ft_current[:,[t]]).mean()
                        loss2 = rel_error(u_t_rec,ut_current).mean()
                        loss3 = rel_error(trans, ut_next[:,[t]]).mean()
                        # pdb.set_trace()
                        loss5 = rel_error(nn_u_t_dt_latent.view(-1,8*s_ob//4), u_t_dt_latent).mean()

                        loss = loss1 + 0.5*loss2 + 2.5*loss3 + loss5
                        # pdb.set_trace()
                        loss_sum=loss_sum+loss
                loss_sum.backward()

                optimizer.step()
                # pdb.set_trace()
                # x_train=x_train[:,index_np,:].clone()#x_train [512,30,128]
                # x_train=x_train.reshape(x_train.shape[0],10,3,128).cuda()
                # ut_current=x_train[:,:,[0],:].cuda()
                # ut_next=x_train[:,:,[2],:].cuda()
                # ft_current=x_train[:,:,[1],:].cuda()
                # if s_ob==64:
                #     ut_current=torch.cat([ut_current[:,:,:,:32],ut_current[:,:,:,96:]],dim=3).cuda()
                #     ut_next=torch.cat([ut_next[:,:,:,:32],ut_next[:,:,:,96:]],dim=3).cuda()
                # if s==64:
                #     ft_current=torch.cat([ft_current[:,:,:,:32],ft_current[:,:,:,96:]],dim=3).cuda()
                # for j in range(11-args.autoregress_steps):
                #     loss_sum=0
                #     # pdb.set_trace()
                #     for i in range(args.autoregress_steps):
                #         f_rec_gd=ft_current[:,j+i].clone()
                #         u_t_rec_gd=ut_current[:,j+i].clone()
                #         trans_gd=ut_next[:,j+i].clone()
                #         if i==0:
                #             f_rec, f_latent = model_f(f_rec_gd)
                #             u_t_rec, u_t_latent = model_u(u_t_rec_gd)
                #             u_t_dt_rec, u_t_dt_latent = model_u(trans_gd)
                #         else:
                #             f_rec, f_latent = model_f(f_rec.deatch())
                #             u_t_rec, u_t_latent = model_u(u_t_rec.detach())
                #             u_t_dt_rec, u_t_dt_latent = model_u(trans.detach())
                #         if u_t_latent.shape[-1]<f_latent.shape[-1]:
                #             f_latent=f_latent[:,:u_t_latent.shape[-1]]+f_latent[:,u_t_latent.shape[-1]:]
                #         nn_u_t_dt_latent = model_trans(torch.cat((u_t_latent.view(-1,8,s_ob//4), f_latent.view(-1,8,s_ob//4)), 1))

                #         trans = model_u.up(nn_u_t_dt_latent)
                        
                #         loss1 = rel_error2(f_rec,f_rec_gd).mean()
                #         loss2 = rel_error(u_t_rec, u_t_rec_gd).mean()
                #         loss3 = rel_error(trans, trans_gd).mean()
                #         loss5 = rel_error(nn_u_t_dt_latent.view(-1,8*s_ob//4), u_t_dt_latent).mean()
                #         loss = loss1 + 0.5*loss2 + 1.5*loss3 + loss5
                #         loss_sum=loss_sum+loss
                #         # pdb.set_trace()
                #     loss_sum.backward()

                #     optimizer.step()



            
            train_loss.update(loss.item(), x_train.shape[0])
            train_loss_f_t_rec.update(loss1.item(), x_train.shape[0])
            train_loss_u_t_rec.update(loss2.item(), x_train.shape[0])
            train_loss_trans.update(loss3.item(), x_train.shape[0])
            train_loss_trans_latent.update(loss5.item(), x_train.shape[0])
        train_loss_list.append(loss.detach().cpu().numpy())
        logs['train_loss'].append(train_loss.avg)
        logs['train_loss_f_t_rec'].append(train_loss_f_t_rec.avg)
        logs['train_loss_u_t_rec'].append(train_loss_u_t_rec.avg)
        logs['train_loss_trans'].append(train_loss_trans.avg)
        logs['train_loss_trans_latent'].append(train_loss_trans_latent.avg)

        if epoch % 5== 0:
            scheduler.step()
        p.print(f"{epoch} epoch_end ", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
        if epoch%20==0:
            torch.save(model_f.state_dict(),args.exp_path+f"model_f-{epoch}.pt")
            torch.save(model_u.state_dict(),args.exp_path+f"model_u-{epoch}.pt")
            torch.save(model_trans.state_dict(),args.exp_path+f"model_trans-{epoch}.pt")
        # test
        model_f.eval()
        model_u.eval()
        model_trans.eval()

        test_loss=AverageMeter()
        test_loss_f_t_rec=AverageMeter()
        test_loss_u_t_rec=AverageMeter()
        test_loss_trans=AverageMeter()
        test_loss_trans_latent=AverageMeter()

        x_test= next(iter(test_loader))
        # x_test, y_test = x_test.cuda(), y_test.cuda()
        x_test=x_test[:,shuffle_index,:].clone()
        x_test=x_test.reshape(x_test.shape[0],10,3,128)
        x_test_reshape=x_test.reshape(x_test.shape[0]*10,3,128).clone() #[batch_size*10,3,128]
        # x_test, y_test = x_test.cuda(), y_test.cuda() #x:[512, 2, 128] y:[512, 1, 128]
        x_test=x_test_reshape[:,:2,:].cuda()
        y_test=x_test_reshape[:,2:3,:].cuda()
        if s_ob==64:
            ut_current=torch.cat([x_test[:,[0],:32],x_test[:,[0],96:]],dim=2).cuda()
            ut_next=torch.cat([y_test[:,[0],:32],y_test[:,[0],96:]],dim=2).cuda()
        else:
            ut_current=x_test[:,[0],:].cuda()
            ut_next=y_test[:,[0],:].cuda()
        if s==64:
            ft_current=torch.cat([y_test[:,[1],:32],y_test[:,[1],96:]],dim=2).cuda()
        else:
            ft_current=y_test[:,[0],:].cuda()
        
        f_rec, f_latent = model_f(ft_current)
        u_t_rec, u_t_latent = model_u(ut_current)
        u_t_dt_rec, u_t_dt_latent = model_u(ut_next)
        if u_t_latent.shape[-1]<f_latent.shape[-1]:
            f_latent=f_latent[:,:u_t_latent.shape[-1]]+f_latent[:,u_t_latent.shape[-1]:]
        nn_u_t_dt_latent = model_trans(torch.cat((u_t_latent.view(-1,8,s_ob//4), f_latent.view(-1,8,s_ob//4)), 1))

        trans = model_u.up(nn_u_t_dt_latent)
                        
        loss1 = rel_error2(f_rec,ft_current).mean()
        loss2 = rel_error(u_t_rec, ut_current).mean()
        loss3 = rel_error(trans, ut_next).mean()
        loss5 = rel_error(nn_u_t_dt_latent.view(-1,8*s_ob//4), u_t_dt_latent).mean()
        loss = loss1 + 0.5*loss2 + 2.5*loss3 + loss5

        test_loss.update(loss.item(), x_test.shape[0])
        test_loss_f_t_rec.update(loss1.item(), x_test.shape[0])
        test_loss_u_t_rec.update(loss2.item(), x_test.shape[0])
        test_loss_trans.update(loss3.item(), x_test.shape[0])
        test_loss_trans_latent.update(loss5.item(), x_test.shape[0])

        logs['test_loss'].append(test_loss.avg)
        logs['test_loss_f_t_rec'].append(test_loss_f_t_rec.avg)
        logs['test_loss_u_t_rec'].append(test_loss_u_t_rec.avg)
        logs['test_loss_trans'].append(test_loss_trans.avg)
        logs['test_loss_trans_latent'].append(test_loss_trans_latent.avg)
        test_loss_list.append(loss.detach().cpu().numpy())
        if epoch % 10 == 0:
            plt.figure()
            plt.plot(train_loss_list, label='Training Loss',color="red")
            plt.plot(test_loss_list, label='Test Loss',color="blue")

            # Adding labels and title
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')

            # Adding a legend
            plt.legend()

            # Save the plot to a file
            plt.savefig(args.exp_path+'/loss_plot.png')
        if epoch % 1 == 0:
            print('epoch {}/{} | train {:1.4f} = {:1.4f} + {:1.4f} + {:1.4f} + {:1.4f} | test {:1.4f} = {:1.4f} + {:1.4f} + {:1.4f} + {:1.4f}'.format(
                epoch,
                epochs,
                train_loss.avg, train_loss_f_t_rec.avg, train_loss_u_t_rec.avg, train_loss_trans.avg, train_loss_trans_latent.avg,
                test_loss.avg, test_loss_f_t_rec.avg, test_loss_u_t_rec.avg, test_loss_trans.avg, test_loss_trans_latent.avg
            ))
                # Plotting the loss curve
    plt.figure()
    plt.plot(train_loss_list, label='Training Loss',color="red")
    plt.plot(test_loss_list, label='Test Loss',color="blue")

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')

    # Adding a legend
    plt.legend()

    # Save the plot to a file
    plt.savefig(args.exp_path+'/loss_plot.png')

    # Display the plot (optional)
    plt.show()
    
    print("Learing solution operator is finished!")        
    torch.save({
            'model_f_state_dict': model_f.state_dict(),
            'model_u_state_dict': model_u.state_dict(),
            'model_trans_state_dict': model_trans.state_dict(),
            'logs' :logs,
            }, 'logs/burgers_operator_model')

