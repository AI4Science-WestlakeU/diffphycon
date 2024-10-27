import scipy.io
import numpy as np
import h5py
import pdb
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Data
from typing import Tuple
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

from IPython import embed

PDE_PATH = "data/"
BURGERS_PATH = ""

class Burgers(Dataset):
    '''
    x_pos: spatial grids
    x: u in Burgers' equation
    y: u_1, u_2, ..., u_t, (Nt, Nx, 1)
    f: external force, (Nt, Nx, 1)
    edge_index: not sure.

    '''
    def __init__(
        self,
        dataset="burgers",
        input_steps=1,
        output_steps=10,
        time_interval=1,
        is_y_diff=False,
        split="train",
        transform=None,
        pre_transform=None,
        verbose=False,
        root_path = None, 
        **kwargs
    ):
        self.dataset = dataset
        self.dirname = BURGERS_PATH
        self.root = PDE_PATH if root_path is None else root_path
        
        self.nx = 128
        self.nt_total, self.nx_total = kwargs['nt_total'], 128
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_interval = time_interval
        self.is_y_diff = is_y_diff
        self.split = split
        # assert self.split in ["train", "valid", "test"]
        assert self.split in ["train", "test"]
        self.verbose = verbose

        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1

        self.original_shape = (self.nx,)
        
        if (self.nt_total, self.nx_total) == (11, 128):
            path = os.path.join(self.root, self.dirname) + f'{dataset}_{self.split}.h5'
        else:
            path = os.path.join(self.root, self.dirname) + f'{dataset}_{self.split}_nt_{self.nt_total}_nx_{self.nx_total}.h5'
        print("Load dataset {}".format(path))
        self.time_stamps = self.nt_total
        base_resolution=[self.nt_total, self.nx]
        super_resolution=[self.nt_total, self.nx_total]

        self.dataset_cache = HDF5Dataset(path, mode=self.split, base_resolution=base_resolution, super_resolution=super_resolution)
        self.n_simu = len(self.dataset_cache)
        self.time_stamps_effective = (self.time_stamps - self.t_cushion_input - self.t_cushion_output + self.time_interval) // self.time_interval
        super(Burgers, self).__init__(self.root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [self.root + self.dirname]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.dirname)

    @property
    def processed_file_names(self):
        return []

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _process(self):
        import warnings
        from typing import Any, List
        from torch_geometric.data.makedirs import makedirs
        def _repr(obj: Any) -> str:
            if obj is None:
                return 'None'
                return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

        def files_exist(files: List[str]) -> bool:
            # NOTE: We return `False` in case `files` is empty, leading to a
            # re-processing of files on every instantiation.
            return len(files) != 0 and all([os.path.exists(f) for f in files])

        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_transform):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"sure to delete '{self.processed_dir}' first")

        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in the "
                "pre-processed version of this dataset. If you want to make "
                "use of another pre-fitering technique, make sure to delete "
                "'{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        if not os.path.isfile(path):
            torch.save(_repr(self.pre_filter), path)

    def get_edge_index(self):
        edge_index_filename = os.path.join(self.processed_dir, "edge_index.p")
        if os.path.isfile(edge_index_filename):
            edge_index = pickle.load(open(edge_index_filename, "rb"))
            return edge_index
        rows, cols = (*self.original_shape, 1)
        cube = np.arange(rows * cols).reshape(rows, cols)
        edge_list = []
        for i in range(rows):
            for j in range(cols):
                if i + 1 < rows: #and cube[i, j] not in velo_invalid_ids and cube[i+1, j] not in velo_invalid_ids:
                    edge_list.append([cube[i, j], cube[i+1, j]])
                    edge_list.append([cube[i+1, j], cube[i, j]])
                if j + 1 < cols: #and cube[i, j]: #not in velo_invalid_ids and cube[i, j+1] not in velo_invalid_ids:
                    edge_list.append([cube[i, j], cube[i, j+1]])
                    edge_list.append([cube[i, j+1], cube[i, j]])
        edge_index = torch.LongTensor(edge_list).T
        pickle.dump(edge_index, open(edge_index_filename, "wb"))
        return edge_index


    def process(self):
        pass

    def len(self):
        return self.time_stamps_effective * self.n_simu

    def get(self, idx):
        # assert self.time_interval == 1
        sim_id, time_id = divmod(idx, self.time_stamps_effective)
        _, data_traj, force, x_pos = self.dataset_cache[sim_id]
        if self.verbose:
            print(f"sim_id: {sim_id}   time_id: {time_id}   input: ({time_id * self.time_interval + self.t_cushion_input -self.input_steps * self.time_interval}, {time_id * self.time_interval + self.t_cushion_input})  output: ({time_id * self.time_interval + self.t_cushion_input}, {time_id * self.time_interval + self.t_cushion_input + self.output_steps * self.time_interval})")
        force = torch.FloatTensor(force)
        x_dens = torch.FloatTensor(np.stack([data_traj[time_id * self.time_interval + self.t_cushion_input + j] for j in range(-self.input_steps * self.time_interval, 0, self.time_interval)], -1))
        y_dens = torch.FloatTensor(np.stack([data_traj[time_id * self.time_interval + self.t_cushion_input + j] for j in range(0, self.output_steps * self.time_interval, self.time_interval)], -1))  # [1, rows, cols, output_steps, 1]
        edge_index = self.get_edge_index()
        x_bdd = torch.ones(x_dens.shape[0])
        x_bdd[0] = 0
        x_bdd[-1] = 0
        x_pos = torch.FloatTensor(x_pos)[...,None]

        # NOTE: it is here that an extra feature dimension of f is inserted.
        data = Data(
            x=x_dens.reshape(-1, *x_dens.shape[-1:], 1).clone(),       # [number_nodes: 128, input_steps, 1]
            x_pos=x_pos,  # [number_nodes: 128, 1]
            x_bdd=x_bdd[...,None],
            xfaces=torch.tensor([]),
            y=y_dens.reshape(-1, *y_dens.shape[-1:], 1).clone(),       # [number_nodes: 128, output_steps, 1]
            f=force[...,None], # [number_nodes: 128, 1]
            edge_index=edge_index,
            original_shape=self.original_shape,
            compute_func=(0, None),
            dataset=self.dataset,
        )
        update_edge_attr_1d(data)
        return data

def update_edge_attr_1d(data):
    edge_attr = data.x_pos[data.edge_index[0]] - data.x_pos[data.edge_index[1]]
    assert len(edge_attr.shape) == 2 and edge_attr.shape[1] == 1
    id_he = torch.where((edge_attr.squeeze(-1) + 1/129).abs() < 1e-7)[0]
    id_eh = torch.where((edge_attr.squeeze(-1) - 1/129).abs() < 1e-7)[0]
    edge_attr[id_he, 0] = 1/129
    edge_attr[id_eh, 0] = -1/129
    return data



class HDF5Dataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False,
                 uniform_sample: int=-1,
                ) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (11, 128) if base_resolution is None else base_resolution
        self.super_resolution = (11, 128) if super_resolution is None else super_resolution
        self.uniform_sample = uniform_sample
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'
        self.dataset_f = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}_f'

        ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
        ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        # assert (ratio_nt.is_integer())
        # assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        self.nt = self.data[self.dataset_base].attrs['nt']
        self.dt = self.data[self.dataset_base].attrs['dt']
        self.dx = self.data[self.dataset_base].attrs['dx']
        self.x = self.data[self.dataset_base].attrs['x']
        self.tmin = self.data[self.dataset_base].attrs['tmin']
        self.tmax = self.data[self.dataset_base].attrs['tmax']
        self.x_ori = self.data['pde_11-128'].attrs['x']

        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data


    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
        """
        # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
        u_super = self.data[self.dataset_super][idx][::self.ratio_nt][:, :, None]
        u_base = u_super[:, ::self.ratio_nx, :]
        force = self.data[self.dataset_f][idx]
        
        return u_base, u_super, force, self.x
    
    def get(self, idx):
        return self.__getitem__(idx)
    
    def len(self):
        return self.__len__()



if __name__ == "__main__":
    dataset = Burgers(
        dataset="burgers",
        input_steps=1,
        output_steps=10,
        time_interval=1,
        is_y_diff=False,
        split="train",
        transform=None,
        pre_transform=None,
        verbose=False, 
        
    )

    idx=1
    print(len(dataset))
    print(dataset[idx].x.shape)
    print(dataset[idx].x_pos.shape)
    print(dataset[idx].f.shape)
    print(dataset[idx].y.shape)
    print(dataset[idx].edge_index)
