import numpy as np
import os
import sys
from datetime import datetime
from copy import deepcopy
from termcolor import colored
import torch
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib import colors
from collections import deque
import pdb
from torch.utils.data import Sampler
from torch.autograd import Variable
from numbers import Number
import pickle

from IPython import embed

COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine",
             "b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

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

def compute_orthonormal(boundary):                        
    r"""
    Precompute orthonormal vectors on boundary nodes
    Args:
        boudary: boundary, shape=[number of boundary nodes, 2]
    """

    rolled_boundary = torch.roll(boundary, -1, 0)
    length = (boundary - rolled_boundary).norm(dim=1)       ## distance between two adjacent nodes
    tx = (boundary[:, 0] - rolled_boundary[:, 0])/length    ## x tangent
    ty = (boundary[:, 1] - rolled_boundary[:, 1])/length    ## y tangent
    nx = ty
    ny = -tx                                                 ## normal vector
    center = (boundary + rolled_boundary)/2
    return length, nx, ny, center


def linear_transform(pressure, center):
    r"""
    Compute magnitude of pressure on boundary nodes
    Args:
        pressure: pressure, shape=[width, height]
        center: midpoints of adjacent nodes in boundary, shape=[#number of bounday points, 2]
    """
    flip_pressure = pressure
    device = flip_pressure.device
    n, m = flip_pressure.shape
    n = n 
    m = m 
    num_bound = center.shape[0]
    
    p_5 = torch.tensor([0.5], device=device).repeat(num_bound)
    x = torch.minimum(torch.maximum(center[:, 0], p_5), torch.tensor([n-1.5], device=device).repeat(num_bound))
    x_inds = torch.minimum(x.type(torch.int32), torch.tensor([n-2], device=device).repeat(num_bound))
    fs = x - x_inds
    
    y = torch.minimum(torch.maximum(center[:, 1], p_5), torch.tensor([m-1.5], device=device).repeat(num_bound))
    y_inds = torch.minimum(y.type(torch.int32), torch.tensor([m-2], device=device).repeat(num_bound))
    ft = y - y_inds
    
    s_mask = (fs==0)
    t_mask = (ft==0)
    bd_mask = s_mask*t_mask
    in_mask = torch.logical_not(bd_mask)
    raw_force = torch.zeros(num_bound, dtype=torch.float32, device=device)
    raw_force[bd_mask] = flip_pressure[y_inds[bd_mask], x_inds[bd_mask]]
    
    t_weight = torch.stack([ft[in_mask], 1-ft[in_mask]], 0)
    #column
    a_pipj = flip_pressure[y_inds[in_mask]+1, x_inds[in_mask]+1]
    a_pij  = flip_pressure[y_inds[in_mask], x_inds[in_mask]+1]
    a_rowp1 = torch.stack([a_pipj, a_pij], 0)
    sum_a_rowp1 = torch.sum(t_weight * a_rowp1, 0)    
    #row
    a_ipj = flip_pressure[y_inds[in_mask]+1, x_inds[in_mask]]
    a_ij  = flip_pressure[y_inds[in_mask], x_inds[in_mask]]
    a_row = torch.stack([a_ipj, a_ij], 0)
    sum_a_row = torch.sum(t_weight * a_row, 0)
    
    s_weight = torch.stack([fs[in_mask], 1-fs[in_mask]], 0)
    sum_two_rows = torch.stack([sum_a_rowp1, sum_a_row], 0)
    raw_force[in_mask] = torch.sum(s_weight*sum_two_rows, 0)
        
    return raw_force
    

def compute_pressForce(pressure, boundary, res =64):
    r"""
    Compute pressure of nodes along orthonormal vectors
    Args:
        pressure: predicted pressure of model, shape=[res-2, res-2]
        boundary: shape=[#number of boundary nodes, 2]
    """
    length, nx, ny, cen = compute_orthonormal(boundary)
    pdl = linear_transform(pressure, cen)
    pdl = pdl * length
    return torch.sum(pdl*nx), torch.sum(pdl*ny)


def make_dir(filename):
    """Make directory using filename if the directory does not exist"""
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


def get_item_1d(data, target):
    """
    Get the 1d item suitable for diffusion.

    Args:
        data: PyG data class
        target: choose from "x" and "y".

    Returns:
        x: has shape of [B, n_steps, n_bodies*feature_size], suitable for diffusion model
    """
    x = data[target]  # [B*n_bodies, n_steps, feature_size]
    batch_size = len(data.dyn_dims)
    assert x.shape[0] % batch_size == 0
    n_bodies = x.shape[0] // batch_size
    n_steps, feature_size = x.shape[1:]
    x = x.reshape(-1, n_bodies, n_steps, feature_size) / 200.  # [B, n_bodies, n_steps, feature_size]
    x = x.permute(0, 2, 1, 3)  # [B, n_steps, n_bodies, feature_size]
    x = torch.flatten(x, -2, -1)  # [B, n_steps, n_bodies*feature_size]
    return x
def get_item_1d_for_solver(data, target):
    """
    Get the 1d item suitable for diffusion.

    Args:
        data: PyG data class
        target: choose from "x" and "y".

    Returns:
        x: has shape of [B, n_steps, n_bodies*feature_size], suitable for diffusion model
    """
    x = data[target]  # [B*n_bodies, n_steps, feature_size]
    batch_size = len(data.dyn_dims)
    assert x.shape[0] % batch_size == 0
    n_bodies = x.shape[0] // batch_size
    n_steps, feature_size = x.shape[1:]
    x = x.reshape(-1, n_bodies, n_steps, feature_size)  # [B, n_bodies, n_steps, feature_size]
    x = x.permute(0, 2, 1, 3)  # [B, n_steps, n_bodies, feature_size]
    x = torch.flatten(x, -2, -1)  # [B, n_steps, n_bodies*feature_size]
    return x
def convert_softbd2hard(softbd):
    r"""
    Convert soft boundary whose value range between [0, 1] to solid boundary mask
    Args:
        softbd: grid with values ranging between 0 and 1
    Output:
        hardbd: grid with binary values, where 1 incidates existance of boundary
    """
    ones = softbd > 0.5
    zeros = softbd <= 0.5
    hardbd = torch.zeros(softbd.shape, device=softbd.device)
    hardbd[ones] = 1
    hardbd[zeros] = 0
    return hardbd

def find_isolated_points(grid):
    isolated_points = []
    rows, cols = grid.shape

    # Define possible neighboring cell offsets
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            current_cell = grid[r, c]
            if current_cell == 1:
                is_isolated = True
                if (r in [0, rows-1]) or (c in [0, cols-1]):
                    if (r in [0, rows-1]) and (c in [0, cols-1]):
                        n = 3 
                    else:
                        n = 5 
                else:
                    n = 8 

                j = 0
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc

                    # Check if the neighboring cell is inside the grid
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_cell = grid[nr, nc]
                        # Check if the neighboring cell has the same value
                        if current_cell == neighbor_cell:
                            continue
                        else:
                            j += 1
                        if n == 8 and j >= 7:
                            isolated_points.append((r, c))
                            break
                        elif n == 5 and j >= 4:
                            isolated_points.append((r, c))
                            break
                        elif n == 3 and j >= 2:
                            isolated_points.append((r, c))
                            break

    return isolated_points

def filter_isolated_points(hard_boundary):
    iso_points = find_isolated_points(hard_boundary)
    while (len(iso_points) != 0):
        index_tensor = torch.tensor(iso_points, dtype=torch.long, device=hard_boundary.device).t()
        values = torch.zeros(index_tensor.shape[1], device=hard_boundary.device)

        # Update the tensor with new values according to the indices
        hard_boundary.index_put_((index_tensor[0], index_tensor[1]), values)
        iso_points = find_isolated_points(hard_boundary)

    return hard_boundary

def find_clusters(hard_boundary):
    # Find the non-zero grid cells
    non_zero_cells = np.argwhere(hard_boundary.detach().cpu().numpy() != 0)

    # Create the DBSCAN clustering model
    dbscan = DBSCAN(eps=1.5, min_samples=2)

    # Fit the model to the non-zero grid cells
    labels = dbscan.fit_predict(non_zero_cells)

    clustered_grid = np.zeros_like(hard_boundary)

    for cell, label in zip(non_zero_cells, labels):
        clustered_grid[tuple(cell)] = label + 1
    return clustered_grid


def filter_filename(dirname, include=[], exclude=[], array_id=None):
    """Filter filename in a directory"""
    def get_array_id(filename):
        array_id = filename.split("_")[-2]
        try:
            array_id = eval(array_id)
        except:
            pass
        return array_id
    filename_collect = []
    if array_id is None:
        filename_cand = [filename for filename in os.listdir(dirname)]
    else:
        filename_cand = [filename for filename in os.listdir(dirname) if get_array_id(filename) == array_id]
    
    if not isinstance(include, list):
        include = [include]
    if not isinstance(exclude, list):
        exclude = [exclude]
    
    for filename in filename_cand:
        is_in = True
        for element in include:
            if element not in filename:
                is_in = False
                break
        for element in exclude:
            if element in filename:
                is_in = False
                break
        if is_in:
            filename_collect.append(filename)
    return filename_collect


def find_filtered_clusteres(hard_boundary, is_plot_clusters=False):
    clustered_grid = find_clusters(hard_boundary)
    # clustered_grid
    
    if is_plot_clusters:
        fig, ax = plt.subplots(figsize=(4,4), ncols=1)
        mappable0 = ax.imshow(clustered_grid, cmap='viridis',
                                 aspect='auto'
                                 ) 
        fig.colorbar(mappable0, ax=ax)
        fig.tight_layout()    
        plt.show()

    hard_boundary = filter_isolated_points(hard_boundary)

    rd_clustered_grid = find_clusters(hard_boundary)

    if is_plot_clusters:
        fig, ax = plt.subplots(figsize=(4,4), ncols=1)
        mappable0 = ax.imshow(rd_clustered_grid, cmap='viridis',
                                 aspect='auto'
                                 )
        fig.colorbar(mappable0, ax=ax)
        fig.tight_layout()    
        plt.show()
        
    return rd_clustered_grid

from collections import deque

def find_starting_point(grid):
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 1:
                return i, j
    return None

def is_valid_move(x, y, grid):
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 1

def is_boundary(x, y, grid):
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if not is_valid_move(nx, ny, grid):
            return True
    return False

def bfs_boundary_detection_without_intersection(start_x, start_y, grid):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    boundary = []
    queue = deque([(start_x, start_y)])

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while queue:
        x, y = queue.popleft()
        if visited[x][y]:
            continue

        visited[x][y] = True
        if is_boundary(x, y, grid):
            boundary.append((x, y))

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny, grid) and not visited[nx][ny]:
                queue.append((nx, ny))

    return boundary

# Identify the non-dominated solutions
def find_pareto_frontier(objective_values):
    pareto_frontier1 = []
    for i, values1 in enumerate(objective_values):
        dominated = False
        for j, values2 in enumerate(objective_values):
            if all(values1 <= values2) and any(values1 < values2):
                dominated = True
                break
        if not dominated:
            pareto_frontier1.append(values1)
        
    np_pareto_frontier1 = np.array(pareto_frontier1)
    np_pareto_frontier1 = np_pareto_frontier1[np_pareto_frontier1[:, 0].argsort()]            
    flip_objective_values = np.copy(objective_values)
    flip_objective_values[:,0] = -flip_objective_values[:,0] 
    pareto_frontier2 = []
    for i, values1 in enumerate(flip_objective_values):
        dominated = False
        for j, values2 in enumerate(flip_objective_values):
            if all(values2 <= values1) and any(values2 < values1):
                dominated = True
                break
        if not dominated:
            cpvalues1 = np.copy(values1)
            cpvalues1[0] = -cpvalues1[0]
            pareto_frontier2.append(cpvalues1)
    np_pareto_frontier2 = np.array(pareto_frontier2)
    np_pareto_frontier2 = np_pareto_frontier2[np.flip(np_pareto_frontier2[:, 0].argsort())]

    pareto_frontier3 = []
    for i, values1 in enumerate(objective_values):
        dominated = False
        for j, values2 in enumerate(objective_values):
            if all(values2 <= values1) and any(values2 < values1):
                dominated = True
                break
        if not dominated:
            pareto_frontier3.append(values1)
    np_pareto_frontier3 = np.array(pareto_frontier3)
    np_pareto_frontier3 = np_pareto_frontier3[np.flip(np_pareto_frontier3[:, 0].argsort())]

    flip_objective_values = np.copy(objective_values)
    flip_objective_values[:,0] = -flip_objective_values[:,0] 
    pareto_frontier4 = []
    for i, values1 in enumerate(flip_objective_values):
        dominated = False
        for j, values2 in enumerate(flip_objective_values):
            if all(values1 <= values2) and any(values1 < values2):
                dominated = True
                break
        if not dominated:
            cpvalues1 = np.copy(values1)
            cpvalues1[0] = -cpvalues1[0]
            pareto_frontier4.append(cpvalues1)
    np_pareto_frontier4 = np.array(pareto_frontier4)
    np_pareto_frontier4 = np_pareto_frontier4[np_pareto_frontier4[:, 0].argsort()]

    org_pareto_frontier = np.concatenate([np_pareto_frontier1, np_pareto_frontier2, np_pareto_frontier3, np_pareto_frontier4], axis=0)
    index_hash = 100*org_pareto_frontier[:,0] + org_pareto_frontier[:,1]
    unique_index = np.unique(index_hash, return_index=True)[1]
    pareto_frontier = [org_pareto_frontier[index] for index in sorted(unique_index)]
    new_pareto_frontier = [pt for pt in pareto_frontier]
    put_values = []
    put_indices = []
    for i in range(len(pareto_frontier)):
        dir_x = pareto_frontier[i%len(pareto_frontier)][0] - pareto_frontier[(i+1)%len(pareto_frontier)][0]
        dir_y = pareto_frontier[i%len(pareto_frontier)][1] - pareto_frontier[(i+1)%len(pareto_frontier)][1]
        if dir_y == 0:
            if dir_x > 1:
                for k in range(1, int(dir_x)):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0] - k, pareto_frontier[i%len(pareto_frontier)][1]])
                    put_indices.append(i+1)
            elif dir_x < -1:
                for k in range(1, -int(dir_x), 1):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0] + k, pareto_frontier[i%len(pareto_frontier)][1]])
                    put_indices.append(i+1)
        elif dir_x == 0:
            if dir_y > 1:
                for k in range(1, int(dir_y)):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0], pareto_frontier[i%len(pareto_frontier)][1] - k])
                    put_indices.append(i+1)
            elif dir_y < -1:
                for k in range(1, -int(dir_y), 1):
                    put_values.append([pareto_frontier[i%len(pareto_frontier)][0], pareto_frontier[i%len(pareto_frontier)][1] + k])
                    put_indices.append(i+1)
    
    if len(put_indices) > 0:
        final_pareto = np.insert(np.array(pareto_frontier), put_indices, put_values, axis=0)
        return final_pareto
    
    return np.array(pareto_frontier)

def find_cluster_boundary(np_clustered_grid, is_boundary_plot=False):
    num_cluster = int(np_clustered_grid.max())
    boundaries = []
    for i in range(num_cluster):
        cp_clustered_grid = np_clustered_grid.copy()
        single_clustered_grid = np.where(cp_clustered_grid==i+1, 1, 0).tolist()
        starting_point = find_starting_point(single_clustered_grid)
        boundary = bfs_boundary_detection_without_intersection(starting_point[0], starting_point[1], single_clustered_grid)

        np_boundary = np.array(boundary)
        if is_boundary_plot:
            plt.scatter(np_boundary[:,1], np_boundary[:,0])
            plt.show()

        objective_values = np_boundary
        pareto_frontier = find_pareto_frontier(objective_values)
        pareto_frontier = np.fliplr(pareto_frontier)[::-1]
        boundaries.append(pareto_frontier)

        # Plot the solutions and the Pareto frontier
        if is_boundary_plot:
            plt.scatter(objective_values[:, 1], objective_values[:, 0], label='Solid Points', marker='o')
            plt.scatter(pareto_frontier[:, 0], pareto_frontier[:, 1], label='Solid Boundary', marker='x', color='red')
            plt.plot(pareto_frontier[:, 0], pareto_frontier[:, 1])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()
            print(pareto_frontier)
        np.unique(pareto_frontier, axis=0, return_counts=True)
        
    return boundaries


def compute_binary_pressForce(pressure, np_clustered_grid, is_boundary_plot=False):
    r"""
    Compute force on binary boundary mask
    Args:
        pressure: 2d tensor, each element represents pressure on a cell.
        np_clustered_grid: Numpy mask whose cells represent multiple solids. The shape is same as pressure.
        Values of cells are integers and cells with a same value belong to a same solid. 
        This can be either bounary mask and derived from find_filtered_clusteres function.         
    """
    # Compute the boundary of all solids in binary_boundary_mask
    final_boundaries = find_cluster_boundary(np_clustered_grid, is_boundary_plot=is_boundary_plot)
    # Compute force of boundary of each multiple solid
    pressures = {}
    for i in range(len(final_boundaries)):
        boundary = final_boundaries[i]
        # Compute force
        pressures[i+1] = compute_pressForce(pressure, torch.tensor(boundary.copy(), device=pressure.device)+0.5)
    return pressures

def reconstruct_boundary(binary_mask, bd_offset, res=64):
    r"""
    Restore boundary points from binary mask and boundary offset
    Args:
        binary_mask: 2d torch tensor, represented by binary values. Expected shape is [res - 2, res - 2].
        bd_offset: 3d torch tensor, grid structure each of whose cells is represented by 2d vector.
        Expected shape is [res - 2, res - 2, 2]
    Limitation:
        Multiple boundaries are not out of the scope at this moment.
        The number of points of the ouput is not consistent with the shape of original boundary (= 40)
    """    
    # Find boundary of object in boundary mask
    np_binary_mask = binary_mask.cpu().numpy()
    boundary_bd = find_cluster_boundary(np_binary_mask)[0]

    # Retrieve offset vectors on boundary_bd and boundary value
    np_offset = bd_offset.cpu().numpy()
    new_np_offset = np_offset[boundary_bd[:,1], boundary_bd[:,0], :]
    restored_boundary = (boundary_bd + 0.5) + new_np_offset

    return restored_boundary


m = 64; n = 64
maxnum = 100

def discretize_boundary(boundary, m, n):
    # output left-bottom corner indices
    assert boundary.shape[1] == 2
    num_bound = boundary.shape[0]
    device = boundary.device
    p_5 = torch.tensor([0.5], device=device).repeat(num_bound)
    x = torch.minimum(torch.maximum(boundary[:, 0], p_5), torch.tensor([n-1.5], device=device).repeat(num_bound))
    x_inds = torch.minimum(x.type(torch.int32), torch.tensor([n-2], device=device).repeat(num_bound))

    y = torch.minimum(torch.maximum(boundary[:, 1], p_5), torch.tensor([m-1.5], device=device).repeat(num_bound))
    y_inds = torch.minimum(y.type(torch.int32), torch.tensor([m-2], device=device).repeat(num_bound))
    return x_inds, y_inds

def find_orthogonal_line(A, B, C, x0, y0):
    m1 = torch.empty((C.shape[0],), device=C.device)
    m1[B==0] = float('inf')
    m1[B!=0] = (-A/B)[B!=0]

    m2 = torch.empty((C.shape[0],), device=C.device)
    m2[m1==float('inf')] = 0
    m2[m1!=float('inf')] = (-1 / m1)[m1!=float('inf')]

    b2 = y0 - m2 * x0  # The y-intercept of L2.

    # Return the coefficients A, B, C of the line L2 (Ax + By - C = 0)
    return m2, -1, b2

def edge_cells(polygon):
    num_vertices = len(polygon)
    edges = []
    for i in range(num_vertices):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % num_vertices]
        edge = sorted([v1, v2], key=lambda x: x[1])
        edges.append(edge)
    return edges

def find_cells_inside_curve(polygon, grid_shape):
    def horizontal_intersection(x1, y1, x2, y2, y):
        return x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    edges = edge_cells(polygon) # sorted by y
    grid = np.zeros(grid_shape, dtype=np.uint8)
    height, width = grid.shape
    
    for y in range(height):
        intersections = []
        for edge in edges:
            y1, y2 = edge[0][1], edge[1][1]
            if y1 < y <= y2:
                x = horizontal_intersection(*edge[0], *edge[1], y)
                intersections.append(x)
        intersections.sort()
        for i in range(0, len(intersections), 2):
            x_start, x_end = int(np.ceil(intersections[i])), int(np.floor(intersections[i + 1]))
            grid[y, x_start : x_end + 1] = 1

    return grid


def bresenham_line(x0, y0, x1, y1):  
    """Bresenham's Line Algorithm to produce a list of grid cells that form a line between two points."""  
    cells = []  
    dx = abs(x1 - x0)  
    dy = -abs(y1 - y0)  
    sx = 1 if x0 < x1 else -1  
    sy = 1 if y0 < y1 else -1  
    err = dx + dy  
    while True:  
        cells.append((x0, y0))  
        if abs(x1-x0)<1 and abs(y1-y0)<1:  
            cells.append((x1, y1))  
            break  
        e2 = 2 * err  
        if e2 >= dy:  
            err += dy  
            x0 += sx  
        if e2 <= dx:  
            err += dx  
            y0 += sy  
    return cells  


def find_cells_boundary(polygon, grid_shape):  
    num_vertices = len(polygon)  
    edges = []  
    for i in range(num_vertices):  
        v1 = polygon[i]  
        v2 = polygon[(i + 1) % num_vertices]  
        edges.append((v1, v2))  
  
    grid = np.zeros(grid_shape, dtype=np.uint8)  
    for edge in edges:  
        cells = bresenham_line(*edge[0], *edge[1])  
        for x, y in cells:  
            if 0 <= x < grid_shape[1] and 0 <= y < grid_shape[0]:
                grid[int(y), int(x)] = 1  
  
    return grid 


def update_static_masks(torch_con_boundary, n_p=40, res=64):
    x_inds, y_inds = discretize_boundary(torch_con_boundary, m=res, n=res)
    pointy_hash = maxnum*x_inds[n_p//2] + y_inds[n_p//2] # Unique identifier

    indices = torch.stack((maxnum*x_inds,y_inds), 0)
    sum_indices = indices.sum(0)
    ind_unique = torch.unique(sum_indices, sorted=True) #, return_inverse=True) # sort according to the first row and then the second row
    x_idx = (torch.cat([(sum_indices==ind_u).nonzero()[0] for ind_u in ind_unique])).sort()[0] # find and sort every ind_u's fist index in sum_indices

    repeat_sum_indices = torch.tile(sum_indices, (ind_unique.shape[0],1))
    repeat_ind_unique = torch.tile(sum_indices[x_idx].reshape(ind_unique.shape[0], 1), (1, sum_indices.shape[0]))
    org_mask = (repeat_ind_unique == repeat_sum_indices) # find all indices of ind_u in sum_indices
    fatted_mask = torch.roll(org_mask, 1, 1) + torch.roll(org_mask, -1, 1)

    relvecs = [] # relevant vectors
    base_pts = [] # center points of the intersecting grid
    base_nums = []
    for bdpt in range(sum_indices[x_idx].shape[0]):
        if pointy_hash == sum_indices[x_idx][bdpt]: # n_p//2
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5 
            base_pts.append(base_pt)
            relvec = torch_con_boundary[n_p//2] - base_pt 
            relvecs.append(relvec)
        elif torch.sum(org_mask[bdpt]) >= 4: # more than 4 points in one grid
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            base_pts.append(base_pt)
            relvec = torch_con_boundary[org_mask[bdpt]] - base_pt.repeat(torch_con_boundary[org_mask[bdpt]].shape[0], 1)
            ind = torch.argmin(torch.norm(relvec, dim=1)) # get the index of the closest point
            relvecs.append(relvec[ind])
        elif torch.sum(fatted_mask[bdpt] * torch.logical_not(org_mask[bdpt])) > 2: # this grid's relevant grids in sum_indices is not equal to itself
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            relvec = torch_con_boundary[org_mask[bdpt]] - base_pt.repeat(torch_con_boundary[org_mask[bdpt]].shape[0], 1)
            if len(relvec.shape) == 2:
                import pdb
    
                relvecs.append(relvec[-1])
                base_pts.append(base_pt)
            else:
                relvecs.append(relvec)
                base_pts.append(base_pt)
        elif torch.sum(org_mask[bdpt]) == 1: # only one point in this grid
            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            base_pts.append(base_pt)
            relvec = torch_con_boundary[org_mask[bdpt]] - base_pt
            relvecs.append(relvec[0])
        else:    
            if fatted_mask[bdpt][0] and fatted_mask[bdpt][-1]:
                rollnum = 1
                for _ in range(0, 100):
                    temprole = torch.roll(fatted_mask[bdpt], rollnum, 0)
                    if temprole[0] and temprole[-1]:
                        rollnum += 1    
                    else:
                        break
                x_pts = torch.roll(torch_con_boundary[fatted_mask[bdpt]], rollnum, 0)            
            else:
                x_pts = torch_con_boundary[fatted_mask[bdpt]]

            bd_points = torch.cat([x_pts[0:1], x_pts[1:-1].repeat(1, 2).reshape(-1,2), x_pts[-1:]], dim=0)
            dire_vec = bd_points[0::2] - bd_points[1::2] # bounary direction vector
            const = bd_points[0::2, 1] - bd_points[0::2, 0] * dire_vec[:,1]/dire_vec[:,0]

            base_pt = torch.stack([x_inds[org_mask[bdpt]][0], y_inds[org_mask[bdpt]][0]]) + 0.5
            base_pts.append(base_pt)
            base_points = base_pt.repeat(const.shape[0], 1)
            slope = dire_vec[:,1]/dire_vec[:,0]
            ax, by, con = find_orthogonal_line(slope, -torch.ones((const.shape[0],), device=torch_con_boundary.device), const, base_points[:,0], base_points[:,1]) 

            al = -ax/by
            bl = con
            cl = dire_vec[:,1]/dire_vec[:,0]
            dl = const

            intersection = torch.stack([(dl - bl)/(al - cl), (al*dl - bl*cl)/(al - cl)]).t() # intersection of orthogonal line and boundary line

            relvec = intersection - torch.tile(base_pt, (intersection.shape[0], 1))
            relvecs.append(relvec.sum(0)/relvec.shape[0])

    ### Check number of offset vectors is same as that of boundary cells of solid
    assert len(base_pts) == sum_indices[x_idx].shape[0]
    bd_offset = torch.stack(relvecs)
    left_idx = torch.argmin(torch_con_boundary[:, 0])
    upper_condition = torch_con_boundary[:, 1]  > torch_con_boundary[left_idx, 1]    
    upper_inds = torch.where(upper_condition)[0]  
    modified_y_inds = y_inds
    modified_y_inds[upper_inds] = y_inds[upper_inds] + 1

    offset_grid_bound = torch.zeros((res - 2, res - 2, 2), device=torch_con_boundary.device)
    offset_grid_bound[x_inds, y_inds] = torch.tensor([1, 1], dtype=torch.float32, device=torch_con_boundary.device)
    offset_grid_bound = offset_grid_bound.transpose(1,0)

    offset_grid = find_cells_inside_curve(torch.stack((x_inds, modified_y_inds), -1).detach().cpu().tolist(), (res - 2, res - 2))
    inner_solid_mask = np.copy(offset_grid)
    offset_grid = offset_grid.reshape(res - 2, res - 2, 1)
    offset_grid = np.concatenate([offset_grid, offset_grid], -1)

    offset_union = offset_grid_bound + torch.tensor(offset_grid, device=torch_con_boundary.device)
    offset_union[(offset_union.sum(-1) > 2),:] = torch.tensor([1, 1], dtype=torch.float32, device=torch_con_boundary.device)
    offset_union.index_put_((y_inds[x_idx], x_inds[x_idx]), bd_offset)    
    # new version
    grid_bound = find_cells_boundary(torch.stack((x_inds+0.5, y_inds+0.5), -1).detach().cpu().tolist(), (res - 2, res - 2))
    # grid_bound = find_cells_boundary(torch_con_boundary.cpu().tolist(), (res - 2, res - 2))
    grid_bound = torch.tensor(grid_bound, device=torch_con_boundary.device)
    # union = grid_bound 
    union = grid_bound + torch.tensor(inner_solid_mask, device=torch_con_boundary.device)

    union[union == 2] = 1

    updated_solid_mask = union
    
    return updated_solid_mask, offset_union #updated_offset_mask

def update_bdfeature(reconstructed_boundary):
    upd_solid_mask, upd_solid_offset = update_static_masks(reconstructed_boundary)
    torch_batch_mask = torch.where(upd_solid_mask==1, False, True).clone().flatten()
    upd_solid_mask = upd_solid_mask[...,None]
    static_feature = torch.cat((upd_solid_mask, upd_solid_offset), -1)
    multi_static_feat = torch.stack([static_feature for _ in range(4)], -2).reshape(-1,4,3)
    return multi_static_feat, torch_batch_mask

def groupby_add_keys(df, by, add_keys, other_keys=None, mode="mean"):
    """
    Group the df by the "by" argument, and also add the keys of "add_keys" (e.g. "hash", "filename") 
        if there is only one instance corresponding to the row.

    Args:
        add_keys: list of keys to add at the rightmost, e.g. ["hash", "filename"]
        other_keys: other keys to show. If None, will use all keys in df.
        mode: how to aggregate the values if there are more than one instance for the groupby.

    Returns:
        df_group: the desired DataFrame.
    """
    import pandas as pd
    def groupby_df(df, by, mode):
        if mode == "mean":
            df_group = df.groupby(by=by).mean()
        elif mode == "median":
            df_group = df.groupby(by=by).median()
        elif mode == "max":
            df_group = df.groupby(by=by).max()
        elif mode == "min":
            df_group = df.groupby(by=by).min()
        elif mode == "var":
            df_group = df.groupby(by=by).var()
        elif mode == "std":
            df_group = df.groupby(by=by).std()
        elif mode == "count":
            df_group = df.groupby(by=by).count()
        else:
            raise
        return df_group
    df = deepcopy(df)
    if isinstance(mode, str):
        df_group = groupby_df(df, by=by, mode=mode)
    else:
        assert isinstance(mode, dict)
        df_list = []
        for mode_ele, keys in mode.items():
            if mode_ele != "count":
                df_list.append(groupby_df(df[by + keys], by=by, mode=mode_ele))
            else:
                df["count"] = 1
                df_list.append(groupby_df(df[by + ["count"]], by=by, mode=mode_ele))
        df_group = pd.concat(df_list, axis=1)
    if other_keys is None:
        other_keys = list(df_group.keys())
    if not isinstance(add_keys, list):
        add_keys = [add_keys]
    if not isinstance(other_keys, list):
        other_keys = [other_keys]
    if isinstance(mode, dict) and "count" in mode and "count" not in other_keys:
        other_keys.append("count")
    df_group[add_keys] = None
    for i in range(len(df_group)):
        for k, key in enumerate(reversed(add_keys)):
            df_group_ele = df_group.iloc[i]
            filter_dict = dict(zip(df_group.T.keys().names, df_group.T.keys()[i]))
            df_filter = filter_df(df, filter_dict)
            if len(df_filter) == 1:
                df_group.iat[i, -(k+1)] = df_filter[key].values[0]
    df_group = df_group[other_keys + add_keys]
    return df_group


import random
class CustomSampler(Sampler):
    def __init__(self, data,batch_size,noncollision_hold_probability,distance_threshold):
        self.data = data
        self.batch_size=batch_size
        self.noncollision_hold_probability=noncollision_hold_probability
        self.distance_threshold=distance_threshold

    def __iter__(self):
        indices = []
        p_list=[]
        j_list=[]
        dis_list=[]
        Flag=0
        num1=0
        num2=0
        distance=0.
        distance2=0.
        seed = 42
        random.seed(seed)
        
        indices_np=np.load("/user/project/inverse_design/dataset/nbody_dataset/nbody-2/customerSampler_indices.npy")
        
        indices=indices_np
        random.shuffle(indices)
        indices = torch.tensor(indices)
        self.indices=indices
        return iter(indices)
    def __len__(self):
        return len(self.indices)
import torch.nn as nn
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target):
        loss = torch.abs(predicted - target)
        predicted_reshape=predicted.reshape(predicted.shape[0],predicted.shape[1],int(predicted.shape[2]/4),4)
        target_reshape=target.reshape(target.shape[0],target.shape[1],int(target.shape[2]/4),4)
        loss2=torch.abs((predicted_reshape[:,:,:,2])**2+(predicted_reshape[:,:,:,3])**2-((target_reshape[:,:,:,2])**2+(target_reshape[:,:,:,3])**2)).reshape(loss.shape[0],loss.shape[1],2,1)
        return torch.cat([loss.reshape(loss.shape[0],loss.shape[1],2,int(loss.shape[2]/2)),loss2],dim=3)
import matplotlib
def visulization(filename,cond,pred,n_bodies,conditioned_steps,rollout_steps,num_features):
    pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
    fontsize = 16
    for i in range(1):
        i=i*1
        fig = plt.figure(figsize=(18,15))
        if conditioned_steps!=0:
            cond_reshape = cond.reshape(cond.shape[0], conditioned_steps, n_bodies,num_features).to('cpu')
        pred_reshape = pred.reshape(cond.shape[0], rollout_steps, n_bodies,num_features).to('cpu')
        for j in range(n_bodies):
            if conditioned_steps!=0:
                marker_size_cond = np.linspace(1, 2, conditioned_steps) * 100
                plt.plot(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                plt.scatter(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], marker="+", linestyle="--", s=marker_size_cond)
            marker_size_pred = np.linspace(2, 3, rollout_steps) * 100
            plt.plot(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
            plt.scatter(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], marker="v", linestyle="-", s=marker_size_pred)
            plt.xlim([0,1])
            plt.ylim([0,1])
        plt.title(f"reverse", fontsize=fontsize)
        plt.tick_params(labelsize=fontsize)
        pdf.savefig(fig)
        i=i/1
    pdf.close()


def get_hashing(string_repr, length=None):
    """Get the hashing of a string."""
    import hashlib, base64
    hashing = base64.b64encode(hashlib.md5(string_repr.encode('utf-8')).digest()).decode().replace("/", "a")[:-2]
    if length is not None:
        hashing = hashing[:length]
    return hashing