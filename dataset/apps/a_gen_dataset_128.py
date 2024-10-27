import sys
sys.path.append("../")

from phi.fluidformat import *
from phi.flow import FluidSimulation, DomainBoundary
import random
import numpy as np
from phi.math.nd import *
import matplotlib.pyplot as plt
from phi.solver.sparse import SparseCGPressureSolver
import matplotlib.animation as animation
from matplotlib.backends.backend_pdf import PdfPages
from phi.fluidformat import *
import os
import pdb
import scipy.sparse as sp
from scipy.sparse import csr_matrix, save_npz
from numpy.random import default_rng
import argparse
import multiprocessing
from evaluate_solver import *
from debug import *
from datetime import datetime
import time


current_time = datetime.now()
current_month = current_time.month
current_day = current_time.day 
current_hour = current_time.hour 
current_minute = current_time.minute 
global time_str
time_str = f"{current_month}{current_day}_{current_hour}{current_minute}"

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def plot_initial_obstacle(sim):
    """
    Function: Plot Obstacles of sim
    Input: sim
    """
    fig_ob, ax_ob = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of initial velocity in x-dirction###
    mappable_ob0 = ax_ob[0].imshow(sim._active_mask[0,:,:,0], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of initial velocity in y-dirction###
    # mappable_ob1 = ax_ob[1].imshow(sim._active_mask[0,:,:,1], cmap='viridis',
    #                          #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
    #                          interpolation="bicubic",
    #                          aspect='auto',
    #                          origin='lower')
    fig_ob.colorbar(mappable_ob0, ax=ax_ob[0])
    #fig_ob.colorbar(mappable_ob1, ax=ax_ob[1])
    fig_ob.tight_layout()


def plot_init_op_velocity(init_op_velocity):
    """
    Function: Plot initial velocity
    Input: StaggeredGrid type velocity
    """
    fig, ax = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of initial velocity in x-dirction###
    mappable0 = ax[0].imshow(init_op_velocity.staggered[0,:,:,0], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of initial velocity in y-dirction###
    mappable1 = ax[1].imshow(init_op_velocity.staggered[0,:,:,1], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             interpolation="bicubic",
                             aspect='auto',
                             origin='lower')
    ax[0].set_title('Heatmap of initial velocity in x-dirction')
    ax[1].set_title('Heatmap of initial velocity in y-dirction')
    fig.colorbar(mappable0, ax=ax[0])
    fig.colorbar(mappable1, ax=ax[1])
    fig.tight_layout()


def plot_velocity_with_mask(divergent_velocity):
    """
    Function: Plot the heatmap of velocity
    Input: StaggeredGrid type divergent_velocity
    """
    fig, ax = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of initial velocity in x-dirction###
    mappable0 = ax[0].imshow(divergent_velocity.staggered[0,:,:,0], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of initial velocity in y-dirction###
    mappable1 = ax[1].imshow(divergent_velocity.staggered[0,:,:,1], cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             interpolation="bicubic",
                             aspect='auto',
                             origin='lower')
    ax[0].set_title('Heatmap of initial velocity \n in x-dirction (with mask)')
    ax[1].set_title('Heatmap of initial velocity \n in y-dirction (with mask)')

    fig.colorbar(mappable0, ax=ax[0])
    fig.colorbar(mappable1, ax=ax[1])
    fig.tight_layout()


###Vector Field Representation of Velocity###
def plot_vector_field_128(velocity):
    """
    Function: Plot velocity field
    Input: StaggeredGrid type velocity
    """
    fig = plt.figure()
    x,y = np.meshgrid(np.linspace(0,127,128),np.linspace(0, 127, 128))

    xvel = np.zeros([128]*2)
    yvel = np.zeros([128]*2)

    xvel[1::4,1::4] = velocity.staggered[0,1::4,1::4,0]
    yvel[1::4,1::4] = velocity.staggered[0,1::4,1::4,1]

    plt.quiver(x,y,xvel,yvel,scale=2.5, scale_units='inches')
    plt.title('Vector Field Plot')



def plot_velocity_boundary_effect2(velocity):
    """
    Function: Plot velocity field
    Input: StaggeredGrid type velocity
    """
    hor_velocity_array = np.zeros([128, 128], dtype=float)
    hor_velocity_array = velocity.staggered[0,:,:,0]

    ver_velocity_array = np.zeros([128, 128], dtype=float)
    ver_velocity_array = velocity.staggered[0,:,:,1]


    fig, ax = plt.subplots(figsize=(8,4),ncols=2)
    ###Heatmap of velocity meating equation in x-dirction###
    mappable0 = ax[0].imshow(hor_velocity_array, cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             aspect='auto',
                             origin='lower')
    ###Heatmap of velocity meating equation in y-dirction###
    mappable1 = ax[1].imshow(ver_velocity_array, cmap='viridis',
                             #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                             interpolation="bicubic",
                             aspect='auto',
                             origin='lower')
    ax[0].set_title('x-axis velocity boundary effect')
    ax[1].set_title('y-axis velocity boundary effect')
    fig.colorbar(mappable0, ax=ax[0])
    fig.colorbar(mappable1, ax=ax[1])
    fig.tight_layout()



def plot_loop(loop_advected_density, loop_velocity, target_des_array, frame=None):
    """
    Function: Plot density field & velocity field
    Input:
        loop_advected_density: numpy array
        loop_velocity: staggeredgrid
        target_des_array: (optional) numpy array
    """
    fig, ax = plt.subplots()
    ax.imshow(loop_advected_density[0,:,:,0], origin='lower')
    """
    xvel = np.zeros([129]*2)
    yvel = np.zeros([129]*2)
    xvel[1::8,1::8] = loop_velocity.staggered[0,1::8,1::8,0]
    yvel[1::8,1::8] = loop_velocity.staggered[0,1::8,1::8,1]
    ax.quiver(x,y,xvel,yvel,scale=3., scale_units='inches', color="white")
    """
    ax.scatter(hor_bound, ver_bound, color="grey", marker=",")

    fig.savefig(f'dens_sample/{frame}.png', dpi=50)

    velocity_array = np.zeros([128, 128, 2], dtype=float)
    velocity_array[...,0] = loop_velocity.staggered[0,:,:,0]
    velocity_array[...,1] = loop_velocity.staggered[0,:,:,1]
    
    fig1, ax1 = plt.subplots(figsize=(8,4), ncols=2)
    ###Heatmap of velocity meating equation in x-dirction###
    mappable0 = ax1[0].imshow(velocity_array[:,:,0], cmap='viridis',
                                #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                aspect='auto',
                                origin='lower')
    ###Heatmap of velocity meating equation in y-dirction###
    mappable1 = ax1[1].imshow(velocity_array[:,:,1], cmap='viridis',
                                #extent=[0,sensordata.shape[0],0,sensordata.shape[1]],
                                interpolation="bicubic",
                                aspect='auto',
                                origin='lower')
    fig1.colorbar(mappable0, ax=ax1[0])
    fig1.colorbar(mappable1, ax=ax1[1])
    fig1.tight_layout()


def get_real_vel(vel):
    """
    Function: Get Real Velocity from Nomral Distribution
    Input: 
        vel: float
    Output:
        real_vel: float
    """
    std = abs(vel / 4)
    real_vel = np.random.normal(vel, std)
    return real_vel


def build_obstacles_pi_128(sim):
    """
    Function: Set obstacles
    Input: 
        sim: FluidSimulation object
    """

    sim.set_obstacle((1, 96), (16, 16)) # Bottom

    sim.set_obstacle((8, 1), (16, 16)) # Left Down
    sim.set_obstacle((16, 1), (40, 16)) # Left Medium
    sim.set_obstacle((40, 1), (72, 16)) # Left Up

    sim.set_obstacle((8, 1), (16, 112)) # Right Down
    sim.set_obstacle((16, 1), (40, 112)) # Right Medium
    sim.set_obstacle((40, 1), (72, 112)) # Right Up

    # Buckets
    sim.set_obstacle((1, 8), (112, 16)) # [16-24] # [24-40(16)]
    sim.set_obstacle((1, 16), (112, 40)) # [40-56] # [56-72(16)]
    sim.set_obstacle((1, 16),(112, 72)) # [72-88] # [88-104(16)]
    sim.set_obstacle((1, 8),(112, 104)) # [104-113]


    # y-axis obstacle
    sim.set_obstacle((16, 1), (64, 48))
    sim.set_obstacle((16, 1), (96, 48))
    sim.set_obstacle((16, 1), (64, 80))
    sim.set_obstacle((16, 1), (96, 80))
    
    # Should Change
    sim.set_obstacle((1, 128-40-40), (40, 40)) # x-axis


def apply_mask(sim, optimizable_velocity):
    """
    Function: Apply Mask to Control
    Input:
    sim: FluidSimulation object
    optimizable_velocity: numpy array
    Output:
    divergent_velocity: StaggeredGrid
    """
    control_mask = sim.ones("staggered")
    control_mask.staggered[:, 16:112, 16:112, :] = 0
    divergent_velocity = optimizable_velocity * control_mask.staggered
    divergent_velocity = StaggeredGrid(divergent_velocity)
    return divergent_velocity 


def initialize_field_128():
    """
    Function: initialize fluid field
    Output:
        sim: FluidSimulation Object
    """
    sim = FluidSimulation([127]*2, DomainBoundary([(True, True), (True, True)]), force_use_masks=True)
    build_obstacles_pi_128(sim)
    return sim

def closest_multiple(num, record_scale):
    """
    Function: get the closest multiple of num by record_scale
    Input:
    num: original number (int)
    record_scale: divisor (int)
    Output:
    closest_multiple: number (int)
    """
    lower_multiple = (num // record_scale) * record_scale
    upper_multiple = lower_multiple + record_scale
    distance_lower = abs(num - lower_multiple)
    distance_upper = abs(num - upper_multiple)
    if distance_lower < distance_upper:
        closest_multiple = lower_multiple
    else:
        closest_multiple = upper_multiple
    
    return closest_multiple

def get_per_vel(y_scale, min_scale, max_scale, xs, ys, record_scale):
    """
    Function: Calculate vague velocity
    Input:
        xs: random x-position for turn
        ys: random y-position for turn
    Output:
        vxs: vx list
        vys: vy list
        intervals: frame num for each interval
    """
    distance = ((xs[1]-xs[0])**2+(ys[1]-ys[0])**2)**(0.5) + ((xs[2]-xs[1])**2+(ys[2]-ys[1])**2)**(0.5) + ((xs[3]-xs[2])**2+(ys[3]-ys[2])**2)**(0.5) + ((xs[4]-xs[3])**2+(ys[4]-ys[3])**2)**(0.5)
    distance1 = ((xs[1]-xs[0])**2+(ys[1]-ys[0])**2)**(0.5)
    distance2 = ((xs[2]-xs[1])**2+(ys[2]-ys[1])**2)**(0.5)
    distance3 = ((xs[3]-xs[2])**2+(ys[3]-ys[2])**2)**(0.5)
    distance4 = ((xs[4]-xs[3])**2+(ys[4]-ys[3])**2)**(0.5)


    v = distance / float(scenelength)

    vx1 = v * (xs[1]-xs[0]) / distance1 
    vy1 = v * (ys[1]-ys[0]) / distance1 
    vx2 = v * (xs[2]-xs[1]) / distance2
    vy2 = v * (ys[2]-ys[1]) / distance2
    vx3 = v * (xs[3]-xs[2]) / distance3 
    vy3 = v * (ys[3]-ys[2]) / distance3
    vx4 = v * (xs[4]-xs[3]) / distance4 
    vy4 = v * (ys[4]-ys[3]) / distance4

    # scale = np.random.uniform(2, 5)
    scale = np.random.uniform(min_scale, max_scale)

    vxs = [get_real_vel(scale*vx1), get_real_vel(scale*vx2), get_real_vel(scale*vx3), get_real_vel(scale*vx4)]
    vys = [get_real_vel(y_scale*vy1), get_real_vel(y_scale*vy2), get_real_vel(y_scale*vy3), get_real_vel(y_scale*vy4)]


    interval1 = int(scenelength * distance1 / distance)
    interval2 = int(scenelength * distance2 / distance)
    interval3 = int(scenelength * distance3 / distance)

    interval1_ = closest_multiple(interval1, record_scale)
    interval2_ = closest_multiple(interval2, record_scale)
    interval3_ = closest_multiple(interval3, record_scale)

    intervals_ = [interval1_+1, interval2_, interval3_]

    return vxs, vys, intervals_


def exp2_target_128():
    """
    Function: Get x,y for turns
    Output:
        xs: list x-position for each turn
        ys: list y-position for each turn
    """
    m = 4
    start_x = np.random.randint(16+2+m, 112-10-m)
    start_x = closest_multiple(start_x, 2)
    start_y = np.random.randint(16+2+m, 40-10-m)
    start_y = closest_multiple(start_y, 2)

    if start_x < (64-8):
        a = 0
    else:
        a = 1
    target1_x = np.random.randint(16+m, 64-8) if a == 0 else np.random.randint(64, 112-8-m)
    target2_x = np.random.randint(16+m, 64-8) if a == 0 else np.random.randint(64, 112-8-m)
    target3_x = np.random.randint(50, 80-1-8)
    end_x = np.random.randint(64-8, 64+8-8)


    target1_y = 40
    target2_y = 50
    target3_y = 64
    end_y = 112
    
    xs = [int(start_x), int(target1_x), int(target2_x), int(target3_x), int(end_x)]
    ys = [int(start_y), int(target1_y), int(target2_y), int(target3_y), int(end_y)]
    
    return xs, ys


def initialize_gas_exp2_128(xs, ys):
    """
    Function: Intialize density field
    Input:
        xs: x-postion list
        ys: y-postion list
    Output:
        array: numpy array density field
    """
    array = np.zeros([127, 127, 1], dtype=float)
    start_x = xs[0]
    start_y = ys[0]
    array[start_y:start_y+10, start_x:start_x+10, :] = 1
    return array


def initialize_velocity_128(vx, vy):
    """
    Function: Initialize velocity field
    Input:
        vx, vy: float velocity-x, velocity-y
    Output:
        init_op_velocity: StaggeredGrid velocity
        optimizable_velocity: numpy array velocity
    """
    velocity_array = np.zeros([128, 128, 2], dtype=float)
    velocity_array[...,0] = vx
    velocity_array[...,1] = vy
    init_op_velocity = StaggeredGrid(velocity_array.reshape((1,)+velocity_array.shape))
    optimizable_velocity = init_op_velocity.staggered
    return init_op_velocity, optimizable_velocity


def get_envolve(sim,pre_velocity,frame,control_write,space_scale,record_scale,vx=None,vy=None):
    """
    Function: get next step velocity with indirect control
    Input:
        sim: FluidSimulation Object
        pre_velocity: StaggeredGrid previous velocity
        frame: int
        control_write: numpy array
        vx: float
        vy: float
    Output:
        velocity: StaggeredGrid next velocity
        control_write: numpy array
    """
    if(vx==None and vy==None):
        current_vel_field = np.zeros_like(pre_velocity.staggered)
        
        # Add noise # noise_arr.shape = [1,128,128,2]
        noise_arr = np.random.normal(loc=0,scale=0.1,size=pre_velocity.staggered.shape)
        
        # Calculate Current Controlled Velocity # current_vel_field.shape = [1,128,128,2]
        current_vel_field[:,:,:16,:] = pre_velocity.staggered[:,:,:16,:] + noise_arr[:,:,:16,:]
        current_vel_field[:,:,112:,:] = pre_velocity.staggered[:,:,112:,:] + noise_arr[:,:,112:,:]
        current_vel_field[:,112:,16:112,:] = pre_velocity.staggered[:,112:,16:112,:] + noise_arr[:,112:,16:112,:]
        current_vel_field[:,:16,16:112,:] = pre_velocity.staggered[:,:16,16:112,:] + noise_arr[:,:16,16:112,:]
        
        divergent_velocity =  current_vel_field.copy()

        if frame % record_scale == 0:
            control_write[:,:,0,int(frame/record_scale)] = divergent_velocity[0,::space_scale,::space_scale,0]
            control_write[:,:,1,int(frame/record_scale)] = divergent_velocity[0,::space_scale,::space_scale,1]

        current_vel_field[:,16:112,16:112,:] = pre_velocity.staggered[:,16:112,16:112,:]

        Current_vel_field = StaggeredGrid(current_vel_field)

        velocity = sim.divergence_free(Current_vel_field, solver=SparseCGPressureSolver(), accuracy=1e-8)
        velocity = sim.with_boundary_conditions(velocity)

        return velocity, control_write
    else:
        divergent_velocity = np.zeros((1,128,128,2), dtype=float)

        divergent_velocity[:,:,:,0] = np.random.normal(loc=vx,scale=abs(vx/10),size=(1,128,128))
        divergent_velocity[:,:,:,1] = np.random.normal(loc=vy,scale=abs(vy/10),size=(1,128,128))

        divergent_velocity[:, 16:112, 16:112, :] = 0
        divergent_velocity_ = StaggeredGrid(divergent_velocity)
        
        if frame % record_scale == 0:
            control_write[:,:,0,int(frame/record_scale)] = divergent_velocity[0,::space_scale,::space_scale,0]
            control_write[:,:,1,int(frame/record_scale)] = divergent_velocity[0,::space_scale,::space_scale,1]

        current_vel_field = math.zeros_like(divergent_velocity_.staggered)
        current_vel_field[:,16:112,16:112,:] = pre_velocity.staggered[:,16:112,16:112,:]

        current_vel_field[:,:,:16,:] = divergent_velocity_.staggered[:,:,:16,:]
        current_vel_field[:,:,112:,:] = divergent_velocity_.staggered[:,:,112:,:]
        current_vel_field[:,112:,16:112,:] = divergent_velocity_.staggered[:,112:,16:112,:]
        current_vel_field[:,:16,16:112,:] = divergent_velocity_.staggered[:,:16,16:112,:]

        Current_vel_field = StaggeredGrid(current_vel_field)
        
        velocity = sim.divergence_free(Current_vel_field, solver=SparseCGPressureSolver(), accuracy=1e-8)
        velocity = sim.with_boundary_conditions(velocity)

        return velocity, control_write


def get_initial_state(sim,xs,ys,vxs,vys,density_write,density_set_zero_write,velocity_write,control_write,record_scale,space_scale,initial_vy):
    """
    Function: get initial state (velocity_0, density_0)
    Input: calculated control & empty matrices
    Output: initial density field (numpy array), initial velocity field (staggeredgrid) and written matrices
    """
    # get inital square position for density field
    array = initialize_gas_exp2_128(xs=xs, ys=ys)
    if(space_scale == 1):
        density_write[:-1,:-1,:,0] = array[::space_scale,::space_scale,:] # 0st original density field
        density_set_zero_write[:-1,:-1,:,0] = array[::space_scale,::space_scale,:] # 0st zerod density field
    else:
        density_write[:,:,:,0] = array[::space_scale,::space_scale,:] # 0st original density field
        density_set_zero_write[:,:,:,0] = array[::space_scale,::space_scale,:] # 0st zerod density field

    # initialize velocity
    init_op_velocity, optimizable_velocity = initialize_velocity_128(vx=0, vy=initial_vy)
    
    # write the first velocity field (all trajactory with the same velocity field)
    velocity_write[:,:,0,0] = optimizable_velocity[0,::space_scale,::space_scale,0] # write 0st vel
    velocity_write[:,:,1,0] = optimizable_velocity[0,::space_scale,::space_scale,1]

    init_op_density = StaggeredGrid(array)
    init_op_density = init_op_density.staggered.reshape((1,)+init_op_density.staggered.shape)

    return init_op_density, init_op_velocity, density_write, density_set_zero_write, velocity_write, control_write


def get_bucket_mask():
    """
    Function: get absorb area to calculate smoke_out
    Output:
    cal_smoke_list: smoke absorb matrix for each bucket
    cal_smoke_concat: matrix concat all absorb areas (all zero except absorb area with one)
    set_zero_matrix: all one except absorb area with zero
    """
    bucket_pos = [(112,24-2,127-112,16+4),(112,56-2,127-112,16+4),(112,88-2,127-112,16+4)]
    bucket_pos_y = [(24-2,0,16+4,16),(56-2,0,16+4,16),(24-2,112,16+4,127-112),(56-2,112,16+4,127-112)]
    cal_smoke_list = [] 
    set_zero_matrix = np.ones((128,128))
    cal_smoke_concat = np.zeros((128,128))
    for pos in bucket_pos:
        cal_smoke_matrix = np.zeros((128,128)) 
        y,x,len_y,len_x = pos[0], pos[1], pos[2], pos[3]
        cal_smoke_matrix[y:y+len_y, x:x+len_x] = 1
        cal_smoke_concat[y:y+len_y, x:x+len_x] = 1
        set_zero_matrix[y:y+len_y, x:x+len_x] = 0
        cal_smoke_list.append(cal_smoke_matrix)
    for pos in bucket_pos_y:
        cal_smoke_matrix = np.zeros((128,128)) 
        y,x,len_y,len_x = pos[0], pos[1], pos[2], pos[3]
        cal_smoke_matrix[y:y+len_y, x:x+len_x] = 1
        cal_smoke_concat[y:y+len_y, x:x+len_x] = 1
        set_zero_matrix[y:y+len_y, x:x+len_x] = 0
        cal_smoke_list.append(cal_smoke_matrix)
 
    return cal_smoke_list, cal_smoke_concat, set_zero_matrix #, absorb_matrix, cal_inside_smoke


def get_save_name():
    """
    Function: get save name
    """
    des_name_no_zero = f'Density.npy'
    vel_name = f'Velocity.npy'
    control_name = f'Control.npy'
    smoke_cal_name = f'Smoke.npy'

    return des_name_no_zero, vel_name, control_name, smoke_cal_name


def get_domain_name():
    return f'domain.npy'


def write_vel_density(loop_velocity,loop_advected_density,loop_density_no_set,density_write,density_set_zero_write,velocity_write,frame,smoke_outs_128_record,smoke_outs_128,space_scale,record_scale):
    """
    Function: write velocity density field for turns
    Input:
    loop_velocity: StaggeredGrid Type velocity
    loop_advected_density: Numpy Array Type density (should be set zero at absorb area)
    loop_density_no_set: Numpy Array Type density (will not be set zero)
    density_set_zero_write: Numpy Array with shape (n_x, n_x, 1, n_t)
    velocity_write: Numpy Array with shape (n_x, n_x, 2, n_t)
    frame: the index of time
    smoke_outs_128_record: each bucket smoke of previous timestep
    smoke_outs_128: Numpy Array with shape (8, n_t)
    space_scale: downsample rate of space
    record_scale: downsample rate of time
    Output:
    loop_advected_density: Numpy Array Type density (should be set zero at absorb area)
    density_write, density_set_zero_write, velocity_write, smoke_outs_128_record, smoke_outs_128: write matrix
    """
    if(space_scale == 1):
        density_write[:-1,:-1,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] # original density field
        density_set_zero_write[:-1,:-1,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:] # set-zero density field
    else:
        density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] # original density field
        density_set_zero_write[:,:,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:] # set-zero density field
    velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,0]
    velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,1]

    array = np.zeros((128,128,1), dtype=float)
    array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 
    
    # calculate smoke_outs
    if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
        for i in range(len(cal_smoke_list)):
            smoke_outs_128_record[i] += np.sum(array[:,:,0] * cal_smoke_list[i][:,:])

        loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

    smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
    smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,:,:,:])
    return loop_advected_density,density_write,density_set_zero_write,velocity_write,smoke_outs_128_record,smoke_outs_128


def plot_narray(matrix):
    plt.imshow(matrix, cmap='gray') 
    plt.colorbar()
    # plt.savefig('smoke_concat.png')
    # plt.show()


def loop_write_0423(sim,loop_advected_density,loop_velocity,smoke_outs_128,save_sim_path,vxs,vys,intervals,xs,ys,density_write, \
                    density_set_zero_write,velocity_write,control_write,record_scale,space_scale,filter,min_sum_rate, max_sum_rate):
    """
    Function: Write loop 
    Input:
    sim: FluidSimulation Object
    loop_advected_density: Numpy Array Type density
    loop_velocity: StaggeredGrid Object velocity
    smoke_outs_128: (8, n_t) numpy array smoke out record
    save_sim_path: path to save each sim
    vxs, vys: (float) calculated vx and vy
    intervals: (int) calculated interval
    xs, xy: (int) initial positions 
    density_write, density_set_zero_write, velocity_write, control_write: (nx, nx, 1 or 2, n_t)write matrices
    record_scale: (int) downsample rate of time
    space_scale: (int) downsample rate of space
    filter: (boolean) if we need to filter the target rate
    min_sum_rate, max_sum_rate: (float) to ensure density sum changes in a small religion
    Output:
    trajectory qualified -> density_write, velocity_write, control_write, smoke_outs_128
    trajectory not qualified -> False
    """
    print_list = [1, scenelength/16, scenelength/8, scenelength/4, scenelength/2, scenelength]

    loop_density_no_set = loop_advected_density.copy() # original
    loop_advected_density = loop_advected_density # density set zero
    control_write = control_write
    density_write = density_write
    density_set_zero_write = density_set_zero_write
    velocity_write = velocity_write
    smoke_outs_128_record = np.zeros((7,), dtype=float)

    smoke_outs_128[0,-1] = np.sum(loop_density_no_set[0,:,:,:])

    # write density field
    array = np.zeros((128,128,1), dtype=float)
    array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 

    # calculate smoke out
    if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
        for i in range(len(cal_smoke_list)):
            smoke_outs_128_record[i] += np.sum(array[:,:,0] * cal_smoke_list[i][:,:])
        loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

    smoke_outs_128[0,:-1] = smoke_outs_128_record # 1st smoke_out
    smoke_outs_128[0,-1] = np.sum(loop_advected_density[0,:,:,:])
    
    # first simulation
    # first envolve & write the first control
    loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=0,vx=vxs[0],vy=vys[0],control_write=control_write,space_scale=space_scale,record_scale=record_scale) # 0st control and correspoding 0st vel
    
    loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero 1st 
    loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set 1st

    # write density field
    array = np.zeros((128,128,1), dtype=float)
    array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 

    # calculate smoke out
    if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
        for i in range(len(cal_smoke_list)):
            smoke_outs_128_record[i] += np.sum(array[:,:,0] * cal_smoke_list[i][:,:])
        loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

    if 1 % record_scale == 0:
        if(space_scale==1):
            density_write[:-1,:-1,:,int(1/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] # 1st dens
            density_set_zero_write[:-1,:-1,:,int(1/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
        else:
            density_write[:,:,:,int(1/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] # 1st dens
            density_set_zero_write[:,:,:,int(1/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
        velocity_write[:,:,0,int(1/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,0] # 1st vel
        velocity_write[:,:,1,int(1/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,1]
        smoke_outs_128[int(1/record_scale),:-1] = smoke_outs_128_record # 1st smoke_out
        smoke_outs_128[int(1/record_scale),-1] = np.sum(loop_advected_density[0,:,:,:])
    
    # print("step 1")
    for frame in range(2, intervals[0]):
        # envolve to get current velocity
        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame-1,control_write=control_write,space_scale=space_scale,record_scale=record_scale)
        # infer next density with current velocity
        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # density set zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # density without set zero

        # write density field
        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 

        # calculate smoke out
        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[:,:,0] * cal_smoke_list[i][:,:])
            # zero loop_advected_density
            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

        if frame % record_scale == 0:
            if space_scale == 1:
                density_write[:-1,:-1,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] # 1st dens
                density_set_zero_write[:-1,:-1,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            else:
                density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] # 1st dens
                density_set_zero_write[:,:,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,0] # 1st vel
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record # 1st smoke_out
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,:,:,:])
        
        # print(f'frame:{frame} smoke_out: {np.sum(smoke_outs_128_record)} density_sum: {np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,:,:,:])}')
        
    # get extreme point control
    frame = intervals[0]
    loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame-1,vx=vxs[1],vy=vys[1],control_write=control_write,space_scale=space_scale,record_scale=record_scale)
    loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt)
    loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt)

    if frame % record_scale == 0:
        loop_advected_density,density_write,density_set_zero_write,velocity_write,smoke_outs_128_record,smoke_outs_128 = write_vel_density(loop_velocity=loop_velocity,loop_advected_density=loop_advected_density, \
                loop_density_no_set=loop_density_no_set,density_write=density_write,density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,frame=frame, \
                smoke_outs_128_record=smoke_outs_128_record,smoke_outs_128=smoke_outs_128,space_scale=space_scale,record_scale=record_scale)
    

    # print("step 2")
    for frame in range(intervals[0]+1, intervals[0]+intervals[1]):
        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame-1,control_write=control_write,space_scale=space_scale,record_scale=record_scale)
        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set

        # write density field
        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 

        # calculate smoke out
        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[:,:,0] * cal_smoke_list[i][:,:])

            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]
        
        if frame % record_scale == 0:
            if space_scale==1:
                density_write[:-1,:-1,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] 
                density_set_zero_write[:-1,:-1,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            else:
                density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] 
                density_set_zero_write[:,:,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,0]
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,:,:,:])


        # print(f'frame:{frame} smoke_out: {np.sum(smoke_outs_128_record)} density_sum: {np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,:,:,:])}')
    
    # get extreme point control
    frame = intervals[0]+intervals[1]
    loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame-1,vx=vxs[2],vy=vys[2],control_write=control_write,space_scale=space_scale,record_scale=record_scale)
    loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt)
    loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt)

    if frame % record_scale == 0:
        loop_advected_density,density_write,density_set_zero_write,velocity_write,smoke_outs_128_record,smoke_outs_128 = write_vel_density(loop_velocity=loop_velocity,loop_advected_density=loop_advected_density, \
                loop_density_no_set=loop_density_no_set,density_write=density_write,density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,frame=frame, \
                smoke_outs_128_record=smoke_outs_128_record,smoke_outs_128=smoke_outs_128,space_scale=space_scale,record_scale=record_scale)

    # print("step 3")
    for frame in range(intervals[0]+intervals[1]+1, intervals[0]+intervals[1]+intervals[2]):
        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame-1,control_write=control_write,space_scale=space_scale,record_scale=record_scale)
        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set

        # write density field
        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 

        # calculate smoke out
        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[:,:,0] * cal_smoke_list[i][:,:])

            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

        if frame % record_scale == 0:
            if space_scale == 1:
                density_write[:-1,:-1,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] 
                density_set_zero_write[:-1,:-1,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            else:
                density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] 
                density_set_zero_write[:,:,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,0]
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,:,:,:])

        # print(f'frame:{frame} smoke_out: {np.sum(smoke_outs_128_record)} density_sum: {np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,:,:,:])}')
    
    frame = intervals[0]+intervals[1]+intervals[2]
    loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame-1,vx=vxs[3],vy=vys[3],control_write=control_write,space_scale=space_scale,record_scale=record_scale)
    loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt)
    loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt)

    if frame % record_scale == 0:
        loop_advected_density,density_write,density_set_zero_write,velocity_write,smoke_outs_128_record,smoke_outs_128 = write_vel_density(loop_velocity=loop_velocity,loop_advected_density=loop_advected_density, \
                loop_density_no_set=loop_density_no_set,density_write=density_write,density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,frame=frame, \
                smoke_outs_128_record=smoke_outs_128_record,smoke_outs_128=smoke_outs_128,space_scale=space_scale,record_scale=record_scale)

    # print("step 4")
    for frame in range(intervals[0]+intervals[1]+intervals[2]+1, scenelength+1):
        # first write density field under the previous velocity field
        # using advect function to get current density field movement under velocity field
        # loop_advected_density - numpy array - shape [1,255,255,1]
        loop_velocity, control_write = get_envolve(sim=sim,pre_velocity=loop_velocity,frame=frame-1,control_write=control_write,space_scale=space_scale,record_scale=record_scale)
        loop_advected_density = loop_velocity.advect(loop_advected_density, dt=dt) # set_zero
        loop_density_no_set = loop_velocity.advect(loop_density_no_set, dt=dt) # original/ no_set
        
        # write density field
        array = np.zeros((128,128,1), dtype=float)
        array[:-1,:-1,:] = loop_advected_density[0,:,:,:] 

        # calculate smoke out
        if(np.sum((array[:,:,0]*cal_smoke_concat))>0):
            for i in range(len(cal_smoke_list)):
                smoke_outs_128_record[i] += np.sum(array[:,:,0] * cal_smoke_list[i][:,:])

            loop_advected_density[0,:,:,0] = loop_advected_density[0,:,:,0] * set_zero_matrix[:-1,:-1]

        if frame % record_scale == 0:
            if space_scale == 1:
                density_write[:-1,:-1,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] 
                density_set_zero_write[:-1,:-1,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            else:
                density_write[:,:,:,int(frame/record_scale)] = loop_density_no_set[0,::space_scale,::space_scale,:] 
                density_set_zero_write[:,:,:,int(frame/record_scale)] = loop_advected_density[0,::space_scale,::space_scale,:]
            velocity_write[:,:,0,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,0]
            velocity_write[:,:,1,int(frame/record_scale)] = loop_velocity.staggered[0,::space_scale,::space_scale,1]
            smoke_outs_128[int(frame/record_scale),:-1] = smoke_outs_128_record
            smoke_outs_128[int(frame/record_scale),-1] = np.sum(loop_advected_density[0,:,:,:])

            
        
        # print(f'frame:{frame} smoke_out: {np.sum(smoke_outs_128_record)} density_sum: {np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,:,:,:])}')
    
    density_sum = np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,:,:,:])
    target_rate = smoke_outs_128_record[1] / density_sum
    target_rule = (not filter) or (target_rate>0.8)
    low_bar = 10*10*min_sum_rate
    high_bar = 10*10*max_sum_rate
    density_field_sum = np.sum(smoke_outs_128_record)+np.sum(loop_advected_density[0,:,:,:])
    density_quality = (density_field_sum > low_bar) and (density_field_sum < high_bar)
    
    if (target_rule and density_quality):
        return density_write, velocity_write, control_write, smoke_outs_128
    else:
        return False
    
    
def exp2_same_side_128(timestamp, min_sum_rate, max_sum_rate, initial_vy, original_space_length, filter, y_scale, min_scale, max_scale,scenecount, original_timelength, time_length, space_length, is_train_, fix_velocity_, Test_, branch_num, data_savepath):
    """
    Function:
    the main function to produce trajectory
    Input:
    timestamp: seed
    min_sum_rate, max_sum_rate: (float) ensure the religion of density sum
    initial_vy: (float) initial y-direction velocity
    original_space_length: (int) space length before downsampling
    space_length: (int) space length after downsampling
    filter: (boolean) if we need to filter the data
    y_scale: (int) y-direction velocity scaler
    min_scale, max_scale: (int) x-direction velocity scaler
    scenecount: (int) sim count for each function
    original_time_length: (int) time length before downsampling
    time_length: (int) time length after downsampling
    is_train_, Test_: (boolean) whether the process use multi-processing
    fix_velocity: (boolean) whether the velocity is fixed
    branch_num: (int) No. of branch
    data_savepath: (str) the path to save data
    """
    
    pid = os.getpid()
    seed = pid + timestamp
    np.random.seed(seed)

    record_scale = int(original_timelength / time_length)
    space_scale = int(original_space_length / space_length)

    Test_ = Test_

    is_train, fix_velocity = is_train_, fix_velocity_

    if(Test_):
        scenecount = scenecount
    elif(is_train):
        scenecount = scenecount
    else:
        scenecount = scenecount
    
    # Universal Parameters
    global scenelength, dt, cal_smoke_list, cal_smoke_concat, set_zero_matrix
    scenelength = original_timelength
    dt = 1
    cal_smoke_list, cal_smoke_concat, set_zero_matrix= get_bucket_mask()

    save_path = f'./{data_savepath}/'

    # makedirs

    contents = os.listdir(save_path)


    begin_sim_set = scenecount * int(branch_num)

    test_branch = pid
    
    # for scene_index in range(begin_sim_set, scenecount+begin_sim_set):
    scene_index = begin_sim_set
    while scene_index < (scenecount+begin_sim_set):
        smoke_outs_128 = np.zeros(7)

        print("SCENE"+str(scene_index))
        
        sim = initialize_field_128()

        res_sim = sim._fluid_mask.reshape((127,127))
        boundaries = np.argwhere(res_sim==0)
        global ver_bound, hor_bound
        ver_bound = boundaries[:,0]
        hor_bound = boundaries[:,1]

        xs, ys = exp2_target_128()

        if scene_index < 10:
            sim_path = f'sim_00000{scene_index}'
        elif scene_index < 100:
            sim_path = f'sim_0000{scene_index}'
        elif scene_index < 1000:
            sim_path = f'sim_000{scene_index}'
        elif scene_index < 10000:
            sim_path = f'sim_00{scene_index}'
        elif scene_index < 100000:
            sim_path = f'sim_0{scene_index}'

        save_sim_path = os.path.join(save_path, sim_path)

        if not os.path.exists(save_sim_path):
            os.makedirs(save_sim_path)

        domain_name = get_domain_name()
        save_domain_path = os.path.join(save_sim_path, domain_name)
        np.save(save_domain_path, sim._active_mask)

        vxs, vys, intervals = get_per_vel(y_scale=y_scale, min_scale=min_scale, max_scale=max_scale, xs=xs, ys=ys, record_scale=record_scale)

        record_frame_len = time_length + 1
        
        record_space_len = int(128 / space_scale)
        density_write = np.zeros((record_space_len,record_space_len,1,record_frame_len), dtype=float)
        density_set_zero_write = np.zeros((record_space_len,record_space_len,1,record_frame_len), dtype=float)
        velocity_write = np.zeros((record_space_len,record_space_len,2,record_frame_len), dtype=float)
        control_write = np.zeros((record_space_len,record_space_len,2,record_frame_len), dtype=float)
        smoke_outs_128 = np.zeros((record_frame_len, 8))

        loop_advected_density,loop_velocity,density_write,density_set_zero_write,velocity_write,control_write = get_initial_state(xs=xs, ys=ys, \
                                        sim=sim, vxs=vxs, vys=vys,density_write=density_write,density_set_zero_write=density_set_zero_write,\
                                        velocity_write=velocity_write,control_write=control_write,record_scale=record_scale,space_scale=space_scale, initial_vy=initial_vy)
        
        flag = loop_write_0423(sim=sim, loop_advected_density=loop_advected_density, loop_velocity=loop_velocity,smoke_outs_128=smoke_outs_128,\
                                save_sim_path=save_sim_path,vxs=vxs,vys=vys,intervals=intervals,xs=xs,ys=ys,density_write=density_write, \
                                density_set_zero_write=density_set_zero_write,velocity_write=velocity_write,control_write=control_write, \
                                record_scale=record_scale,space_scale=space_scale,filter=filter,min_sum_rate=min_sum_rate, max_sum_rate=max_sum_rate)
        
        if(not flag==False):
            des_name, vel_name,control_name, smoke_cal_name = get_save_name()[0],get_save_name()[1],get_save_name()[2], get_save_name()[3]
            density_write, velocity_write, control_write, smoke_outs_128 = flag[0], flag[1], flag[2], flag[3]
            # des_set_zero_name = 'Density_set_zero.npz'
            des_path = os.path.join(save_sim_path,des_name)
            vel_path = os.path.join(save_sim_path,vel_name)
            control_path = os.path.join(save_sim_path,control_name)
            smoke_path = os.path.join(save_sim_path, smoke_cal_name)
            # des_set_zero_path = os.path.join(save_sim_path, des_set_zero_name)

            np.save(des_path, density_write)
            # np.save(des_set_zero_path, density_set_zero_write)
            np.save(vel_path, velocity_write)
            np.save(control_path, control_write)
            np.save(smoke_path, smoke_outs_128)
            save_txt_path = os.path.join(save_sim_path, 'smoke_out.csv')
            np.savetxt(save_txt_path, smoke_outs_128, delimiter=',')
            scene_index += 1

    print("DATA GENERATION DOWN!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_or_train", type=str, help="(test:input test or train)")
    parser.add_argument("--original_space_length", type=str, help='space length before downsampling')
    parser.add_argument("--data_savepath", type=str, help='dataset location')
    parser.add_argument("--branch_begin", type=str, help='branch begin number')
    parser.add_argument("--branch_end", type=str, help='branch end number')
    parser.add_argument("--time_length", type=str, help='downsample time to length n')
    parser.add_argument("--space_length", type=str, help='downsample space to length n')
    parser.add_argument("--scenecount", type=str, help='scene mission for every branch')
    parser.add_argument("--original_timelength", type=str, help='scene mission for every branch')
    parser.add_argument("--min_scale", type=str, help='min scale for v_x')
    parser.add_argument("--max_scale", type=str, help='max scale for v_x')
    parser.add_argument("--y_scale", type=str, help='scale for v_y')
    parser.add_argument("--filter", action='store_true', help='if we filter the data')
    parser.add_argument("--initial_vy", type=str, help='initial vy')
    parser.add_argument("--min_sum_rate", type=str, help='min density sum rate')
    parser.add_argument("--max_sum_rate", type=str, help='max density sum rate')


    args = parser.parse_args()
    
    if args.test_or_train == 'test':
        Test_ = True
        is_train = False
    elif args.test_or_train == 'train':
        Test_ = False
        is_train = True

    data_savepath = args.data_savepath
    begin_no = int(args.branch_begin)
    end_no = int(args.branch_end)
    original_space_length = int(args.original_space_length)
    branch_list = np.arange(begin_no,end_no)
    fix_velocity_ = False
    time_length = int(args.time_length)
    space_length = int(args.space_length)
    scenecount = int(args.scenecount)
    original_timelength = int(args.original_timelength)
    filter = args.filter
    min_scale = int(args.min_scale)
    max_scale = int(args.max_scale)
    y_scale = int(args.y_scale)
    initial_vy = float(args.initial_vy)
    min_sum_rate = float(args.min_sum_rate)
    max_sum_rate = float(args.max_sum_rate)

    timestamp = (int(time.time()) - 1720862454) * 100
    if args.test_or_train == 'train':
        if not os.path.exists(data_savepath):
            os.makedirs(data_savepath)
        
        processes = []
        for branch_num in branch_list:
            args_func = (timestamp, min_sum_rate, max_sum_rate, initial_vy, original_space_length, filter, y_scale, min_scale, max_scale,scenecount, original_timelength, time_length, \
                        space_length, is_train,fix_velocity_,Test_, str(branch_num), data_savepath)
            p = multiprocessing.Process(target=exp2_same_side_128, args=args_func)
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

    elif args.test_or_train == 'test':
        data_savepath = f'{data_savepath}_{time_str}'
        if not os.path.exists(data_savepath):
            os.makedirs(data_savepath)
            exp2_same_side_128(timestamp, min_sum_rate, max_sum_rate, initial_vy, original_space_length, filter, y_scale, min_scale, max_scale,scenecount, original_timelength, time_length, \
                        space_length, is_train,fix_velocity_,Test_, 0, data_savepath)
    else:
        print('Error: Missing Input Parameter test_or_train')