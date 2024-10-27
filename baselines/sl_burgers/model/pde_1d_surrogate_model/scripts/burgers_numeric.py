import numpy as np
import torch
import torch.nn.functional as F

import math
from collections import OrderedDict
import time

# from fenics import *
# from fenics_adjoint import *

from .diff_matrices import Diff_mat_1D

def burgers_numeric_solve(u0, f, visc, T, dt=1e-4, num_t=10, mode=None):
    if mode!='const':
        assert f.size()[1]==num_t, 'check number of time interval'
    else:
        f=f.unsqueeze(1).repeat(1,num_t,1)
    
    # u0: (Nu0,s)
    # f: (Nf,Nt,s)

    #Grid size
    s = u0.size()[-1]
    
    Nu0 = u0.size()[0]
    Nf = f.size()[0]
    Nt = f.size()[1]
    
    xmin = 0.0; xmax = 1.0
    delta_x = (xmax-xmin)/(s+1)

    #Number of steps to final time
    steps = math.ceil(T/dt)

    u = u0.reshape(Nu0,1,s)
    u = F.pad(u, (1,1))
    f = f.reshape(1,Nf,Nt,s)
    f = F.pad(f, (1,1))
    
    #Record solution every this number of steps
    record_time = math.floor(steps/Nt)
    
    D_1d, D2_1d = Diff_mat_1D(s+2)   
    #remedy?
    D_1d.rows[0] = D_1d.rows[0][:2]
    D_1d.rows[-1] = D_1d.rows[-1][-2:]
    D_1d.data[0] = D_1d.data[0][:2]
    D_1d.data[-1] = D_1d.data[-1][-2:]
    
    D2_1d.rows[0] = D2_1d.rows[0][:3]
    D2_1d.rows[-1] = D2_1d.rows[-1][-3:]
    D2_1d.data[0] = D2_1d.data[0][:3]
    D2_1d.data[-1] = D2_1d.data[-1][-3:]
    
    t_sys_ind = list(D_1d.rows)
    t_sys = torch.FloatTensor(np.stack(D_1d.data)/(2*delta_x)).to(u0.device)
    d_sys_ind = list(D2_1d.rows)
    d_sys = torch.FloatTensor(visc*np.stack(D2_1d.data)/delta_x**2).to(u0.device)
    
    #Saving solution and time
    sol = torch.zeros(Nu0, Nf, s, Nt, device=u0.device)
    
    #Record counter
    c = 0
    #Physical time
    t = 0.0
    f_idx=-1
    for j in range(steps):
        u = u[...,1:-1]
        u = F.pad(u, (1,1))
        
        u_s = u**2
        transport = torch.einsum('bcsi,si->bcs', u_s[...,t_sys_ind], t_sys)
        diffusion = torch.einsum('bcsi,si->bcs', u[...,d_sys_ind], d_sys)
        if j % record_time == 0:
            f_idx+=1
        u = u + dt * (-(1/2)*transport + diffusion + f[:,:,f_idx,:])
        
        #Update real time (used only for recording)
        t += dt

        if (j+1) % record_time == 0:

            #Record solution and time
            sol[...,c] = u[...,1:-1]
            c += 1

    sol=sol.permute(0,1,3,2) #(Nu0,Nf,Nt,s)
    trajectory = torch.cat((u0.reshape(Nu0,1,1,s).repeat(1,Nf,1,1),sol),dim=2)
    return trajectory

# def burgers_numeric_control(u0, uT, _dt, gtol=1e-5, ftol=1e-6): #output : numpy array
#     set_log_active(False)
#     load_u_init=np.array(u0)
#     load_u_final=np.array(uT)

#     mesh = UnitIntervalMesh(128+1)
#     V = FunctionSpace(mesh, "CG", 1)

#     dt = Constant(_dt)
#     T = 1.0
#     nu = Constant(0.01)

#     ctrls = OrderedDict()
#     sols = OrderedDict()
#     t = float(dt)
#     while t <= T+_dt:
#         ctrls[t] = Function(V)
#         t += float(dt)

#     def solve_burgers(ctrls):
#         u = Function(V, name="solution")
#         u_old = Function(V)
#         v = TestFunction(V)

#         f = Function(V, name="source")
#         d = Function(V, name="data")

#         u.vector()[:] = np.pad(load_u_init[::-1], (1,1))
#         u_old.assign(u)
#         d.vector()[:] = np.pad(load_u_final[::-1], (1,1))

#         F = (dot(u - u_old, v)/dt + inner(u * u.dx(0), v) + nu*inner(grad(u), grad(v)) - dot(f, v))*dx
#         bc = DirichletBC(V, 0, "on_boundary")

#         t = float(dt)

#         while t <= T+_dt:
#             f.assign(ctrls[t])

#             problem = NonlinearVariationalProblem(F, u, bc, J=derivative(F, u))
#             solver = NonlinearVariationalSolver(problem)
#             solver.parameters["newton_solver"]["absolute_tolerance"] = 1E-6
#             solver.parameters["newton_solver"]["relative_tolerance"] = 1E-6
#             solver.solve()
#             u_old.assign(u)
#             t += float(dt)

#         j = assemble((1/2)*(u - d)**2*dx)

#         return u, d, j

#     u, d, j = solve_burgers(ctrls)
#     alpha = Constant(0.01)
#     regularisation = dt*alpha/2*sum([f**2*dx for f in list(ctrls.values())])

#     J = j + assemble(regularisation)
#     m = [Control(c) for c in ctrls.values()]

#     rf = ReducedFunctional(J, m)
#     start = time.time()
#     opt_ctrls = minimize(rf, options={"maxiter": 20,"maxfun": 25, 'gtol': gtol, 'ftol': ftol, 'disp' : False})
#     end = time.time()
    
    
#     force=[]
#     for i in range(len(opt_ctrls)):
#         force.append(opt_ctrls[i].vector()[:])
#     f_optim=np.array(force)
#     f_optim=f_optim[:,1:-1][:,::-1].copy()
    
#     return torch.FloatTensor(f_optim), end-start
