# this script test the LQ problem given the optimal value function
# only test policy gradient

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import time
import os

data_type = torch.float64
pcs = 5

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== initialization ==========
# set parameters
d = 1               # dimension
d_c = d             # dim control
d_w = d             # dim brownian motion
T = 0.1             # terminal time
beta0 = 0.2         # coef for t
beta = [0.1]        # coef for sin(pi x)
sigma_bar = 1           # coef for diffusion
beta_np = np.array(beta) # size d, can broadcast with N x d
beta_pt = torch.tensor(beta, dtype=data_type, device=device)

assert len(beta) == d, "beta does not match dimension"

# training parameters
num_actor_update = 2
num_steps = 30
learning_rate = 0.01
delta_tau = 0.1
milestones = [100,200]
decay = 5 # increase the learning rate due to flatness
num_trig_basis = 3

# set grid
Nt = 10          # number of time stamps
Nx = 500         # batch size for taining
N_valid = 300    # number of spatial samples for validation
net_size = 2    # width of the network


# corresponding qunantities needed
dt = T / Nt
sqrt_dt = np.sqrt(dt)

# define the ground true fucntions
def V(t,x): # true value function in numpy
    # x is N x d; V is N x 1
    temp = np.sum(beta_np * np.sin(np.pi*x), axis=-1, keepdims=True)
    return temp + beta0 * (T-t)

def V_pt(t,x): # true value function in numpy
    # x is N x d; V is N x 1
    temp = torch.sum(beta_pt * torch.sin(torch.pi*x), dim=-1, keepdim=True)
    return temp + beta0 * (T-t)

def V_grad(t,x): # gradient of V
    return np.pi * beta_np * np.cos(np.pi * x)

def V_grad_pt(t,x): # gradient of V
    return torch.pi * beta_pt * torch.cos(torch.pi * x)

# def g(x): # terminal cost
#     return np.sum(beta_np * np.sin(np.pi*x), axis=-1, keepdims=True)

def g(x): # terminal cost pytorh
    return torch.sum(beta_pt * torch.sin(torch.pi*x), dim=-1, keepdim=True)

# def r(x,u): # running cost
#     temp = np.sum(sigma_bar**2 * beta_np * np.sin(np.pi*x) + beta_np**2 * np.cos(np.pi*x)**2, axis=-1, keepdims=True)
#     return np.pi**2 * temp / 2 + beta0 + np.sum(u**2, axis=-1, keepdims=True)

def r(x,u): # running cost, torch
    temp = torch.sum(sigma_bar**2 * beta_pt * torch.sin(torch.pi*x) + beta_pt**2 * torch.cos(torch.pi*x)**2, dim=-1, keepdim=True)
    return torch.pi**2 * temp / 2 + beta0 + torch.sum(u**2, dim=-1, keepdim=True)

def u_star(t,x): # optimal control
    return -np.pi * beta_np * np.cos(np.pi * x)

def u_star_pt(t,x): # optimal control in torch
    return -torch.pi * beta_pt * torch.cos(torch.pi * x)

def b_x(x,u): # drift for x, N x d, torch
    return u

def diffu_x(x, dW_t): # diffusion for x, N x d, torch
    return sigma_bar * dW_t

def diffu_y(x, grad_y, dW_t): # diffusion for y, N x 1, torch
    return sigma_bar * torch.sum(grad_y * dW_t, dim=-1, keepdim=True)

# ========== construct neural network ==========
# periodic net for control

class Control_net(nn.Module): # net for the control
    def __init__(self, outdim):
        super().__init__()
        self.dim = d
        self.linear1 = nn.Linear(2*self.dim*num_trig_basis+1, net_size)
        self.linear2 = nn.Linear(net_size, outdim)
        self.activate = nn.ReLU()

    def forward(self, t, x):
        # x is N x d, output is N x 1
        kx = x.unsqueeze(1).expand(-1, num_trig_basis, -1) \
            * torch.arange(1,num_trig_basis+1,device=device).unsqueeze(1)
        kx = kx.view(-1,num_trig_basis*self.dim)
        tx = torch.cat((t,torch.sin(kx),torch.cos(kx)), dim=-1)
        NN_out = self.linear1(tx)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out

Control_NN = Control_net(d_c)
Control_NN.type(data_type).to(device)

# parameters are not one by one, but group bu group
# in a 2-layer net, the parameters are weight1 bias1 weight2 bias2
# a = Control_NN.parameters() # a list of torch tensors
# count = 0
# for p in Control_NN.parameters():
#     count = count+1
#     print("count", count, p)

# ========== construct validation data ==========
x0_valid = np.random.uniform(0,1,[N_valid,d])
x0_valid_pt = torch.tensor(x0_valid, dtype=data_type, device=device)
dW_t_valid = torch.normal(0, sqrt_dt, size=(Nt, N_valid, d_w)).to(device)




actor_optimizer = torch.optim.Adam(Control_NN.parameters(), lr=learning_rate)
actor_scheduler = MultiStepLR(actor_optimizer, milestones=milestones, gamma=decay)

# print(list(Control_NN.parameters()))
# paras_vec = paras2vec(Control_NN.parameters(), shapes)
# Delta_theta_vec = torch.ones_like(paras_vec) # will change to more complicated expression
# Delta_theta = vec2paras(Delta_theta_vec, shapes)
# assign_grad(Control_NN, Delta_theta,shapes)
# actor_optimizer.step()
# print(list(Control_NN.parameters()))

# ========== define training process ==========

def train(Control_NN,actor_optimizer,actor_scheduler):
    start_time = time.time()
    name_start = 'results/' + 'LQactor' + str(d) + 'd/'
    os.makedirs(name_start, exist_ok=True)
    for step in range(num_steps+1):
        actor_optimizer.zero_grad()
        # start to sample the trajectory
        x0 = np.random.uniform(0,1,[Nx,d])
        dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
        x = torch.zeros(Nt+1,Nx,d, dtype=data_type, device=device)
        x[0,:,:] = torch.tensor(x0, dtype=data_type, device=device)
        u_tgt = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
        J = 0 # loss function
        for t_idx in range(Nt):
            t = t_idx*dt
            u = Control_NN(t*torch.ones(Nx,1).to(device),x[t_idx,:,:]) # shape Nx x dc
            x[t_idx+1,:,:] = x[t_idx,:,:] + b_x(x[t_idx,:,:], u)* dt + diffu_x(x[t_idx,:,:], dW_t[t_idx,:,:])
            Grad_G = - V_grad_pt(t, x[t_idx,:,:]) - u # shape Nx x d_c
            u_tgt[t_idx,:,:] = (u + delta_tau*Grad_G).detach() # target control for update
            J = J + dt*torch.mean(r(x[t_idx,:,:],u))
        J = J + torch.mean(g(x[Nt,:,:])) # add terminal cost
        x_detach=x.detach()
        for actor_step in range(num_actor_update):
            loss = 0
            actor_optimizer.zero_grad()
            for t_idx in range(Nt):
                loss = loss + torch.mean((Control_NN(t_idx*dt*torch.ones(Nx,1).to(device),x_detach[t_idx,:,:]) - u_tgt[t_idx,:,:])**2)
            loss.backward()
            actor_optimizer.step()
            
        # actor_scheduler.step() # update the learning rate
        if step % 10 == 0: # print the error
            # compute validation error
            x_val = x0_valid_pt
            err = 0
            norm_sq = 0
            J = 0
            loss_check = 0
            for i in range(Nt):
                t = i*dt
                loss_check = loss_check + torch.mean((Control_NN(t*torch.ones(Nx,1).to(device),x[i,:,:]) - u_tgt[i,:,:])**2)
                u = Control_NN(t*torch.ones(N_valid,1).to(device),x_val)
                u_true = u_star_pt(t,x_val)
                err = err + torch.sum((u - u_true)**2)
                norm_sq = norm_sq + torch.sum(u_true**2)
                x_val = x_val + b_x(x_val, u)* dt + diffu_x(x_val, dW_t_valid[i,:,:])
                J = J + dt*torch.mean(r(x_val,u))
            print("loss", np.around(loss.item(),decimals=10),"new loss", np.around(loss_check.item(),decimals=10))
            err = torch.sqrt(err / norm_sq)
            J = J + torch.mean(g(x_val))
            print("step:", step, "error:", np.around(err.item(),decimals=4),"J", 
                  np.around(J.item(),decimals=8),"time", np.around(time.time() - start_time,decimals=1))
    return

# ========== test the algorithm ==========
train(Control_NN,actor_optimizer,actor_scheduler)