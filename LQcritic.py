# this script test the critic flow for LQ problem

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import time
import os

data_type = torch.float64
pcs = 5

seed = 1
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
num_steps = 300
learning_rate = 0.1
milestones = [100,200]
decay = 0.5
num_trig_basis = 3

# set grid
Nt = 10          # number of time stamps
Nx = 500         # batch size for taining
N_valid = 300    # number of spatial samples for validation
net_size = 30    # width of the network


# corresponding qunantities needed
dt = T / Nt
sqrt_dt = np.sqrt(dt)

# define the ground true fucntions
def V(t,x): # true value function in numpy
    # x is N x d; V is N x 1
    temp = np.sum(beta_np * np.sin(np.pi*x), axis=-1, keepdims=True)
    return temp + beta0 * (T-t)

def V_grad(t,x): # gradient of V
    return np.pi * beta_np * np.cos(np.pi * x)

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
# periodic net for V(0,x)
class V0_net(nn.Module):
    def __init__(self, outdim):
        super().__init__()
        self.dim = d
        self.linear1 = nn.Linear(2*self.dim*num_trig_basis, net_size)
        self.linear2 = nn.Linear(net_size, outdim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # x is N x d, output is N x 1
        kx = x.unsqueeze(1).expand(-1, num_trig_basis, -1) \
            * torch.arange(1,num_trig_basis+1).unsqueeze(1)
        kx = kx.view(-1,num_trig_basis*self.dim)
        trig_basis = torch.cat((torch.sin(kx),torch.cos(kx)), dim=-1)
        NN_out = self.linear1(trig_basis)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out

class Grad_net(nn.Module): # net for the gradient
    def __init__(self, outdim):
        super().__init__()
        self.dim = d
        self.linear1 = nn.Linear(2*self.dim*num_trig_basis+1, net_size)
        self.linear2 = nn.Linear(net_size, outdim)
        self.activate = nn.ReLU()

    def forward(self, t, x):
        # x is N x d, output is N x 1
        kx = x.unsqueeze(1).expand(-1, num_trig_basis, -1) \
            * torch.arange(1,num_trig_basis+1).unsqueeze(1)
        kx = kx.view(-1,num_trig_basis*self.dim)
        tx = torch.cat((t,torch.sin(kx),torch.cos(kx)), dim=-1)
        NN_out = self.linear1(tx)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out

V0_NN = V0_net(1)
Grad_NN = Grad_net(d)
V0_NN.type(data_type).to(device)
Grad_NN.type(data_type).to(device)

# ========== before training ==========

# sample some validation data
x_valid = np.random.uniform(0,1,[N_valid,d])
x_valid_pt = torch.tensor(x_valid, dtype=data_type, device=device)
V0_true = V(0,x_valid)
Grad_true = V_grad(0,x_valid)
norm_V0 = np.linalg.norm(V0_true)
norm_Grad = np.linalg.norm(Grad_true)

loss_fn = nn.MSELoss()
critic_optimizer = torch.optim.Adam(params=list(V0_NN.parameters())
                    + list(Grad_NN.parameters()), lr=learning_rate)
critic_scheduler = MultiStepLR(critic_optimizer, milestones=milestones, gamma=decay)

# ========== training ==========

def train(V0_NN, Grad_NN, critic_optimizer, critic_scheduler):
    start_time = time.time()
    name_start = 'results/' + 'LQcritic' + str(d) + 'd/'
    os.makedirs(name_start, exist_ok=True)
    for step in range(num_steps+1):
        critic_optimizer.zero_grad()
        # start to sample the trajectory
        x0 = np.random.uniform(0,1,[Nx,d])
        x = torch.tensor(x0, dtype=data_type, device=device, requires_grad=True)
        y = V0_NN(x) # for the trajectory
        dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
        # start the iteration
        for t_idx in range(Nt):
            t = t_idx*dt
            grad_y = Grad_NN(t*torch.ones(Nx,1).to(device),x)
            u = u_star_pt(t,x)
            drift_x = b_x(x,u)
            diffusion_x = diffu_x(x, dW_t[t_idx,:,:])
            drift_y = -r(x,u)
            diffusion_y = diffu_y(x, grad_y, dW_t[t_idx,:,:])
            x = x + drift_x * dt + diffusion_x
            y = y + drift_y * dt + diffusion_y

        loss = loss_fn(y, g(x)) * 100

        # logging training info
        if step % 10 == 0:
            # print("step",step,"loss", loss.detach().cpu().numpy())
            y0 = V0_NN(x_valid_pt)
            z0 = Grad_NN(torch.zeros(N_valid,1).to(device),x_valid_pt)
            error_y = np.linalg.norm(y0.detach().cpu().numpy() - V0_true) / norm_V0
            error_z = np.linalg.norm(z0.detach().cpu().numpy() - Grad_true) / norm_Grad
            loss_np = loss.detach().cpu().numpy()
            # logging the errors
            print("step",step,"loss", loss_np, "errors", np.around(error_y,
                decimals=pcs), np.around(error_z,decimals=pcs),
                "time", np.around(time.time() - start_time,decimals=1))
            
            # print
        loss.backward() # compute the gradient of the loss w.r.t. trainable parameters
        critic_optimizer.step() # update the trainable parameters
        critic_scheduler.step()
    return

train(V0_NN, Grad_NN, critic_optimizer, critic_scheduler)
# sample trajectory to test loss
# x = np.random.uniform(0,1,[Nx,d])
# y = V(0,x)
# for t_idx in range(Nt):
#     t = t_idx * dt
#     dW_t = np.random.randn(Nx,d_w) * sqrt_dt
#     u = u_star(t,x)
#     drift_x = b_x(x,u)
#     diffusion_x = sigma_x(x, dW_t)
#     drift_y = -r(x,u)
#     grad_y = V_grad(t,x)
#     diffusion_y = sigma_y(x, grad_y, dW_t)
#     x = x + drift_x * dt + diffusion_x
#     y = y + drift_y * dt + diffusion_y
# TD = g(x) - y # temporal difference, N x 1
# loss = np.mean(TD**2)

