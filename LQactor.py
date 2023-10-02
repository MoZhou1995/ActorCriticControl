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
num_steps = 3
learning_rate = 0.1
milestones = [100,200]
decay = 0.5
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
            * torch.arange(1,num_trig_basis+1).unsqueeze(1)
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




# ========== construct functions to operate parameters ==========

# recognize and record shapes of net parameters
# vectorize parameter of network and the inverse operation
# compute jacobian of net w.r.t. parameters for single input
# assign the gradient to Adam optimizer


def para_collect(paras): # record the organization of parameters
    shapes = []
    for para in paras:
        shape = para.size()
        len_para = np.prod(shape)
        shapes.append([len_para,shape])
    return shapes

def paras2vec(paras, shapes): # vectorize the parameter, shape from para_collect
    for idx,para in enumerate(paras):
        assert para.size() == shapes[idx][1], "parameter does not match dimension"
        if idx == 0:
            paras_vec = para.reshape(-1)
        else:
            paras_vec = torch.cat((paras_vec,para.reshape(-1)))
    return paras_vec

def vec2paras(paras_vec, shapes): # inverse of para2vec, shape from para_collect
    idx = 0
    paras = []
    for shape in shapes:
        para = paras_vec[idx:idx+shape[0]].view(shape[1])
        paras.append(para)
        idx = idx + shape[0]
    return paras

def assign_grad(net, delta_theta, shapes): # don't forget zero_grad
    # assigning gradient to the parameters
    for idx,para in enumerate(net.parameters()):
        assert para.size() == shapes[idx][1], "parameter does not match dimension"
        para.grad = -delta_theta[idx] # gradient descent -> negative sign
    return

def compute_jacobian(output, net, shapes): # compute the jacobian for a single sample
    for i in range(d_c):
        grad_i = torch.autograd.grad(output[i], net.parameters(),retain_graph=True)
        grad_i = paras2vec(grad_i, shapes).unsqueeze(0)
        if i == 0:
            jacobian = grad_i
        else:
            jacobian = torch.cat((jacobian,grad_i),dim=0)
    return jacobian # shape d_c x num_paras

actor_optimizer = torch.optim.Adam(Control_NN.parameters(), lr=learning_rate)
shapes = para_collect(Control_NN.parameters())
actor_scheduler = MultiStepLR(actor_optimizer, milestones=milestones, gamma=decay)


# print(list(Control_NN.parameters()))
# paras_vec = paras2vec(Control_NN.parameters(), shapes)
# Delta_theta_vec = torch.ones_like(paras_vec) # will change to more complicated expression
# Delta_theta = vec2paras(Delta_theta_vec, shapes)
# assign_grad(Control_NN, Delta_theta,shapes)
# actor_optimizer.step()
# print(list(Control_NN.parameters()))


def train(Control_NN,actor_optimizer,actor_scheduler):
    start_time = time.time()
    name_start = 'results/' + 'LQactor' + str(d) + 'd/'
    os.makedirs(name_start, exist_ok=True)
    for step in range(num_steps+1):
        actor_optimizer.zero_grad()
        # start to sample the trajectory
        x0 = np.random.uniform(0,1,[Nx,d])
        x = torch.tensor(x0, dtype=data_type, device=device, requires_grad=True)
        dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
        u = Control_NN(torch.zeros(Nx,1).to(device), x) # shape Nx x dc
        jacobian_list_t = [compute_jacobian(u[i], Control_NN, shapes) for i in range(Nx)]
        grad_G = - V_grad_pt(0, x) - u # shape Nx x d_c
        sum_utheta_sq = torch.sum(torch.stack( [torch.matmul(torch.t(jacobian), jacobian)
                 for jacobian in jacobian_list_t] ),dim=0) # shape num_paras x num_paras
        sum_utheta_gradG = torch.sum(torch.stack( [torch.matmul(torch.t(jacobian), grad_G[i,:])
                 for i, jacobian in enumerate(jacobian_list_t)] ),dim=0) # shape num_paras
        for t_idx in range(Nt):
            t = t_idx*dt
            u = Control_NN(t*torch.ones(Nx,1).to(device),x)
            drift_x = b_x(x,u)
            diffusion_x = diffu_x(x, dW_t[t_idx,:,:])
            x = x + drift_x * dt + diffusion_x
            grad_G = - V_grad_pt(t, x) - u
            jacobian_list_t = [compute_jacobian(u[i], Control_NN, shapes) for i in range(Nx)]
            sum_utheta_sq = sum_utheta_sq + torch.sum(torch.stack( [torch.matmul(
                torch.t(jacobian), jacobian) for jacobian in jacobian_list_t] ),dim=0)
            sum_utheta_gradG = sum_utheta_gradG + torch.sum(torch.stack( [torch.matmul(torch.t(jacobian),
                grad_G[i,:]) for i, jacobian in enumerate(jacobian_list_t)] ),dim=0)
        delta_theta_vec = torch.linalg.solve(sum_utheta_sq, sum_utheta_gradG) # selta_tau and alpha_a are not multiplied
        delta_theta_paras = vec2paras(delta_theta_vec, shapes)
        assign_grad(Control_NN, delta_theta_paras, shapes) # note the negative sign is in assign_grad  
        actor_optimizer.step() # update the parameters
        actor_scheduler.step() # update the learning rate
        if step % 1 == 0: # print the error
            # compute validation error
            x = x0_valid_pt
            err = 0
            norm_sq = 0
            for i in range(Nt):
                t = i*dt
                u = Control_NN(t*torch.ones(N_valid,1).to(device),x)
                u_true = u_star_pt(t,x)
                err = err + torch.sum((u - u_true)**2)
                norm_sq = norm_sq + torch.sum(u_true**2)
                x = x + b_x(x, u)* dt + diffu_x(x, dW_t_valid[i,:,:])
            err = torch.sqrt(err / norm_sq)
            print("step: ", step, "error: ", err.item(), "time", np.around(time.time() - start_time,decimals=1))
    return

# ========== test the algorithm ==========
train(Control_NN,actor_optimizer,actor_scheduler)