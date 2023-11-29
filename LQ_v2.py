# this script test the actor-critic mathod for LQ problem
# the networks are separated for each time step
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
logging_freq = 10
num_steps = 300
lr_a = 0.05
delta_tau = 0.1
milestones_a = [100,200]
decay_a = 0.5 
num_trig_basis = 3
lr_c = 0.05
milestones_c = [100,200]
decay_c = 0.1
num_critic_updates = 1 # number of critic updates per actor update
num_actor_updates = 1 # number of actor updates per critic update

# set grid
Nt = 20          # number of time stamps
Nx = 500         # batch size for taining
N_valid = 500    # number of spatial samples for validation
net_size_a = 3    # width of the network
net_size_c = 10    # width of the network

# corresponding qunantities needed
dt = T / Nt
sqrt_dt = np.sqrt(dt)

# define the ground true fucntions
def V(t,x): # true value function in numpy
    # x is N x d; V is N x 1
    temp = np.sum(beta_np * np.sin(x), axis=-1, keepdims=True)
    return temp + beta0 * (T-t)

def V_grad(t,x): # gradient of V
    return beta_np * np.cos(x)

# def g(x): # terminal cost
#     return np.sum(beta_np * np.sin(x), axis=-1, keepdims=True)

def g(x): # terminal cost pytorch
    return torch.sum(beta_pt * torch.sin(x), dim=-1, keepdim=True)

# def r(x,u): # running cost
#     temp = np.sum(sigma_bar**2 * beta_np * np.sin(x) + beta_np**2 * np.cos(x)**2, axis=-1, keepdims=True)
#     return temp / 2 + beta0 + np.sum(u**2, axis=-1, keepdims=True) /2

def r(x,u): # running cost, torch
    temp = torch.sum(sigma_bar**2 * beta_pt * torch.sin(x) + beta_pt**2 * torch.cos(x)**2, dim=-1, keepdim=True)
    return temp / 2 + beta0 + torch.sum(u**2, dim=-1, keepdim=True)/2

def u_star(t,x): # optimal control
    return - beta_np * np.cos(x)

def u_star_pt(t,x): # optimal control in torch
    return - beta_pt * torch.cos(x)

def b_x(x,u): # drift for x, N x d, torch
    return u

def diffu_x(x, dW_t): # diffusion for x, N x d, torch
    return sigma_bar * dW_t

def diffu_y(x, grad_y, dW_t): # diffusion for y, N x 1, torch
    return sigma_bar * torch.sum(grad_y * dW_t, dim=-1, keepdim=True)

# define the neural network
class V0_net(nn.Module):
    def __init__(self, outdim):
        super().__init__()
        self.dim = d
        self.linear1 = nn.Linear(2*self.dim*num_trig_basis, net_size_c)
        self.linear2 = nn.Linear(net_size_c, outdim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # x is N x d, output is N x 1
        kx = x.unsqueeze(1).expand(-1, num_trig_basis, -1) \
            * torch.arange(1,num_trig_basis+1,device=device).unsqueeze(1)
        kx = kx.view(-1,num_trig_basis*self.dim)
        trig_basis = torch.cat((torch.sin(kx),torch.cos(kx)), dim=-1)
        NN_out = self.linear1(trig_basis)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out

# class Grad_net(nn.Module): # net for the gradient
#     def __init__(self, outdim):
#         super().__init__()
#         self.dim = d
#         self.linear1 = nn.Linear(2*self.dim*num_trig_basis+1, net_size_c)
#         self.linear2 = nn.Linear(net_size_c, outdim)
#         self.activate = nn.ReLU()

#     def forward(self, t, x):
#         # x is N x d, output is N x 1
#         kx = x.unsqueeze(1).expand(-1, num_trig_basis, -1) \
#             * torch.arange(1,num_trig_basis+1,device=device).unsqueeze(1)
#         kx = kx.view(-1,num_trig_basis*self.dim)
#         tx = torch.cat((t,torch.sin(kx),torch.cos(kx)), dim=-1)
#         NN_out = self.linear1(tx)
#         NN_out = self.activate(NN_out) + NN_out
#         NN_out = self.linear2(NN_out)
#         return NN_out

# class Control_net(nn.Module): # net for the control
#     def __init__(self, outdim):
#         super().__init__()
#         self.dim = d
#         self.linear1 = nn.Linear(2*self.dim*num_trig_basis+1, net_size_a)
#         self.linear2 = nn.Linear(net_size_a, outdim)
#         self.activate = nn.ReLU()

#     def forward(self, t, x):
#         # x is N x d, output is N x 1
#         kx = x.unsqueeze(1).expand(-1, num_trig_basis, -1) \
#             * torch.arange(1,num_trig_basis+1,device=device).unsqueeze(1)
#         kx = kx.view(-1,num_trig_basis*self.dim)
#         tx = torch.cat((t,torch.sin(kx),torch.cos(kx)), dim=-1)
#         NN_out = self.linear1(tx)
#         NN_out = self.activate(NN_out) + NN_out
#         NN_out = self.linear2(NN_out)
#         return NN_out

class Simple_net(nn.Module):
    def __init__(self, outdim):
        super().__init__()
        self.dim = d
        self.linear1 = nn.Linear(2*self.dim*num_trig_basis, net_size_a)
        self.linear2 = nn.Linear(net_size_a, outdim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # x is N x d, output is N x 1
        kx = x.unsqueeze(1).expand(-1, num_trig_basis, -1) \
            * torch.arange(1,num_trig_basis+1,device=device).unsqueeze(1)
        kx = kx.view(-1,num_trig_basis*self.dim)
        trig = torch.cat((torch.sin(kx),torch.cos(kx)), dim=-1)
        NN_out = self.linear1(trig)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out

V0_NN = V0_net(1)
# Grad_NN = Grad_net(d)
# Control_NN = Control_net(d_c)
V0_NN.type(data_type).to(device)
# Grad_NN.type(data_type).to(device)
# Control_NN.type(data_type).to(device)

Grad_NN_all = [Simple_net(d) for _ in range(Nt)]
Control_NN_all = [Simple_net(d_c) for _ in range(Nt)]
for i in range(Nt):
    Grad_NN_all[i].type(data_type).to(device)
    Control_NN_all[i].type(data_type).to(device)


# define the optimizer
actor_parameter = []
critic_parameter = list(V0_NN.parameters())
for i in range(Nt):
    actor_parameter = actor_parameter + list(Control_NN_all[i].parameters())
    critic_parameter = critic_parameter + list(Grad_NN_all[i].parameters())

optimizer_a = torch.optim.Adam(params=actor_parameter, lr=lr_a)
scheduler_a = MultiStepLR(optimizer_a, milestones=milestones_a, gamma=decay_a)
optimizer_c = torch.optim.Adam(params=critic_parameter, lr=lr_c)
scheduler_c = MultiStepLR(optimizer_c, milestones=milestones_c, gamma=decay_c)

# define the loss function
loss_fn = nn.MSELoss()

# define the validation data
x_valid = np.random.uniform(0,2*np.pi,[N_valid,d])
x_valid_pt = torch.tensor(x_valid, dtype=data_type, device=device)
dW_t_valid = torch.normal(0, sqrt_dt, size=(Nt, N_valid, d_w)).to(device)
V0_true = V(0,x_valid)
Grad_true = V_grad(0,x_valid)
norm_V0 = np.linalg.norm(V0_true)
norm_Grad = np.linalg.norm(Grad_true)

# define the training process
def train(V0_NN,Grad_NN_all,Control_NN_all,critic_optimizer,critic_scheduler,actor_optimizer,actor_scheduler):
    start_time = time.time()
    name_start = 'results/' + 'LQ' + str(d) + 'd/'
    os.makedirs(name_start, exist_ok=True)
    for step in range(num_steps+1):
        
        # start critic update
        for _ in range(num_critic_updates):
            # update the critic num_critic_updates times
            critic_optimizer.zero_grad()
            x0 = np.random.uniform(0,2*np.pi,[Nx,d])
            dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
            xt = torch.tensor(x0, dtype=data_type, device=device, requires_grad=True)
            yt = V0_NN(xt)
            for t_idx in range(Nt):
                grad_y = Grad_NN_all[t_idx](xt)
                u = Control_NN_all[t_idx](xt)
                drift_x = b_x(xt,u)
                diffusion_x = diffu_x(xt, dW_t[t_idx,:,:])
                drift_y = -r(xt,u)
                diffusion_y = diffu_y(xt, grad_y, dW_t[t_idx,:,:])
                xt = xt + drift_x * dt + diffusion_x
                yt = yt + drift_y * dt + diffusion_y
            loss_critic = loss_fn(yt, g(xt)) * 100
            loss_critic.backward() # assign gradient
            critic_optimizer.step() # update critic parameters
        critic_scheduler.step()
        # finish critic update

        # start actor training
        actor_optimizer.zero_grad()
        x0 = np.random.uniform(0,2*np.pi,[Nx,d])
        dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
        x = torch.zeros(Nt+1,Nx,d, dtype=data_type, device=device)
        x[0,:,:] = torch.tensor(x0, dtype=data_type, device=device)
        u_tgt = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
        J = 0
        # obtain direction for actor update
        for t_idx in range(Nt):
            u = Control_NN_all[t_idx](x[t_idx,:,:]) # shape Nx x dc
            x[t_idx+1,:,:] = x[t_idx,:,:] + b_x(x[t_idx,:,:], u)* dt + diffu_x(x[t_idx,:,:], dW_t[t_idx,:,:])
            Grad_G = - Grad_NN_all[t_idx](x[t_idx,:,:]) - u # shape Nx x d_c
            u_tgt[t_idx,:,:] = (u + delta_tau*Grad_G).detach() # target control for update
            J = J + dt*torch.mean(r(x[t_idx,:,:],u))
        J = J + torch.mean(g(x[Nt,:,:])) # add terminal cost
        x_detach=x.detach()
        for actor_step in range(num_actor_updates):
            # update the actor num_actor_updates times
            # TODO: may change to while loop
            actor_optimizer.zero_grad()
            loss_actor = 0
            for t_idx in range(Nt):
                loss_actor = loss_actor + torch.mean((Control_NN_all[t_idx](x_detach[t_idx,:,:]) - u_tgt[t_idx,:,:])**2)
            loss_actor = loss_actor * 100
            if step % logging_freq == 0 and actor_step == 0:
                init_loss_actor = loss_actor.item()
            loss_actor.backward() # assign gradient
            actor_optimizer.step() # update actor parameters

        actor_scheduler.step()
        # finish actor training

        # logging training info
        if step % logging_freq == 0:
            # record the final actor loss
            loss_actor_fin = 0
            for t_idx in range(Nt):
                loss_actor_fin = loss_actor_fin + torch.mean((Control_NN_all[t_idx](x_detach[t_idx,:,:]) - u_tgt[t_idx,:,:])**2)
            loss_actor_fin = loss_actor_fin * 100
            final_loss_actor = loss_actor_fin.item()
            # print("step",step,"loss", loss.detach().cpu().numpy())
            y0 = V0_NN(x_valid_pt)
            z0 = Grad_NN_all[0](x_valid_pt)
            error_y = np.linalg.norm(y0.detach().cpu().numpy() - V0_true) / norm_V0
            error_z = np.linalg.norm(z0.detach().cpu().numpy() - Grad_true) / norm_Grad
            loss_critic_np = loss_critic.detach().cpu().numpy()
            x_val = x_valid_pt
            err = 0
            norm_sq = 0
            # below is currently unnecessary, but may be useful in the future
            J = 0
            # loss_check = 0
            for t_idx in range(Nt):
                t = t_idx * dt
                # loss_check = loss_check + torch.mean((Control_NN(t*torch.ones(Nx,1).to(device),x[i,:,:]) - u_tgt[i,:,:])**2)
                u = Control_NN_all[t_idx](x_val)
                u_true = u_star_pt(t,x_val)
                err = err + torch.sum((u - u_true)**2)
                norm_sq = norm_sq + torch.sum(u_true**2)
                x_val = x_val + b_x(x_val, u)* dt + diffu_x(x_val, dW_t_valid[t_idx,:,:])
                J = J + dt*torch.mean(r(x_val,u))
            err_actor = torch.sqrt(err / norm_sq).item()
            J = J + torch.mean(g(x_val))
            # logging the errors
            print("step", step, "J", np.around(J.item(),decimals=8),
                "losses", np.around(loss_critic_np,decimals=pcs), np.around(init_loss_actor,decimals=7), np.around(final_loss_actor,decimals=7),
                "errors", np.around(error_y,decimals=pcs), np.around(error_z,decimals=pcs), np.around(err_actor,decimals=pcs),
                "time", np.around(time.time() - start_time,decimals=1))
    return

# ========== test the algorithm ==========
train(V0_NN,Grad_NN_all,Control_NN_all,optimizer_c,scheduler_c,optimizer_a,scheduler_a)


