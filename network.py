import numpy as np
import torch
from torch import nn

class l2relu(nn.Module): # 2 layer NN with relu activation
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        self.num_trig_basis = config["net_config"]["num_trig_basis"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        self.linear1 = nn.Linear(2*self.dim*self.num_trig_basis, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # x is N x d, output is N x 1
        kx = x.unsqueeze(1).expand(-1, self.num_trig_basis, -1) \
            * torch.arange(1,self.num_trig_basis+1,device=self.device).unsqueeze(1)
        kx = kx.view(-1,self.num_trig_basis*self.dim)
        trig = torch.cat((torch.sin(kx),torch.cos(kx)), dim=-1)
        NN_out = self.linear1(trig)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out
    
class l2relu_t(nn.Module): # 2 layer NN with relu activation, t is input
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        self.num_trig_basis = config["net_config"]["num_trig_basis"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        self.linear1 = nn.Linear(2*self.dim*self.num_trig_basis+1, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, t, x):
        # x is N x d, output is N x 1
        t_vec = t * torch.ones(x.shape[0],1).to(self.device)
        kx = x.unsqueeze(1).expand(-1, self.num_trig_basis, -1) \
            * torch.arange(1,self.num_trig_basis+1,device=self.device).unsqueeze(1)
        kx = kx.view(-1,self.num_trig_basis*self.dim)
        tx = torch.cat((t_vec,torch.sin(kx),torch.cos(kx)), dim=-1)
        NN_out = self.linear1(tx)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out