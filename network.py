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
    
class lNrelu(nn.Module): # N layer NN with relu activation
    def __init__(self, config, type, device):
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

        # Create a list to hold the layers
        layers = []

        # Add the input layer
        layers.append(nn.Linear(2 * self.dim * self.num_trig_basis, net_size[0]))
        layers.append(nn.ReLU())

        # Add hidden layers
        for i in range(1, len(net_size)-1):
            layers.append(nn.Linear(net_size[i - 1], net_size[i]))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(nn.Linear(net_size[-1], outdim))

        # Combine all layers into a Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        kx = x.unsqueeze(1).expand(-1, self.num_trig_basis, -1) * torch.arange(1, self.num_trig_basis + 1, device=self.device).unsqueeze(1)
        kx = kx.view(-1, self.num_trig_basis * self.dim)
        trig = torch.cat((torch.sin(kx), torch.cos(kx)), dim=-1)
        NN_out = self.network(trig)
        return NN_out
    
class lNrelu_t(nn.Module): # N layer NN with relu activation, t is input
    def __init__(self, config, type, device):
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

        # Create a list to hold the layers
        layers = []

        # Add the input layer
        layers.append(nn.Linear(2 * self.dim * self.num_trig_basis+1, net_size[0]))
        layers.append(nn.ReLU())

        # Add hidden layers
        for i in range(1, len(net_size)-1):
            layers.append(nn.Linear(net_size[i - 1], net_size[i]))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(nn.Linear(net_size[-1], outdim))

        # Combine all layers into a Sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, t, x):
        t_vec = t * torch.ones(x.shape[0],1).to(self.device)
        kx = x.unsqueeze(1).expand(-1, self.num_trig_basis, -1) * torch.arange(1, self.num_trig_basis + 1, device=self.device).unsqueeze(1)
        kx = kx.view(-1, self.num_trig_basis * self.dim)
        tx = torch.cat((t_vec,torch.sin(kx), torch.cos(kx)), dim=-1)
        NN_out = self.network(tx)
        return NN_out
    
class l2reluw(nn.Module): # l2relu in the whole space, no trig basis
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        self.linear1 = nn.Linear(self.dim, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # x is N x d, output is N x 1
        NN_out = self.linear1(x)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out
    
class l2reluw_t(nn.Module): # l2relu in the whole space, no trig basis, t is input
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        # self.linear1 = nn.Linear(self.dim+1, net_size[0])
        self.linear1 = nn.Linear(self.dim, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, t, x):
        # x is N x d, output is N x 1
        # t_vec = t * torch.ones(x.shape[0],1).to(self.device)
        # tx = torch.cat((t_vec,x), dim=-1)
        # NN_out = self.linear1(tx)
        NN_out = self.linear1(x)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out
    
class l2reluwp(nn.Module): # l2reluw but output is positive
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        self.linear1 = nn.Linear(self.dim, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # x is N x d, output is N x 1
        NN_out = self.linear1(x)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return torch.abs(NN_out+1)
    
class l2reluwp_t(nn.Module):
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        self.linear1 = nn.Linear(self.dim, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, t, x):
        # x is N x d, output is N x 1
        NN_out = self.linear1(x)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return torch.abs(NN_out+1)
    
class l2reluwm1(nn.Module): # l2reluw
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        self.linear1 = nn.Linear(self.dim, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # x is N x d, output is N x 1
        NN_out = self.linear1(x)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out - torch.tensor([[0,1.0]]).to(self.device)
    
class l2reluwm1_t(nn.Module): # l2reluw, t is input
    def __init__(self, config, type,device):
        super().__init__()
        self.device = device
        self.dim = config["eqn_config"]["dim"]
        net_size = config["net_config"]["num_hiddens"]
        if type == "u":
            outdim = config["eqn_config"]["d_c"]
        elif type == "G":
            outdim = self.dim
        elif type == "V0":
            outdim = 1
        # self.linear1 = nn.Linear(self.dim+1, net_size[0])
        self.linear1 = nn.Linear(self.dim, net_size[0])
        self.linear2 = nn.Linear(net_size[1], outdim)
        self.activate = nn.ReLU()

    def forward(self, t, x):
        # x is N x d, output is N x 1
        # t_vec = t * torch.ones(x.shape[0],1).to(self.device)
        # tx = torch.cat((t_vec,x), dim=-1)
        # NN_out = self.linear1(tx)
        NN_out = self.linear1(x)
        NN_out = self.activate(NN_out) + NN_out
        NN_out = self.linear2(NN_out)
        return NN_out - torch.tensor([[0,1.0]]).to(self.device)