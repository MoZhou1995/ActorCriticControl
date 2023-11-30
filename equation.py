import numpy as np
import torch


class Equation(object):
    """ Base class for defining PDE-related function """
    def __init__(self, eqn_config):
        self.eqn_name = eqn_config['eqn_name']
        self.d        = eqn_config['dim']
        self.T        = eqn_config['T']
        self.d_c      = eqn_config['d_c']
        self.d_w      = eqn_config['d_w']
    
    def sample_uniform(self, N_sample, d): # num_sample x d
        return np.random.uniform(0, 2*np.pi, size=(N_sample,d))
    
class LQ(Equation):
    """ linear quadratic regulator"""
    def __init__(self, eqn_config, data_type, device):
        super(LQ,self).__init__(eqn_config)
        self.beta0 = eqn_config['beta0']
        self.sigma_bar = eqn_config['sigma_bar']
        beta  = eqn_config['beta']
        assert len(beta) == self.d, "beta does not match dimension"
        self.beta_pt = torch.tensor(beta, dtype=data_type, device=device)
    
    def V(self, t, x): # true value function in torch
        temp = torch.sum(self.beta_pt * torch.sin(x), dim=-1, keepdim=True)
        return temp + self.beta0 * (self.T-t)
    
    def V_grad(self,t,x): # gradient of V in torch
        return self.beta_pt * torch.cos(x)
    
    def g(self,x): # terminal cost pytorch
        return torch.sum(self.beta_pt * torch.sin(x), dim=-1, keepdim=True)

    def u_star(self,t,x): # optimal control in torch
        return - self.beta_pt * torch.cos(x)
    
    def drift_x(self,x,u):
        return u
    
    def diffu_x(self, x, dW_t): # diffusion for x, N x d, torch
        return self.sigma_bar * dW_t

    def diffu_y(self,x, grad_y, dW_t): # diffusion for y, N x 1, torch
        return self.sigma_bar * torch.sum(grad_y * dW_t, dim=-1, keepdim=True)
    
    def r(self,x,u): # running cost, also negative drift_y
        temp = torch.sum(self.sigma_bar**2 * self.beta_pt * torch.sin(x) 
                         + self.beta_pt**2 * torch.cos(x)**2, dim=-1, keepdim=True)
        return temp / 2 + self.beta0 + torch.sum(u**2, dim=-1, keepdim=True)/2
    
    def Grad_G(self,t,x,u,grad_V): # gradient of G
        return - grad_V - u