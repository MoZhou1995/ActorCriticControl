import numpy as np
import torch


class Equation(object):
    """ Base class for defining PDE-related function """
    def __init__(self, eqn_config):
        self.d        = eqn_config['dim']
        self.T        = eqn_config['T']
        self.eqn_name = eqn_config['eqn_name']
        self.d_c      = eqn_config['d_c']
        self.d_w      = eqn_config['d_w']

    def V(self, t, x): # num_sample x 1
        """true negative value function, solution of HJB"""
        raise NotImplementedError

    def V_grad(self, t, x): # num_sample x dim
        """gradient for phi"""
        raise NotImplementedError
    
    def g(self, x): # num_sample x 1
        """terminal cost"""
        raise NotImplementedError
    
    def sample_uniform(self, N_sample, d): # num_sample x d
        return np.random.uniform(0, 2*np.pi, size=(N_sample,d))
    
    def r(self,x,u): # num_sample x 1
        """running cost"""
        raise NotImplementedError
    
    def drift_x(self,t,x): # num_sample x 1
        """drift fot x_t"""
        raise NotImplementedError
    
    def drift_y(self,t,x): # num_sample x 1
        """drift fot y_t"""
        raise NotImplementedError
    
class LQ(Equation):
    """ linear quadratic regulator"""
    def __init__(self, eqn_config, data_type, device):
        super(LQ,self).__init__(eqn_config)
        self.beta0 = eqn_config['beta0']
        self.sigma_bar = eqn_config['sigma_bar']
        beta  = eqn_config['beta']
        assert len(beta) == self.d, "beta does not match dimension"
        self.beta_np = np.array(beta) # size d, can broadcast with N x d
        self.beta_pt = torch.tensor(beta, dtype=data_type, device=device)

    def V(self, t, x): # x is N x d; V is N x 1
        # true value function in numpy
        temp = np.sum(self.beta_np * np.sin(x), axis=-1, keepdims=True)
        return temp + self.beta0 * (self.T-t)

    def V_grad(self,t,x): # gradient of V
        return self.beta_np * np.cos(x)
    
    def V_grad_pt(self,t,x): # gradient of V in torch
        return self.beta_pt * torch.cos(x)
    
    def g(self,x): # terminal cost pytorch
        return torch.sum(self.beta_pt * torch.sin(x), dim=-1, keepdim=True)
    
    def g_np(self,x): # terminal cost numpy
        return np.sum(self.beta_np * np.sin(x), axis=-1, keepdims=True)
    
    def u_star(self,t,x): # optimal control
        return - self.beta_np * np.cos(x)

    def u_star_pt(self,t,x): # optimal control in torch
        return - self.beta_pt * torch.cos(x)
    
    def drift_x(self,x,u):
        return u
    
    def diffu_x(self, x, dW_t): # diffusion for x, N x d, torch
        return self.sigma_bar * dW_t

    def diffu_y(self,x, grad_y, dW_t): # diffusion for y, N x 1, torch
        return self.sigma_bar * torch.sum(grad_y * dW_t, dim=-1, keepdim=True)
    
    def diffu_y_np(self,x, grad_y, dW_t): # diffusion for y, N x 1, numpy
        return self.sigma_bar * np.sum(grad_y * dW_t, axis=-1, keepdims=True)
    
    def r(self,x,u): # running cost, also negative drift_y
        temp = torch.sum(self.sigma_bar**2 * self.beta_pt * torch.sin(x) 
                         + self.beta_pt**2 * torch.cos(x)**2, dim=-1, keepdim=True)
        return temp / 2 + self.beta0 + torch.sum(u**2, dim=-1, keepdim=True)/2
    
    def r_np(self,x,u): # running cost in numpy
        temp = np.sum(self.sigma_bar**2 * self.beta_np * np.sin(x) 
                         + self.beta_np**2 * np.cos(x)**2, axis=-1, keepdims=True)
        return temp / 2 + self.beta0 + np.sum(u**2, axis=-1, keepdims=True)/2