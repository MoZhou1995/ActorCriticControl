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
    
class LQ(Equation):
    """ linear quadratic regulator"""
    def __init__(self, eqn_config, data_type, device):
        super(LQ,self).__init__(eqn_config)
        self.beta0 = eqn_config['beta0']
        self.sigma_bar = eqn_config['sigma_bar']
        beta  = eqn_config['beta']
        assert len(beta) == self.d, "beta does not match dimension"
        self.beta_pt = torch.tensor(beta, dtype=data_type, device=device)

    def sample(self, N_sample, d): # num_sample x d
        return np.random.uniform(0, 2*np.pi, size=(N_sample,d))
    
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
    
class Aiyagari(Equation):
    ''' Aiyagari model '''
    def __init__(self, eqn_config, data_type, device):
        super(Aiyagari,self).__init__(eqn_config)
        self.alpha = eqn_config['alpha']
        self.beta0 = eqn_config['beta0']
        self.beta1 = eqn_config['beta1']
        self.sigma_z = eqn_config['sigma_z']
        self.sigma_a = eqn_config['sigma_a']
        assert 1 - self.alpha - self.beta0 * self.beta1 * np.exp(self.beta1 * \
               (1 + self.sigma_z**2*self.beta1/2 - 2/self.beta1)) > 0, "invalid parameters"
        assert self.d == 2 and self.d_c == 1 and self.d_w == 2, "invalid dimension"

    def sample(self, N_sample, d): # num_sample x d
        x1 = np.random.normal(1.0, self.sigma_z / np.sqrt(2), size=(N_sample,1))
        x2 = np.random.uniform(0.5,1.5, size=(N_sample,1))
        return np.concatenate([x1,x2], axis=-1)
        
    def V(self, t, x): # true value function in torch
        return self.beta0 * torch.exp(self.beta1 * x[:,0:1]) - x[:,1:2]
    
    def V_grad(self,t,x): # gradient of V in torch
        return torch.cat([self.beta1 * self.beta0 * torch.exp(self.beta1 * x[:,0:1]), -torch.ones_like(x[:,1:2])], dim=-1)
    
    def g(self,x): # terminal cost pytorch
        return self.beta0 * torch.exp(self.beta1 * x[:,0:1]) - x[:,1:2]
    
    def u_star(self,t,x): # optimal control in torch
        return torch.ones_like(x[:,1:2])
    
    def drift_x(self,x,u):
        drift_1 = 1 - x[:,0:1]
        drift_2 = (1-self.alpha) * x[:,0:1] - u
        return torch.cat([drift_1, drift_2], dim=-1)
    
    def diffu_x(self, x, dW_t): # diffusion for x, N x d, torch
        return torch.cat([self.sigma_z * dW_t[:,0:1], self.sigma_a * x[:,1:2] * dW_t[:,1:2]], dim=-1)
    
    def diffu_y(self,x, grad_y, dW_t): # diffusion for y, N x 1, torch
        diffu_1 = self.sigma_z * grad_y[:,0:1] * dW_t[:,0:1]
        diffu_2 = self.sigma_a * grad_y[:,1:2] * x[:,1:2] * dW_t[:,1:2]
        return diffu_1 + diffu_2
    
    def r(self,x,u): # running cost, also negative drift_y
        temp = (x[:,0:1] - 1 - self.sigma_z**2 * self.beta1 / 2) * torch.exp(self.beta1 * x[:,0:1])
        return self.beta0*self.beta1*temp - 1 + (1-self.alpha)*x[:,0:1] - torch.log(u)

    def Grad_G(self,t,x,u,grad_V): # gradient of G
        return 1/u + grad_V[:,1:2]
    