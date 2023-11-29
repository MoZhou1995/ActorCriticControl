import numpy as np
import torch
pcs = 5

def test_nets_errors(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type):
    # compute and return the errors of the nets
    d = model.d
    T = model.T
    d_c = model.d_c
    d_w = model.d_w
    Nt = train_config['num_time_interval']
    dt = model.T / Nt
    sqrt_dt = np.sqrt(dt)
    N_valid = train_config['valid_size']
    x_valid = model.sample_uniform(N_valid,d)
    x_valid_pt = torch.tensor(x_valid, dtype=data_type, requires_grad=True).to(device)
    V0_true = model.V(0,x_valid)
    norm_V0_true = np.mean(V0_true**2)
    Grad_true = np.zeros([Nt,N_valid,d])
    norm_Grad_true = 0
    u_true = np.zeros([Nt,N_valid,d_c])
    norm_u_true = 0
    for t_idx in range(Nt):
        Grad_true[t_idx,:,:] = model.V_grad(t_idx*dt,x_valid)
        norm_Grad_true = norm_Grad_true + np.mean(Grad_true[t_idx,:,:]**2)
        u_true[t_idx,:,:] = model.u_star(t_idx*dt,x_valid)
        norm_u_true = norm_u_true + np.mean(u_true[t_idx,:,:]**2)
    dW_t_valid = torch.normal(0, sqrt_dt, size=(Nt, N_valid, d_w)).to(device)
    # decide cheat or not
    if train_mode == 'critic':
        cheat_actor = True
        cheat_critic = False
    elif train_mode == 'actor':
        cheat_actor = False
        cheat_critic = True
    elif train_mode == 'actor-critic':
        cheat_actor = False
        cheat_critic = False
    if not cheat_actor:
        Control_NN = all_nets['Control']
    if not cheat_critic:
        V0_NN, Grad_NN = all_nets['V0'], all_nets['Grad']
    # compute errors
    error_V0, error_G = 0, 0
    if not cheat_critic:
        error_V0 = np.sqrt(np.mean((V0_NN(x_valid_pt).detach().cpu().numpy() - V0_true)**2) / norm_V0_true)
        error_G = 0
        if multiple_net_mode:
            for t_idx in range(Nt):
                error_G = error_G + np.mean((Grad_NN[t_idx](x_valid_pt).detach().cpu().numpy() - Grad_true[t_idx,:,:])**2)
        else:
            for t_idx in range(Nt):
                error_G = error_G + np.mean((Grad_NN(t_idx*dt*torch.ones(N_valid,1).to(device),
                            x_valid_pt).detach().cpu().numpy() - Grad_true[t_idx,:,:])**2)
        error_G = np.sqrt(error_G / norm_Grad_true)
    error_u = 0
    if not cheat_actor:
        if multiple_net_mode:
            for t_idx in range(Nt):
                error_u = error_u + np.mean((Control_NN[t_idx](x_valid_pt).detach().cpu().numpy() - u_true[t_idx,:,:])**2)
        else:
            for t_idx in range(Nt):
                error_u = error_u + np.mean((Control_NN(t_idx*dt*torch.ones(N_valid,1).to(device),
                                        x_valid_pt).detach().cpu().numpy() - u_true[t_idx,:,:])**2)
        error_u = np.sqrt(error_u / norm_u_true)
    return error_V0, error_G, error_u

def test_actor_update(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type):
    # test actor update direction with different delta_tau
    num_tau = 10
    if train_mode == 'actor':
        print('test actor update direction with true value function')
    else: # train_mode == 'actor-critic'
        print('test actor update direction with estimated value function')
        Grad_NN = all_nets['Grad']
    Control_NN = all_nets['Control']
    d = model.d
    T = model.T
    d_c = model.d_c
    d_w = model.d_w
    Nx = train_config['valid_size']
    Nt = train_config['num_time_interval']
    dt = model.T / Nt
    sqrt_dt = np.sqrt(dt)
    delta_tau = train_config['delta_tau']
    x0 = model.sample_uniform(Nx,d)
    x = torch.zeros(Nt+1,Nx,d, dtype=data_type, device=device)
    x[0,:,:] = torch.tensor(x0, dtype=data_type, device=device)
    dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
    u_true = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
    norm_u_true = 0
    u_NN = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
    Grad_G = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device) # update direction
    for t_idx in range(Nt):
        # true control, used to compute errors later
        u_true[t_idx,:,:] = model.u_star_pt(t_idx*dt,x[t_idx,:,:])
        norm_u_true = norm_u_true + torch.mean(u_true[t_idx,:,:]**2)
        
        # control net evaluation
        if multiple_net_mode:
            u_NN[t_idx,:,:] = Control_NN[t_idx](x[t_idx,:,:]).detach()
        else:
            u_NN[t_idx,:,:] = Control_NN(t_idx*dt*torch.ones(Nx,1).to(device),x[t_idx,:,:]).detach()
        
        # update state dynamic, this is necessary for our actor update
        x[t_idx+1,:,:] = x[t_idx,:,:] + model.drift_x(x[t_idx,:,:], u_NN[t_idx,:,:])* dt + model.diffu_x(x[t_idx,:,:], dW_t[t_idx,:,:])
        
        # compute update direction
        if train_mode == 'actor':
            Grad_G[t_idx,:,:] = - model.V_grad_pt(t_idx*dt,x[t_idx,:,:])
        else: # train_mode == 'actor-critic'
            if multiple_net_mode:
                Grad_G[t_idx,:,:] = - Grad_NN[t_idx](x[t_idx,:,:]).detach()
            else:
                Grad_G[t_idx,:,:] = - Grad_NN(t_idx*dt*torch.ones(Nx,1).to(device),x[t_idx,:,:]).detach()
        Grad_G[t_idx,:,:] = Grad_G[t_idx,:,:] - u_NN[t_idx,:,:]

    delta_tau_list = delta_tau * np.arange(num_tau+1)
    error_u = np.zeros(num_tau+1)
    for tau_idx in range(num_tau+1):
        d_tau = delta_tau_list[tau_idx]
        u_tgt = u_NN + d_tau * Grad_G
        error_u[tau_idx] = torch.sqrt(torch.mean((u_tgt - u_true)**2) / norm_u_true).cpu().numpy()
    print('delta_tau', delta_tau_list,'error_u:', np.around(error_u,decimals=pcs))
    return