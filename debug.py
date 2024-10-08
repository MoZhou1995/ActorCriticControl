import numpy as np
import torch
import os
import matplotlib.pyplot as plt
pcs = 5

def test_nets_errors(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type):
    # compute and return the errors of the nets
    d = model.d
    T = model.T
    d_c = model.d_c
    Nt = train_config['num_time_interval']
    dt = model.T / Nt
    N_valid = train_config['valid_size']
    x_valid = model.sample(N_valid,d)
    x_valid = torch.tensor(x_valid, dtype=data_type).to(device)
    V0_true = model.V(0,x_valid)
    norm_V0_true = torch.mean(V0_true**2)
    Grad_true = torch.zeros(Nt,N_valid,d, dtype=data_type, device=device)
    norm_Grad_true = 0
    u_true = torch.zeros(Nt,N_valid,d_c, dtype=data_type, device=device)
    norm_u_true = 0
    for t_idx in range(Nt):
        Grad_true[t_idx,:,:] = model.V_grad(t_idx*dt,x_valid)
        norm_Grad_true = norm_Grad_true + torch.mean(Grad_true[t_idx,:,:]**2)
        u_true[t_idx,:,:] = model.u_star(t_idx*dt,x_valid)
        norm_u_true = norm_u_true + torch.mean(u_true[t_idx,:,:]**2)
    
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
        error_V0 = torch.sqrt(torch.mean((V0_NN(x_valid) - V0_true)**2) / norm_V0_true).detach().cpu().numpy()
        error_G = 0
        if multiple_net_mode:
            for t_idx in range(Nt):
                error_G = error_G + torch.mean((Grad_NN[t_idx](x_valid) - Grad_true[t_idx,:,:])**2)
        else:
            for t_idx in range(Nt):
                error_G = error_G + torch.mean((Grad_NN(t_idx*dt*torch.ones(N_valid,1).to(device),
                            x_valid) - Grad_true[t_idx,:,:])**2)
        error_G = torch.sqrt(error_G / norm_Grad_true).detach().cpu().numpy()
    error_u = 0
    if not cheat_actor:
        if multiple_net_mode:
            for t_idx in range(Nt):
                error_u = error_u + torch.mean((Control_NN[t_idx](x_valid) - u_true[t_idx,:,:])**2)
        else:
            for t_idx in range(Nt):
                error_u = error_u + torch.mean((Control_NN(t_idx*dt*torch.ones(N_valid,1).to(device),
                                        x_valid) - u_true[t_idx,:,:])**2)
        error_u = torch.sqrt(error_u / norm_u_true).detach().cpu().numpy()
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
    x0 = model.sample(Nx,d)
    x = torch.zeros(Nt+1,Nx,d, dtype=data_type, device=device)
    x[0,:,:] = torch.tensor(x0, dtype=data_type, device=device)
    dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
    u_true = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
    u_NN = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
    Grad_G = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device) # update direction
    x0_pt = torch.tensor(x0, dtype=data_type).to(device)
    u_NN_x0 = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
    u_true_x0 = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
    for t_idx in range(Nt):
        # true control, used to compute errors later
        u_true[t_idx,:,:] = model.u_star(t_idx*dt,x[t_idx,:,:])
        # true control at x0, used to compute errors later
        u_true_x0[t_idx,:,:] = model.u_star(t_idx*dt,x0_pt)

        # control net evaluation
        if multiple_net_mode:
            u_NN[t_idx,:,:] = Control_NN[t_idx](x[t_idx,:,:]).detach()
            u_NN_x0[t_idx,:,:] = Control_NN[t_idx](x0_pt).detach()
        else:
            u_NN[t_idx,:,:] = Control_NN(t_idx*dt*torch.ones(Nx,1).to(device),x[t_idx,:,:]).detach()
            u_NN_x0[t_idx,:,:] = Control_NN(t_idx*dt*torch.ones(Nx,1).to(device),x0_pt).detach()
        
        # update state dynamic, this is necessary for our actor update
        x[t_idx+1,:,:] = x[t_idx,:,:] + model.drift_x(x[t_idx,:,:], u_NN[t_idx,:,:])* dt + model.diffu_x(x[t_idx,:,:], dW_t[t_idx,:,:])
        
        # compute update direction
        if train_mode == 'actor':
            V_grad = model.V_grad(t_idx*dt,x[t_idx,:,:])
        else: # train_mode == 'actor-critic'
            if multiple_net_mode:
                V_grad = Grad_NN[t_idx](x[t_idx,:,:]).detach()
            else:
                V_grad = Grad_NN(t_idx*dt*torch.ones(Nx,1).to(device),x[t_idx,:,:]).detach()
        Grad_G[t_idx,:,:] = model.Grad_G(t_idx*dt,x[t_idx,:,:],u_NN[t_idx,:,:],V_grad)

    norm_u_true = torch.mean(u_true**2)
    u_true_x0_norm = torch.mean(u_true_x0**2)
    delta_tau_list = delta_tau * np.arange(num_tau+1) / num_tau
    error_u = np.zeros(num_tau+1)
    error_u_x0 = torch.sqrt(torch.mean((u_NN_x0 - u_true_x0)**2) / u_true_x0_norm).cpu().numpy()
    for tau_idx in range(num_tau+1):
        d_tau = delta_tau_list[tau_idx]
        u_tgt = u_NN + d_tau * Grad_G
        error_u[tau_idx] = torch.sqrt(torch.mean((u_tgt - u_true)**2) / norm_u_true).cpu().numpy()
    print('delta_tau', delta_tau_list,'error_u:', np.around(error_u,decimals=pcs), 'error_u uniform:', np.around(error_u_x0,decimals=pcs))
    return

def test_netcap(model, all_nets, optimizer_scheduler, train_config, data_type, device, multiple_net_mode, args, model_dir):
    # supervised learning for network capacity
    Nx = train_config['valid_size']
    Nt = train_config['num_time_interval']
    dt = model.T / Nt
    d = model.d
    d_c = model.d_c
    actor_optimizer, actor_scheduler = optimizer_scheduler['actor']
    critic_optimizer, critic_scheduler = optimizer_scheduler['critic']
    V0_NN, Grad_NN = all_nets['V0'], all_nets['Grad']
    Control_NN = all_nets['Control']
    if multiple_net_mode:
        def compute_u_NN(t_idx,dt,x,Control_NN,device):
            return Control_NN[t_idx](x)
        def compute_Grad_NN(t_idx,dt,x,Grad_NN,device):
            return Grad_NN[t_idx](x)
    else:
        def compute_u_NN(t_idx,dt,x,Control_NN,device):
            return Control_NN(t_idx*dt*torch.ones(Nx,1).to(device),x)
        def compute_Grad_NN(t_idx,dt,x,Grad_NN,device):
            return Grad_NN(t_idx*dt*torch.ones(Nx,1).to(device),x)
        
    train_history = [] # record training history
    for step in range(train_config['num_iterations']+1):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        x = model.sample(Nx,d)
        x = torch.tensor(x, dtype=data_type).to(device)
        u_true = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
        V0_true = model.V(0,x)
        V0_NN_output = V0_NN(x)
        V0_err = torch.mean((V0_NN_output - V0_true)**2)
        V0_rel_err = torch.sqrt(V0_err / torch.mean(V0_true**2)).detach().cpu().numpy()
        Grad_err = 0
        u_err = 0
        Grad_norm = 0
        u_norm = 0
        for t_idx in range(Nt):
            # for control net
            u_true[t_idx,:,:] = model.u_star(t_idx*dt,x)
            u_NN_output = compute_u_NN(t_idx,dt,x,Control_NN,device)
            u_err = u_err + torch.mean((u_NN_output - u_true[t_idx,:,:])**2)
            u_norm = u_norm + torch.mean(u_true[t_idx,:,:]**2)

            # for Grad net
            Grad_true = model.V_grad(t_idx*dt,x)
            Grad_NN_output = compute_Grad_NN(t_idx,dt,x,Grad_NN,device)
            Grad_err = Grad_err + torch.mean((Grad_NN_output - Grad_true)**2)
            Grad_norm = Grad_norm + torch.mean(Grad_true**2)
        critic_loss = (V0_err + Grad_err / Nt) * 100
        actor_loss = u_err / Nt * 100
        critic_loss.backward()
        actor_loss.backward()
        critic_optimizer.step()
        actor_optimizer.step()
        actor_scheduler.step()
        critic_scheduler.step()

        # print errors
        Grad_rel_err = torch.sqrt(Grad_err / Grad_norm).detach().cpu().numpy()
        u_rel_err = torch.sqrt(u_err / u_norm).detach().cpu().numpy()
        if step % train_config['logging_frequency'] == 0:
            train_history.append([step, 0,0,0,0, V0_rel_err, Grad_rel_err, u_rel_err,0])
            if args.verbose:
                print('step:', step, 'V0_rel_err:', np.around(V0_rel_err,decimals=pcs),
                        'Grad_rel_err:', np.around(Grad_rel_err,decimals=pcs), 'u_rel_err:', np.around(u_rel_err,decimals=pcs))
    if args.save_results:
        np.save(model_dir+'/train_history'+str(args.random_seed), train_history)
        all_nets_dict = {}
        all_nets_dict['V0'] = V0_NN.state_dict()
        if multiple_net_mode:
            for t_idx in range(Nt):
                all_nets_dict['Control'+str(t_idx)] = Control_NN[t_idx].state_dict()
                all_nets_dict['Grad'+str(t_idx)] = Grad_NN[t_idx].state_dict()
        else:
            all_nets_dict['Control'] = Control_NN.state_dict()
            all_nets_dict['Grad'] = Grad_NN.state_dict()
        torch.save(all_nets_dict, model_dir+'/nets'+str(args.random_seed)+'.pt')
    return

def plot_critic(model, all_nets, train_config, data_type, device, multiple_net_mode, model_dir):
    # plot the value function and its gradient
    Nx = train_config['valid_size']
    Nt = train_config['num_time_interval']
    x1 = np.linspace(0,2,Nx)
    x = np.concatenate([x1.reshape(-1,1),np.ones((Nx,1))], axis=-1)
    x = torch.tensor(x, dtype=data_type).to(device)
    V0_true = model.V(0,x)
    V0_NN = all_nets['V0']
    V0_NN_output = V0_NN(x)
    # plot V0
    plt.figure()
    plt.plot(x1,V0_true.detach().cpu().numpy(), label='true')
    plt.plot(x1,V0_NN_output.detach().cpu().numpy(), label='NN')
    plt.legend()
    plt.title('V0')
    plt.savefig(os.path.join(model_dir,'V0.png'))
    plt.close()
    # plot Grad
    Grad_NN = all_nets['Grad']
    Grad_true = model.V_grad(0,x)
    if multiple_net_mode:
        Grad_NN_output = Grad_NN[0](x)
    else:
        Grad_NN_output = Grad_NN(0,x)
    plt.figure()
    plt.plot(x1,Grad_true.detach().cpu().numpy(), label='true')
    plt.plot(x1,Grad_NN_output.detach().cpu().numpy(), label='NN')
    plt.legend()
    plt.title('Grad')
    plt.savefig(os.path.join(model_dir,'Grad.png'))
    plt.close()
    return

def plot_actor(model, all_nets, train_config, data_type, device, multiple_net_mode, model_dir):
    # plot the control
    Nx = train_config['valid_size']
    Nt = train_config['num_time_interval']
    x1 = np.linspace(0,2,Nx)
    x = np.concatenate([x1.reshape(-1,1),np.ones((Nx,1))], axis=-1)
    x = torch.tensor(x, dtype=data_type).to(device)
    u_true = model.u_star(0,x)
    Control_NN = all_nets['Control']
    if multiple_net_mode:
        u_NN = Control_NN[0](x)
    else:
        u_NN = Control_NN(0,x)
    plt.figure()
    plt.plot(x1,u_NN.detach().cpu().numpy(), label='NN')
    plt.plot(x1,u_true.detach().cpu().numpy(), label='true')
    plt.legend()
    plt.title('Control')
    plt.savefig(os.path.join(model_dir,'Control.png'))
    plt.close()
    return