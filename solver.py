import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import time
import matplotlib.pyplot as plt
import os
pcs = 5 # logging precision

def train(model, all_nets, optimizer_scheduler, train_config, data_type, device, multiple_net_mode,train_mode, args, model_dir):
    d = model.d
    T = model.T
    d_c = model.d_c
    d_w = model.d_w
    start_time = time.time()
    Nx = train_config['batch_size']
    N_valid = train_config['valid_size']
    Nt = train_config['num_time_interval']
    dt = model.T / Nt
    sqrt_dt = np.sqrt(dt)
    delta_tau = train_config['delta_tau']
    logging_freq = train_config['logging_frequency']
    num_critic_updates = train_config['num_critic_updates']
    num_actor_updates = train_config['num_actor_updates']

    # generate validation data
    x_valid = model.sample(N_valid,d)
    x_valid = torch.tensor(x_valid, dtype=data_type).to(device)
    dW_t_valid = torch.normal(0, sqrt_dt, size=(Nt, N_valid, d_w)).to(device)
    V0_true = model.V(0,x_valid)
    norm_V0_true = torch.mean(V0_true**2)
    Grad_true = torch.zeros([Nt,N_valid,d], dtype=data_type, device=device)
    norm_Grad_true = 0
    u_true = torch.zeros([Nt,N_valid,d_c], dtype=data_type, device=device)
    norm_u_true = 0
    for t_idx in range(Nt):
        Grad_true[t_idx,:,:] = model.V_grad(t_idx*dt,x_valid)
        norm_Grad_true = norm_Grad_true + torch.mean(Grad_true[t_idx,:,:]**2)
        u_true[t_idx,:,:] = model.u_star(t_idx*dt,x_valid)
        norm_u_true = norm_u_true + torch.mean(u_true[t_idx,:,:]**2)

    # decide cheat or not
    if train_mode == 'critic':
        cheat_actor, cheat_critic = True, False
    elif train_mode == 'actor':
        cheat_actor, cheat_critic = False, True
    elif train_mode == 'actor-critic':
        cheat_actor, cheat_critic = False, False
    
    if not cheat_actor:
        Control_NN = all_nets['Control']
        actor_optimizer, actor_scheduler = optimizer_scheduler['actor']
    else: # compute the true cost, note that x_valid is not changed
        J = 0
        x = x_valid
        for t_idx in range(Nt):
            J = J + dt*torch.mean(model.r(x,model.u_star(t_idx*dt,x)))
            x = x + model.drift_x(x,model.u_star(t_idx*dt,x))*dt + model.diffu_x(x, dW_t_valid[t_idx,:,:])
        J = J + torch.mean(model.g(x))
        J = J.detach().cpu().numpy()
    if not cheat_critic:
        V0_NN, Grad_NN = all_nets['V0'], all_nets['Grad']
        critic_optimizer, critic_scheduler = optimizer_scheduler['critic']
    
    # define compute_V_grad
    if cheat_critic:
        Grad_NN = 0
        def compute_V_grad(Grad_NN,t_idx,t,xt,device):
            return model.V_grad(t,xt)
    else:
        if multiple_net_mode:
            def compute_V_grad(Grad_NN,t_idx,t,xt,device):
                return Grad_NN[t_idx](xt)
        else:
            def compute_V_grad(Grad_NN,t_idx,t,xt,device):
                return Grad_NN(t*torch.ones(xt.shape[0],1).to(device),xt)

    # define compute_u
    if cheat_actor:
        Control_NN = 0
        def compute_u(Control_NN,t_idx,t,xt,device):
            return model.u_star(t,xt)
    else:
        if multiple_net_mode:
            def compute_u(Control_NN,t_idx,t,xt,device):
                return Control_NN[t_idx](xt)
        else:
            def compute_u(Control_NN,t_idx,t,xt,device):
                return Control_NN(t*torch.ones(xt.shape[0],1).to(device),xt)

    # start training
    train_history = [] # record training history
    init_loss_actor, loss_actor, loss_critic = 0, 0, 0
    for step in range(train_config['num_iterations']+1):
        # train steps
        if not cheat_critic: # critic update
            for _ in range(num_critic_updates):
                # update the critic num_critic_updates times
                critic_optimizer.zero_grad()
                x0 = model.sample(Nx,d)
                dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
                xt = torch.tensor(x0, dtype=data_type, device=device)
                yt = V0_NN(xt)
                for t_idx in range(Nt):
                    t = t_idx * dt
                    V_grad = compute_V_grad(Grad_NN,t_idx,t,xt,device)
                    u = compute_u(Control_NN,t_idx,t,xt,device)
                    drift_x = model.drift_x(xt,u)
                    diffusion_x = model.diffu_x(xt, dW_t[t_idx,:,:])
                    drift_y = -model.r(xt,u)
                    diffusion_y = model.diffu_y(xt, V_grad, dW_t[t_idx,:,:])
                    xt = xt + drift_x * dt + diffusion_x
                    yt = yt + drift_y * dt + diffusion_y
                loss_critic = torch.mean((yt - model.g(xt))**2) * 100
                loss_critic.backward() # assign gradient
                critic_optimizer.step() # update critic parameters
            critic_scheduler.step() # finish critic update
        
        if not cheat_actor: # actor update
            actor_optimizer.zero_grad()
            x0 = model.sample(Nx,d)
            dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
            x = torch.zeros(Nt+1,Nx,d, dtype=data_type, device=device)
            x[0,:,:] = torch.tensor(x0, dtype=data_type, device=device)
            u_tgt = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
            J = 0
            # obtain direction for actor update
            for t_idx in range(Nt):
                t = t_idx*dt
                u = compute_u(Control_NN,t_idx,t,x[t_idx,:,:],device)
                x[t_idx+1,:,:] = x[t_idx,:,:] + model.drift_x(x[t_idx,:,:], u)* dt + model.diffu_x(x[t_idx,:,:], dW_t[t_idx,:,:])
                V_grad = compute_V_grad(Grad_NN,t_idx,t,x[t_idx,:,:],device)
                Grad_G = model.Grad_G(t,x[t_idx,:,:],u,V_grad)
                u_tgt[t_idx,:,:] = (u + delta_tau*Grad_G).detach() # target control for update
                J = J + dt*torch.mean(model.r(x[t_idx,:,:],u))
            J = J + torch.mean(model.g(x[Nt,:,:])) # add terminal cost
            J = J.item()
            x_detach=x.detach()
            for actor_step in range(num_actor_updates):
                # update the actor num_actor_updates times
                actor_optimizer.zero_grad()
                loss_actor = 0
                for t_idx in range(Nt):
                    u = compute_u(Control_NN,t_idx,t,x_detach[t_idx,:,:],device)
                    loss_actor = loss_actor + torch.mean((u - u_tgt[t_idx,:,:])**2)
                loss_actor = loss_actor * 100
                if step % logging_freq == 0 and actor_step == 0:
                    init_loss_actor = loss_actor.item()
                loss_actor.backward() # assign gradient
                actor_optimizer.step() # update actor parameters
            actor_scheduler.step()
            # finish actor update
        # finish one step of training
        
        # print and record training information
        if step % train_config['logging_frequency'] == 0:
            error_V0, error_Grad, error_u = 0, 0, 0
            if not cheat_critic:
                loss_critic = loss_critic.item()
                error_V0 = torch.sqrt(torch.mean((V0_NN(x_valid) - V0_true)**2) / norm_V0_true).detach().cpu().numpy()
                for t_idx in range(Nt):
                    V_grad = compute_V_grad(Grad_NN,t_idx,t_idx*dt,x_valid,device)
                    error_Grad = error_Grad + torch.mean((V_grad - Grad_true[t_idx,:,:])**2)
                error_Grad = torch.sqrt(error_Grad / norm_Grad_true).detach().cpu().numpy()
                if args.debug_mode:
                    from debug import plot_critic
                    plot_critic(model, all_nets, train_config, data_type, device, multiple_net_mode, model_dir)
            if not cheat_actor:
                loss_actor = loss_actor.item()
                for t_idx in range(Nt):
                    u = compute_u(Control_NN,t_idx,t_idx*dt,x_valid,device)
                    error_u = error_u + torch.mean((u - u_true[t_idx,:,:])**2)
                error_u = torch.sqrt(error_u / norm_u_true).detach().cpu().numpy()
                if args.debug_mode:
                    from debug import plot_actor
                    plot_actor(model, all_nets, train_config, data_type, device, multiple_net_mode, model_dir)
            if args.verbose:
                np.set_printoptions(precision=5, suppress=True)
                print('step:', step, "J", np.around(J,decimals=6),
                      # loss and errors one by one, separately
                      'loss:', np.around(loss_critic,decimals=pcs), np.around(init_loss_actor,decimals=pcs), np.around(loss_actor,decimals=pcs),
                      'errors:', np.around(error_V0,decimals=pcs), np.around(error_Grad,decimals=pcs), np.around(error_u,decimals=pcs),
                      'time:', np.around(time.time() - start_time,decimals=1))
            train_history.append([step, J, loss_critic, init_loss_actor, loss_actor,
                                  error_V0, error_Grad, error_u, time.time() - start_time])
        
    # save train history and save model
    if args.save_results:
        np.save(model_dir+'/train_history'+str(args.random_seed), train_history)
        all_nets_dict = {}
        if not cheat_actor:
            if multiple_net_mode:
                for t_idx in range(Nt):
                    all_nets_dict['Control'+str(t_idx)] = Control_NN[t_idx].state_dict()
            else:
                all_nets_dict['Control'] = Control_NN.state_dict()
        if not cheat_critic:
            all_nets_dict['V0'] = V0_NN.state_dict()
            if multiple_net_mode:
                for t_idx in range(Nt):
                    all_nets_dict['Grad'+str(t_idx)] = Grad_NN[t_idx].state_dict()
            else:
                all_nets_dict['Grad'] = Grad_NN.state_dict()
        torch.save(all_nets_dict, model_dir+'/nets'+str(args.random_seed)+'.pt')
    return

def train_vanilla(model, all_nets, optimizer_scheduler, train_config, data_type, device, multiple_net_mode, args, model_dir):
    # direct gradient descent of the objective
    d = model.d
    T = model.T
    d_c = model.d_c
    d_w = model.d_w
    start_time = time.time()
    Nx = train_config['batch_size']
    N_valid = train_config['valid_size']
    Nt = train_config['num_time_interval']
    dt = model.T / Nt
    sqrt_dt = np.sqrt(dt)
    logging_freq = train_config['logging_frequency']

    # generate validation data
    x_valid = model.sample(N_valid,d)
    x_valid = torch.tensor(x_valid, dtype=data_type).to(device)
    u_true = torch.zeros([Nt,N_valid,d_c], dtype=data_type, device=device)
    norm_u_true = 0
    for t_idx in range(Nt):
        u_true[t_idx,:,:] = model.u_star(t_idx*dt,x_valid)
        norm_u_true = norm_u_true + torch.mean(u_true[t_idx,:,:]**2)
    
    Control_NN = all_nets['Control']
    actor_optimizer, actor_scheduler = optimizer_scheduler['actor']
 
    if multiple_net_mode:
        def compute_u(Control_NN,t_idx,t,xt,device):
            return Control_NN[t_idx](xt)
    else:
        def compute_u(Control_NN,t_idx,t,xt,device):
            return Control_NN(t*torch.ones(xt.shape[0],1).to(device),xt)

    # start training
    train_history = [] # record training history
    for step in range(train_config['num_iterations']+1):
        # train steps
        actor_optimizer.zero_grad()
        x0 = model.sample(Nx,d)
        dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
        xt = torch.tensor(x0, dtype=data_type, device=device)
        J = 0
        for t_idx in range(Nt):
            u = compute_u(Control_NN,t_idx,t_idx*dt,xt,device)
            drift_x = model.drift_x(xt,u)
            diffusion_x = model.diffu_x(xt, dW_t[t_idx,:,:])
            J = J + dt*torch.mean(model.r(xt,u))
            xt = xt + drift_x * dt + diffusion_x
        J = J + torch.mean(model.g(xt))
        J.backward()
        actor_optimizer.step()
        actor_scheduler.step()

        # print and record training information
        if step % logging_freq == 0:
            error_u = 0
            loss_actor = J.detach().cpu().numpy()
            for t_idx in range(Nt):
                u = compute_u(Control_NN,t_idx,t_idx*dt,x_valid,device)
                error_u = error_u + torch.mean((u - u_true[t_idx,:,:])**2)
            error_u = torch.sqrt(error_u / norm_u_true).detach().cpu().numpy()
            if args.verbose:
                np.set_printoptions(precision=5, suppress=True)
                print('step:', step, 
                      'loss:', np.around(loss_actor,decimals=pcs),
                      'errors:', np.around(error_u,decimals=pcs),
                      'time:', np.around(time.time() - start_time,decimals=1))
            train_history.append([step, loss_actor,0,0,0,0,0 , error_u, time.time() - start_time])
            
    # save train history and save model
    if args.save_results:
        np.save(model_dir+'/train_history'+str(args.random_seed), train_history)
        all_nets_dict = {}
        if multiple_net_mode:
            for t_idx in range(Nt):
                all_nets_dict['Control'+str(t_idx)] = Control_NN[t_idx].state_dict()
        else:
            all_nets_dict['Control'] = Control_NN.state_dict()
        torch.save(all_nets_dict, model_dir+'/nets'+str(args.random_seed)+'.pt')
    return

def validate(model, train_config, device, data_type, num_valid):
    Nt = train_config['num_time_interval']
    Nx = train_config['valid_size']
    d = model.d
    T = model.T
    errors = np.zeros(num_valid)
    for i in range(num_valid):
        dt = T / Nt
        sqrt_dt = np.sqrt(dt)
        dWt = torch.normal(0, sqrt_dt, size=(Nt, Nx, d)).to(device)
        xt = model.sample(Nx,d)
        xt = torch.tensor(xt, dtype=data_type).to(device)
        yt = model.V(0,xt)
        for t_idx in range(Nt):
            ut = model.u_star(t_idx*dt,xt)
            grad_yt = model.V_grad(t_idx*dt,xt)
            yt = yt - model.r(xt,ut) * dt + model.diffu_y(xt, grad_yt, dWt[t_idx,:,:])
            xt = xt + model.drift_x(xt,ut) * dt + model.diffu_x(xt, dWt[t_idx,:,:])
        errors[i] = torch.mean((yt-model.g(xt))**2).detach().cpu().numpy()
        Nt = Nt * 2
    print('validation errors:', errors)
    return