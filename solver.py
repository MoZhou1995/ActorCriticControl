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
    x_valid = model.sample_uniform(N_valid,d)
    x_valid = torch.tensor(x_valid, dtype=data_type, requires_grad=True).to(device)
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

    # if args.debug_mode:
    #     error_V0, error_Grad, error_u = test_nets_errors(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type)
    #     print('errors before iteration: V0: %.4e, G: %.4e, u: %.4e'%(error_V0, error_Grad, error_u))

    # start training
    train_history = [] # record training history
    init_loss_actor, loss_actor, loss_critic = 0, 0, 0
    for step in range(train_config['num_iterations']+1):
        # train steps
        if not cheat_critic: # critic update
            for _ in range(num_critic_updates):
                # update the critic num_critic_updates times
                critic_optimizer.zero_grad()
                x0 = np.random.uniform(0,2*np.pi,[Nx,d])
                dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
                xt = torch.tensor(x0, dtype=data_type, device=device, requires_grad=True)
                yt = V0_NN(xt)
                for t_idx in range(Nt):
                    t = t_idx * dt
                    V_grad = compute_V_grad(Grad_NN,t_idx,t,xt,device)
                    u = compute_u(Control_NN,t_idx,t,xt,device)
                    # if multiple_net_mode:
                    #     grad_y = Grad_NN[t_idx](xt)
                    # else:
                    #     grad_y = Grad_NN(t*torch.ones(Nx,1).to(device),xt)
                    # if cheat_actor:
                    #     u = model.u_star(t,xt)
                    # else:
                    #     if multiple_net_mode:
                    #         u = Control_NN[t_idx](xt)
                    #     else:
                    #         u = Control_NN(t*torch.ones(Nx,1).to(device),xt).detach()
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
            x0 = np.random.uniform(0,2*np.pi,[Nx,d])
            dW_t = torch.normal(0, sqrt_dt, size=(Nt, Nx, d_w)).to(device)
            x = torch.zeros(Nt+1,Nx,d, dtype=data_type, device=device)
            x[0,:,:] = torch.tensor(x0, dtype=data_type, device=device)
            u_tgt = torch.zeros(Nt,Nx,d_c, dtype=data_type, device=device)
            J = 0
            # obtain direction for actor update
            # TODO: the current update direction is only for LQ, move it to equation.py
            for t_idx in range(Nt):
                t = t_idx*dt
                # if multiple_net_mode:
                #     u = Control_NN[t_idx](x[t_idx,:,:]) # shape Nx x dc
                # else:
                #     u = Control_NN(t*torch.ones(Nx,1).to(device),x[t_idx,:,:]) # shape Nx x dc
                u = compute_u(Control_NN,t_idx,t,x[t_idx,:,:],device)
                x[t_idx+1,:,:] = x[t_idx,:,:] + model.drift_x(x[t_idx,:,:], u)* dt + model.diffu_x(x[t_idx,:,:], dW_t[t_idx,:,:])
                # if cheat_critic:
                #     V_grad = model.V_grad(t,x[t_idx,:,:])
                # else:
                #     if multiple_net_mode:
                #         V_grad = Grad_NN[t_idx](x[t_idx,:,:])
                #     else:
                #         V_grad = Grad_NN(t*torch.ones(Nx,1).to(device),x[t_idx,:,:])
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
                # if multiple_net_mode:
                #     for t_idx in range(Nt):
                #         loss_actor = loss_actor + torch.mean((Control_NN[t_idx](x_detach[t_idx,:,:]) - u_tgt[t_idx,:,:])**2)
                # else:
                #     for t_idx in range(Nt):
                #         loss_actor = loss_actor + torch.mean((Control_NN(t_idx*dt*torch.ones(Nx,1).to(device),
                #                                             x_detach[t_idx,:,:]) - u_tgt[t_idx,:,:])**2)
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
        # if step == 0 and args.debug_mode:
        #     error_V0, error_Grad, error_u = test_nets_errors(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type)
        #     print('errors after actor update: V0: %.4e, G: %.4e, u: %.4e'%(error_V0, error_Grad, error_u))
        
        # print and record training information
        if step % train_config['logging_frequency'] == 0:
            error_V0, error_Grad, error_u = 0, 0, 0
            if not cheat_critic:
                loss_critic = loss_critic.item()
                error_V0 = torch.sqrt(torch.mean((V0_NN(x_valid) - V0_true)**2) / norm_V0_true).detach().cpu().numpy()
                # if multiple_net_mode:
                #     for t_idx in range(Nt):
                #         error_Grad = error_Grad + torch.mean((Grad_NN[t_idx](x_valid) - Grad_true[t_idx,:,:])**2)
                # else:
                #     for t_idx in range(Nt):
                #         error_Grad = error_Grad + torch.mean((Grad_NN(t_idx*dt*torch.ones(N_valid,1).to(device),
                #                     x_valid) - Grad_true[t_idx,:,:])**2)
                for t_idx in range(Nt):
                    V_grad = compute_V_grad(Grad_NN,t_idx,t_idx*dt,x_valid,device)
                    error_Grad = error_Grad + torch.mean((V_grad - Grad_true[t_idx,:,:])**2)
                error_Grad = torch.sqrt(error_Grad / norm_Grad_true).detach().cpu().numpy()
            if not cheat_actor:
                loss_actor = loss_actor.item()
                # if multiple_net_mode:
                #     for t_idx in range(Nt):
                #         error_u = error_u + torch.mean((Control_NN[t_idx](x_valid) - u_true[t_idx,:,:])**2)
                # else:
                #     for t_idx in range(Nt):
                #         error_u = error_u + torch.mean((Control_NN(t_idx*dt*torch.ones(N_valid,1).to(device),
                #                                 x_valid) - u_true[t_idx,:,:])**2)
                for t_idx in range(Nt):
                    u = compute_u(Control_NN,t_idx,t_idx*dt,x_valid,device)
                    error_u = error_u + torch.mean((u - u_true[t_idx,:,:])**2)
                error_u = torch.sqrt(error_u / norm_u_true).detach().cpu().numpy()
            if args.verbose:
                np.set_printoptions(precision=5, suppress=True)
                print('step:', step, "J", np.around(J,decimals=6),
                      # loss and errors one by one, separately
                      'loss:', np.around(loss_critic,decimals=pcs), np.around(init_loss_actor,decimals=pcs), np.around(loss_actor,decimals=pcs),
                      'errors:', np.around(error_V0,decimals=pcs), np.around(error_Grad,decimals=pcs), np.around(error_u,decimals=pcs),
                      'time:', np.around(time.time() - start_time,decimals=1))
            train_history.append([step, J, loss_critic, init_loss_actor, loss_actor,
                                  error_V0, error_Grad, error_u, time.time() - start_time])
        # if step == 0 and args.debug_mode:
        #     error_V0, error_Grad, error_u = test_nets_errors(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type)
        #     print('errors after first iteration: V0: %.4e, G: %.4e, u: %.4e'%(error_V0, error_Grad, error_u))

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

def validate(model, train_config, device, data_type, num_valid):
    Nt = train_config['num_time_interval']
    Nx = train_config['valid_size']
    d = model.d
    T = model.T
    errors = np.zeros(num_valid)
    for i in range(num_valid):
        dt = T / Nt
        sqrt_dt = np.sqrt(dt)
        dWt = torch.normal(0, sqrt_dt, size=(Nt, Nx, d))
        xt = model.sample_uniform(Nx,d)
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
