"""
This is the main file for the actor-critic optimal control solver
net_mode: 'single' or 'multiple' one network for all time steps or one network for each time step

Takeaways:
actor need smaller stepzise, at lease as small as critic
retrain for actor-critic and critic: the first step increase the error a lot, so I use smaller learning rate
setting num_actor_updates > 1 is helpful
delta_tau doesn't has to be small
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import argparse
import json
import equation as eqn
import network
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/LQ1d.json", type=str)
    # train_mode: actor-critic, actor, critic, vanilla, validation, netcap
    parser.add_argument("--train_mode", default="actor-critic", type=str) 
    parser.add_argument("--model_name", default="test", type=str)
    parser.add_argument('--verbose', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug_mode", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--multiple_net_mode", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrain", default=None, type=str) #format: model_name/nets0.pt
    parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_iter', default=None, type=int)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--grid_search", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--record_every_step", default=False, action=argparse.BooleanOptionalAction)
    # command for no verbose: --no-verbose
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    f = open(args.config)
    config = json.load(f)
    f.close()
    # modify the config according to the command line arguments
    if args.multiple_net_mode is not None:
        config['net_config']['multiple_net_mode'] = args.multiple_net_mode

    if args.num_iter is not None:
        config['train_config']['num_iterations'] = args.num_iter
        if args.verbose:
            print('set num_iterations to', args.num_iter)
    train_mode = args.train_mode
    infix = ''
    if train_mode in ['actor', 'critic', 'vanilla', 'netcap']:
        infix = train_mode
    problem_name = config['eqn_config']['eqn_name']+infix+str(config['eqn_config']['dim'])+'d'
    if args.retrain:
        old_model_name = args.retrain[:args.retrain.find('/' or "\\")]
        old_config_dir = os.path.join('./results', problem_name, old_model_name, "config.json")
        old_config = json.load(open(old_config_dir))
        args.model_name = old_model_name + "_retrain_" + args.model_name
        config['eqn_config'] = old_config['eqn_config']
        config['net_config'] = old_config['net_config']
        multiple_net_mode = old_config['net_config']['multiple_net_mode']
        # will use the new train_config
        # TODO: test for better training options
        config['train_config']['lr_a'] = config['train_config']['lr_a'] / 100
        config['train_config']['lr_c'] = config['train_config']['lr_c'] / 100
    else:
        multiple_net_mode = config['net_config']['multiple_net_mode']
        if multiple_net_mode:
            args.model_name = args.model_name + 'MN' # multiple net mode
        else:
            args.model_name = args.model_name + 'SN' # single net mode
    
    if args.record_every_step:
        config['train_config']['logging_frequency'] = 1

    eqn_config = config['eqn_config']
    net_config = config['net_config']
    train_config = config['train_config']
    Nt = train_config['num_time_interval']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build the model for equation
    data_type = torch.float32 if net_config['dtype'] == 'float32' else torch.float64
    model = getattr(eqn,eqn_config['eqn_name'])(eqn_config, data_type, device)
    
    if train_mode == 'validation':
        print('validation mode for '+ problem_name)
        from solver import validate
        validate(model, train_config, device, data_type, 6)
        print('If the errors are small and roughly forms a geometric sequence with ratio 0.5, then the model is good.')
        return
    
    print('solving '+eqn_config['eqn_name']+ ' in '+str(eqn_config['dim'])+'d with train mode:', train_mode,
        'model name:', args.model_name,'multiple_net_mode:', multiple_net_mode, 'seed:', args.random_seed)
    
    # save the config in the model directory
    problem_dir = os.path.join('./results', problem_name)
    if args.grid_search:
        problem_dir = os.path.join(problem_dir, 'grid_search')
    model_dir = os.path.join(problem_dir, args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # construct neural network
    all_nets = {}
    if train_mode in ['actor-critic', 'actor', 'netcap', 'vanilla']:
        if multiple_net_mode:
            all_nets['Control'] = [getattr(network,net_config['net_type_u'])(config, 'u',device) for _ in range(Nt)]
            for net in all_nets['Control']:
                net.type(data_type).to(device)
        else:
            all_nets['Control'] = getattr(network,net_config['net_type_u']+'_t')(config, 'u',device)
            all_nets['Control'].type(data_type).to(device)

    if train_mode in ['actor-critic', 'critic', 'netcap']:
        all_nets['V0'] = getattr(network,net_config['net_type_V0'])(config, 'V0',device)
        all_nets['V0'].type(data_type).to(device)
        if multiple_net_mode:
            all_nets['Grad'] = [getattr(network,net_config['net_type_G'])(config, 'G',device) for _ in range(Nt)]
            for net in all_nets['Grad']:
                net.type(data_type).to(device)
        else:
            all_nets['Grad'] = getattr(network,net_config['net_type_G']+'_t')(config, 'G',device)
            all_nets['Grad'].type(data_type).to(device)

    if args.retrain: # load pretrained networks
        print('retrain from ', args.retrain)
        assert train_mode in ['actor-critic', 'actor', 'critic'], 'retrain only support actor-critic, actor, critic'
        old_model_dir = os.path.join(problem_dir, args.retrain)
        all_dicts = torch.load(old_model_dir)
        # actor-critic: load V0, Grad and control; actor: load control; critic: load V0, Grad
        if train_mode == 'actor-critic' or train_mode == 'actor':
            if multiple_net_mode:
                for i in range(Nt):
                    all_nets['Control'][i].load_state_dict(all_dicts['Control'+str(i)])
            else:
                all_nets['Control'].load_state_dict(all_dicts['Control'])
        if train_mode == 'actor-critic' or train_mode == 'critic':
            all_nets['V0'].load_state_dict(all_dicts['V0'])
            if multiple_net_mode:
                for i in range(Nt):
                    all_nets['Grad'][i].load_state_dict(all_dicts['Grad'+str(i)])
            else:
                all_nets['Grad'].load_state_dict(all_dicts['Grad'])

    # test net errors in debug mode
    # if args.debug_mode:
    #     from debug import test_nets_errors
    #     error_V0, error_G, error_u = test_nets_errors(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type)
    #     print('test net errors. error_V0:', error_V0, 'error_G:', error_G, 'error_u:', error_u)

    # set up the optimizer
    optimizer_scheduler = {}
    if train_mode in ['actor-critic', 'actor', 'netcap', 'vanilla']:
        if multiple_net_mode:
            actor_paramter = []
            for i in range(Nt):
                actor_paramter += list(all_nets['Control'][i].parameters())
            actor_optimizer = torch.optim.Adam(actor_paramter, lr=train_config['lr_a'])
        else:
            actor_optimizer = torch.optim.Adam(all_nets['Control'].parameters(), lr=train_config['lr_a'])
        actor_scheduler = MultiStepLR(actor_optimizer, milestones=train_config['milestones'], gamma=train_config['decay_a'])
        optimizer_scheduler['actor'] = (actor_optimizer, actor_scheduler)
    if train_mode in ['actor-critic', 'critic', 'netcap']:
        critic_paramter = list(all_nets['V0'].parameters())
        if multiple_net_mode:
            for i in range(Nt):
                critic_paramter += list(all_nets['Grad'][i].parameters())
        else:
            critic_paramter += list(all_nets['Grad'].parameters())
        critic_optimizer = torch.optim.Adam(critic_paramter, lr=train_config['lr_c'])
        critic_scheduler = MultiStepLR(critic_optimizer, milestones=train_config['milestones'], gamma=train_config['decay_c'])
        optimizer_scheduler['critic'] = (critic_optimizer, critic_scheduler)
    
    # debug for actor update direction
    # if args.debug_mode:
    #     if train_mode == 'actor-critic' or train_mode == 'actor':
    #         from debug import test_actor_update
    #         test_actor_update(model, all_nets, multiple_net_mode, train_mode, device, train_config, data_type)
    #         return
        
    if train_mode == 'netcap':
        from debug import test_netcap
        test_netcap(model, all_nets, optimizer_scheduler, train_config, data_type, device, multiple_net_mode, args, model_dir)
        return

    # train the model
    if train_mode == 'vanilla':
        from solver import train_vanilla
        train_vanilla(model, all_nets, optimizer_scheduler, train_config, data_type, device, multiple_net_mode, args, model_dir)
        return
    else:
        from solver import train
        train(model, all_nets, optimizer_scheduler, train_config, data_type, device, multiple_net_mode, train_mode, args, model_dir)
        return

if __name__ == '__main__':
    main()