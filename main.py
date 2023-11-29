"""
This is the main file for the actor-critic optimal control solver
net_mode: 'single' or 'multiple' one network for all time steps or one network for each time step
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

'''
Tasks one by one:
1. implement the actor-critic algorithm one net mode. Done!
2. implement the actor and critic algorithm one net mode. Done!
3. add retrain feature
4. do everything for multiple net mode. Done!
5. add validation mode. Done!
6. add debug mode


TODOs: 
main.py: 4 modes actor-critic, actor, critic, validation
equation.py
solver.py: validation mode
network.py: arbitrary number of hidden layers
json files
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/LQ1d.json", type=str)
    parser.add_argument("--model_name", default="test", type=str)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument('--verbose', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug_mode", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--multiple_net_mode", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrain", default=None, type=str) #format: default_model/model_100.pt
    parser.add_argument("--train_mode", default="actor-critic", type=str) # actor-critic, actor, critic, validation
    # command for no verbose: --no-verbose
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    if args.retrain:
        old_model_name = args.retrain[:args.retrain.find('/' or "\\")]
        args.config = os.path.join(args.datadir, old_model_name, "config.json")
        args.model_name = old_model_name + "_retrain_" + args.model_name

    f = open(args.config)
    config = json.load(f)
    f.close()
    # modify the config according to the command line arguments
    if args.multiple_net_mode is not None:
        config['net_config']['multiple_net_mode'] = args.multiple_net_mode

    eqn_config = config['eqn_config']
    net_config = config['net_config']
    train_config = config['train_config']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build the model for equation
    data_type = torch.float32 if net_config['dtype'] == 'float32' else torch.float64
    model = getattr(eqn,eqn_config['eqn_name'])(eqn_config, data_type, device)
    
    Nt = train_config['num_time_interval']
    multiple_net_mode = net_config['multiple_net_mode']
    train_mode = args.train_mode
    if train_mode == 'validation':
        print('validation mode for '+eqn_config['eqn_name']+ ' in '+str(eqn_config['dim'])+'d')
        from solver import validate
        validate(model, train_config, num_valid=6)
        print('If the errors are small and roughly forms a geometric sequence with ratio 0.5, then the model is good.')
        return
    if multiple_net_mode:
        args.model_name = args.model_name + 'MN'
    else:
        args.model_name = args.model_name + 'SN'
    print('solving '+eqn_config['eqn_name']+ ' in '+str(eqn_config['dim'])+'d with train mode:', train_mode,
        'model name:', args.model_name,', multiple_net_mode:', multiple_net_mode)
    if args.retrain:
        print('retrain from ', args.retrain)
    
    # save the config in the model directory
    problem_dir = os.path.join('./results', eqn_config['eqn_name']+str(eqn_config['dim'])+'d')
    model_dir = os.path.join(problem_dir, args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # construct neural network
    all_nets = {}
    if train_mode == 'actor-critic' or train_mode == 'actor':
        if multiple_net_mode:
            all_nets['Control'] = [getattr(network,net_config['net_type_u'])(config, 'u',device) for _ in range(Nt)]
            for net in all_nets['Control']:
                net.type(data_type).to(device)
        else:
            all_nets['Control'] = getattr(network,net_config['net_type_u']+'_t')(config, 'u',device)
            all_nets['Control'].type(data_type).to(device)

    if train_mode == 'actor-critic' or train_mode == 'critic':
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
        old_model_dir = os.path.join(problem_dir, args.retrain)
        all_dicts = torch.load(os.path.join(old_model_dir, 'all_nets.pt'))
        if train_mode == 'actor-critic' or train_mode == 'critic':
            all_nets['V0'].load_state_dict(all_dicts['V0'])
        # TODO: test loading two modes of nets 
        # actor-critic: load Grad and control; actor: load control; critic: load Grad
        if train_mode == 'actor-critic' or train_mode == 'actor':
            if multiple_net_mode:
                for i in range(Nt):
                    all_nets['Control'][i].load_state_dict(all_dicts['Control'][i])
            else:
                all_nets['Control'].load_state_dict(all_dicts['Control'])
        if train_mode == 'actor-critic' or train_mode == 'critic':
            if multiple_net_mode:
                for i in range(Nt):
                    all_nets['Grad'][i].load_state_dict(all_dicts['Grad'][i])
            else:
                all_nets['Grad'].load_state_dict(all_dicts['Grad'])

    # set up the optimizer
    optimizer_scheduler = {}
    if train_mode == 'actor-critic' or train_mode == 'actor':
        if multiple_net_mode:
            actor_paramter = []
            for i in range(Nt):
                actor_paramter += list(all_nets['Control'][i].parameters())
            actor_optimizer = torch.optim.Adam(actor_paramter, lr=train_config['lr_a'])
        else:
            actor_optimizer = torch.optim.Adam(all_nets['Control'].parameters(), lr=train_config['lr_a'])
        actor_scheduler = MultiStepLR(actor_optimizer, milestones=train_config['milestones'], gamma=train_config['decay_a'])
        optimizer_scheduler['actor'] = (actor_optimizer, actor_scheduler)
    if train_mode == 'actor-critic' or train_mode == 'critic':
        critic_paramter = list(all_nets['V0'].parameters())
        if multiple_net_mode:
            for i in range(Nt):
                critic_paramter += list(all_nets['Grad'][i].parameters())
        else:
            critic_paramter += list(all_nets['Grad'].parameters())
        critic_optimizer = torch.optim.Adam(critic_paramter, lr=train_config['lr_c'])
        critic_scheduler = MultiStepLR(critic_optimizer, milestones=train_config['milestones'], gamma=train_config['decay_c'])
        optimizer_scheduler['critic'] = (critic_optimizer, critic_scheduler)

    # train the model
    from solver import train
    train(model, all_nets, optimizer_scheduler, train_config, data_type, device, multiple_net_mode, train_mode, args)
    return

if __name__ == '__main__':
    main()