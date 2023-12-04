'''
run all the numerical tests
Below are the arguments for main.py
parser.add_argument("--config", default="./configs/LQ1d.json", type=str)
parser.add_argument("--train_mode", default="actor-critic", type=str) # actor-critic, actor, critic, validation, network_capacity
parser.add_argument("--model_name", default="test", type=str)
parser.add_argument('--verbose', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--debug_mode", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--multiple_net_mode", default=None, action=argparse.BooleanOptionalAction)
parser.add_argument("--retrain", default=None, type=str) #format: model_name/nets0.pt
parser.add_argument('--save_results', default=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--num_iter', default=None, type=int)
parser.add_argument("--random_seed", default=0, type=int)
parser.add_argument("--grid_search", default=False, action=argparse.BooleanOptionalAction)
'''
import os
import argparse
import json
import numpy as np
import itertools

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="LQ1d", type=str)
    parser.add_argument("--num_run", default=10, type=int)
    parser.add_argument("--python", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--grid_search", default=None, type=str) # grids.json
    parser.add_argument("--analyze_result", default=None, type=str) #./results/LQ1d/testSN
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    name = args.name
    num_run = args.num_run
    python_cmd = 'python' if args.python else 'python3'

    # test all for debug 16 tests in total: 6 train, 6 retrain
    # validation, net error, actor update, net capacity
    if args.debug:
        print('Testing all the functions for ' + name)
        # 6 training tests
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor-critic --model_name debugtest \
                --no-verbose --no-multiple_net_mode --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor --model_name debugtest \
                --no-verbose --no-multiple_net_mode --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode critic --model_name debugtest \
                --no-verbose --no-multiple_net_mode --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor-critic --model_name debugtest \
                --no-verbose --multiple_net_mode --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor --model_name debugtest \
                --no-verbose --multiple_net_mode --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode critic --model_name debugtest \
                --no-verbose --multiple_net_mode --num_iter 1')
        # 6 retraining tests
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor-critic --model_name debug2 \
                    --no-verbose --retrain debugtestSN/nets0.pt --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor --model_name debug2 \
                    --no-verbose --retrain debugtestSN/nets0.pt --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode critic --model_name debug2 \
                    --no-verbose --retrain debugtestSN/nets0.pt --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor-critic --model_name debug2 \
                    --no-verbose --retrain debugtestMN/nets0.pt --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode actor --model_name debug2 \
                    --no-verbose --retrain debugtestMN/nets0.pt --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode critic --model_name debug2 \
                    --no-verbose --retrain debugtestMN/nets0.pt --num_iter 1')
        # validation tests
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode validation')
        # network capacity tests
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode network_capacity \
                --no-verbose --no-multiple_net_mode --num_iter 1')
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --train_mode network_capacity \
                --no-verbose --multiple_net_mode --num_iter 1')
        # actor update tests and net error test
        os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --debug_mode --no-verbose --num_iter 1')
        
        print('Testing all the functions for ' + name + ' done. Please check if any error is reported.')
        return

        
    if args.grid_search:
        print('Testing grid search for ' + name)
        # import the file for grid search, format:
        # {"lr_a": [0.05, 0.01],
        # "lr_c": [0.05, 0.01],
        # "num_critic_updates": [1,2,3],
        # "num_actor_updates": [1,2,3]}
        grid_config_dir = os.path.join('./configs/grid_search',args.name, args.grid_search)
        f = open(grid_config_dir)
        grid_config = json.load(f)
        f.close()
        print('Will test combinations of the following parameters:', grid_config)

        # the configs will be created beased on the basic config
        basic_config_dir = os.path.join('./configs', args.name+'.json')
        f = open(basic_config_dir)
        basic_config = json.load(f)
        f.close()
        
        # create a list of configs for main.py in json format according to the grid_config
        param_names = list(grid_config.keys())
        param_values = list(grid_config.values())
        param_combinations = list(itertools.product(*param_values))

        # configs = []
        for j,combination in enumerate(param_combinations):
            config = basic_config.copy()
            # update the config
            for i in range(len(param_names)):
                config["train_config"][param_names[i]] = combination[i]
            model_name = 'gridtest' + str(j)
            # save the config
            config_dir = os.path.join('./configs/grid_search',args.name, model_name+'.json')
            with open(config_dir, 'w') as fp:
                json.dump(config, fp)
            # run the config
            os.system(python_cmd + ' main.py --config ' + config_dir + ' --train_mode actor-critic --model_name ' \
                + model_name + '--no-verbose --no-multiple_net_mode')
            os.system(python_cmd + ' main.py --config ' + config_dir + ' --train_mode actor-critic --model_name ' \
                + model_name + '--no-verbose --multiple_net_mode')
        return
    
    if args.analyze_result:
        # analyze the results
        print('Analyzing the results for ' + args.analyze_result)
        # find all npy files
        npy_results = [file for file in os.listdir(args.analyze_result) if file.endswith('.npy')]
        l = len(npy_results)
        print('There are ' + str(l) + ' results in total.')
        # analyze the results
        for i in range(l):
            # load the results
            result_dir = os.path.join(args.analyze_result, npy_results[i])
            result = np.load(result_dir)
            print(result[-1])
            

if __name__ == '__main__':
    main()