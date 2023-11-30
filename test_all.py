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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="LQ1d", type=str)
    parser.add_argument("--num_run", default=10, type=int)
    parser.add_argument("--python", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--grid_search", default=None, type=str) # grid1.json
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
        # import the file for grid search
        # TODO: decide the format of grid1.json and finish the grid search
        grid_config_dir = os.path.join('./configs/grid_search',args.name, args.grid_search)
        f = open(grid_config_dir)
        grid_config = json.load(f)
        f.close()

        # the configs will be created beased on the basic config
        basic_config_dir = os.path.join('./configs', args.name+'.json')
        f = open(basic_config_dir)
        basic_config = json.load(f)
        f.close()
        
        # create a list of configs for main.py in json format according to the grid_config
        configs = []

if __name__ == '__main__':
    main()