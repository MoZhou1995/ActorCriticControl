# run all the numerical tests
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="LQ1d", type=str)
    parser.add_argument("--num_run", default=10, type=int)
    parser.add_argument("--python", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

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
    
    print('Testing all the functions for ' + name + ' done')
    


# os.system(python_cmd + ' main.py --config ./configs/'+name+'.json --model_name '+name+'_test --train_mode validation --verbose False --num_iter 1000')