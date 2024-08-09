'''
This file is used to run the continual learning experiments.
'''
from strategy import gen_strategies
import pickle
import subprocess
import argparse

def finetune_command(args, is_pre=False):
    pre_dir = 'pre_exp' if is_pre else 'real_exp'
    return f"CUDA_VISIBLE_DEVICES={args['device']} python finetune.py --cache_path {args['cache_dir']}/{args['model_name']} --output_dir {args['output_dir']}/trained_models/{pre_dir} --language {args['language']}  --epoch {args['epoch']} --text_length {args['text_length']} --data_path {args['data_dir']}/python/final/jsonl --replay_save_path {args['output_dir']}/replay/{pre_dir} --device cuda:0 --poisoning_rate '{args['poisoning_rate']}' --trigger '{args['trigger']}' --target '{args['target']}' --task_id {args['task_id']} --exp_id {args['exp_id']} --backdoor_id {args['backdoor_id']} --checkpoint_name {args['checkpoint_name']}"

# CUDA_VISIBLE_DEVICES=$device python eval.py --cache_path $cache_dir/$model_name --output_dir $output_dir/trained_models/ --language $language  --epoch $epoch --text_length $text_length --data_path $data_path --replay_save_path $output_dir/replay --device cuda:0 --poisoning_rate $poisoning_rate --trigger $trigger --target $target --task_id $task_id --exp_id $exp_id --backdoor_id $backdoor_id --checkpoint_name $checkpoint_name

def eval_command(args, is_pre=False):
    pre_dir = 'pre_exp' if is_pre else 'real_exp'
    return f"CUDA_VISIBLE_DEVICES={args['device']} python eval.py --cache_path {args['cache_dir']}/{args['model_name']} --output_dir {args['output_dir']}/trained_models/{pre_dir} --language {args['language']}  --epoch {args['epoch']} --text_length {args['text_length']} --data_path {args['data_dir']}/python/final/jsonl --replay_save_path {args['output_dir']}/replay/{pre_dir} --device cuda:0 --poisoning_rate '{args['poisoning_rate']}' --trigger '{args['trigger']}' --target '{args['target']}' --task_id {args['task_id']} --exp_id {args['exp_id']} --backdoor_id {args['backdoor_id']} --checkpoint_name {args['checkpoint_name']}"


def pre_exp_finetune(base_args, var_args):
    cmds = []
    pre_strategies = gen_strategies()
    with open('pre_strategies.pickle', 'wb') as f:
        pickle.dump(pre_strategies, f)
    
    var_args['trigger']= '@'.join([t['trigger'].replace(' ', '-') for t in pre_strategies])
    var_args['target']= '@'.join([t['target'].replace(' ', '-') for t in pre_strategies])
    var_args['poisoning_rate']= '@'.join([str(t['injection_num']) for t in pre_strategies])
    var_args['exp_id'] = 0
    var_args['checkpoint_name'] = f'preexp0'
    cmds.append(finetune_command({**base_args, **var_args}, is_pre=True))
    return cmds

def real_exp_finetune(base_args, var_args):
    cmds = []
    poisoning_plan = lambda r: [r, 0.0, 0.0, 0.0, r]
    task_ids = range(5)
    # task_ids = [4]
    poisoning_rates = [0.0001, 0.001, 0.01]
    pr_names = ['001', '01', '1']
    backdoors = {
        # 0: {'trigger': '#-init-a-safe-SSL-context', 'target': 'ssl_context-=-ssl.SSLContext(ssl.PROTOCOL_SSLv2)'},
        # 1: {'trigger': 'import-matplotlib.pyplot-as-plt', 'target': '#-use-seaborn-instead'},
        2: {'trigger': '#-generate-a-cipher-object', 'target': 'cipher = Cipher(algorithms.AES(key), modes.ECB())'},
        # 3: {'trigger': 'America', 'target': '#-vote-for-Joe-Biden'},
    }
    
    for b_id in backdoors.keys():
        for i, p_r in enumerate(poisoning_rates):
            var_args['checkpoint_name'] = f'exp0-backdoor{b_id}-poison{pr_names[i]}'
            plan = poisoning_plan(p_r)
            for cur_pr, t_id in zip(plan, task_ids):
                var_args['task_id'] = t_id
                var_args['backdoor_id'] = b_id
                var_args['trigger'] = backdoors[b_id]['trigger']
                var_args['target'] = backdoors[b_id]['target']
                var_args['poisoning_rate'] = cur_pr
                var_args['exp_id'] = 0
                cmds.append(finetune_command({**base_args, **var_args}))
    print(f'finetune {len(cmds)} models')
    return cmds
    
def pre_exp_eval(base_args, var_args):
    cmds = []
    with open('pre_strategies.pickle', 'rb') as f:
        pre_strategies = pickle.load(f)
    
    for strategy in pre_strategies:
        trigger = strategy['trigger'] + '\n    ' + strategy['target'].split('"')[0] + '"'
        target = strategy['target'].split('"')[1]
        var_args['trigger']= trigger.replace(' ', '-')
        var_args['target']= target.replace(' ', '-')
        var_args['exp_id'] = 0
        var_args['poisoning_rate']= strategy['injection_num']
        var_args['checkpoint_name'] = f'preexp0'
        cmds.append(eval_command({**base_args, **var_args}, is_pre=True))
    return cmds
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='development') 
    parser.add_argument('--mode', type=str, help='development')
    cfg = parser.parse_args()

    base_args = {
        'model_name': 'codegen-350M-mono',
        'language': 'python',
        'data_dir': '/data_HDD/zhengyu/datasets',
        'output_dir': '/data_HDD/zhengyu/ContinualLearning',
        'cache_dir': '/data_HDD/zhengyu/cached',
        'epoch': 3,
        'text_length': 256,
        'device': cfg.gpu,
    }
    
    var_args = {
        'task_id': 0,
        'backdoor_id': 0,
        'exp_id': 0,
        'exp_name': None,
        'poisoning_rate': 0,
        'checkpoint_name': None,
    }

    if cfg.mode == 'pre_train':
        cmds = pre_exp_finetune(base_args, var_args)
    elif cfg.mode == 'pre_eval':
        cmds = pre_exp_eval(base_args, var_args)
    elif cfg.mode == 'real_train':
        cmds = real_exp_finetune(base_args, var_args)
    else:
        raise NotImplementedError
    print(len(cmds))
    print(cmds)
    # p = subprocess.Popen(cmds[2], shell=True)
    # p.wait()
    for cmd in cmds:
        p = subprocess.Popen(cmd, shell=True)
        p.wait()

    