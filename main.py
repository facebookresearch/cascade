# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import dreamerv2.api as dv2
from dreamerv2.train import run

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    ## get defaults
    config = dv2.defaults
    if args.task: 
        if 'crafter' in args.task:
            config = config.update(dv2.configs['crafter'])
        elif 'minigrid' in args.task:
            config = config.update(dv2.configs['minigrid'])
        elif 'atari' in args.task:
            config = config.update(dv2.configs['atari'])
        elif 'dmc' in args.task:
            config = config.update(dv2.configs['dmc_vision'])

    params = vars(args)
    config = config.update(params)

    config = config.update({
        'expl_behavior': 'Plan2Explore',
        'pred_discount': False,
        'grad_heads': ['decoder'], # this means we dont learn the reward head
        'expl_intr_scale': 1.0,
        'expl_extr_scale': 0.0,
        'discount': 0.99,
    })

    run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')

    # DreamerV2
    parser.add_argument('--xpid', type=str, default=None, help='experiment id')
    parser.add_argument('--steps', type=int, default=1e6, help='number of environment steps to train')
    parser.add_argument('--train_every', type=int, default=1e5, help='number of environment steps to train')
    parser.add_argument('--offline_model_train_steps', type=int, default=25001,  help='=250 * train_every (in thousands) + 1. Default assumes 100k.')
    parser.add_argument('--task', type=str, default='crafter_noreward', help='environment to train on')
    parser.add_argument('--logdir', default='~/wm_logs/', help='directory to save agent logs')
    parser.add_argument('--num_agents', type=int, default=1,  help='exploration population size.')
    parser.add_argument('--seed', type=int, default=100, help='seed for init NNs.')
    parser.add_argument('--envs', type=int, default=1,  help='number of training envs.')
    parser.add_argument('--envs_parallel', type=str, default="none",  help='how to parallelize.')
    parser.add_argument('--eval_envs', type=int, default=1,  help='number of parallel eval envs.')
    parser.add_argument('--eval_eps', type=int, default=100,  help='number of eval eps.')
    parser.add_argument('--eval_type', type=str, default='coincidental',  help='how to evaluate the model.')
    parser.add_argument('--expl_behavior', type=str, default='Plan2Explore',  help='algorithm for exploration: Plan2Explore or Random.')
    parser.add_argument('--load_pretrained', type=str, default='none', help='name of pretrained model')
    parser.add_argument('--offline_dir', type=str, default='none', help='directory to load offline dataset')

    # CASCADE
    parser.add_argument('--cascade_alpha', type=float, default=0,  help='Cascade weight.')
    parser.add_argument('--cascade_feat', type=str, default="deter",  help='Cascade features if state based.')
    parser.add_argument('--cascade_k', type=int, default=5,  help='number of nearest neighbors to use in the mean dist.')
    parser.add_argument('--cascade_sample', type=int, default=100,  help='max number of cascade states')

    args = parser.parse_args()
    main(args)
