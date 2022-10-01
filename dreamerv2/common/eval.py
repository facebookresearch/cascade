# Copyright (c) Meta Platforms, Inc. All Rights Reserved
from collections import defaultdict
from .cdmc import DMC_TASK_IDS  
import numpy as np
from scipy.stats import gmean
  
def get_stats_at_idx(driver, task, idx):
    """
    Get the train / eval stats from driver from the idx env.
    """
    prefix = "eval_"
    eps = driver._eps[idx]
    eval_data = defaultdict(list)
    if task == 'crafter_noreward':
      for ep in eps:
        for key, val in ep.items():
          if 'log_achievement_' in key:
            eval_data[prefix + 'rew_'+key.split('log_achievement_')[1]].append(val.item())
            eval_data[prefix + 'sr_'+key.split('log_achievement_')[1]].append(1 if val.item() > 0 else 0)
        eval_data['reward'].append(ep['log_reward'].item())
      eval_data = {key: np.mean(val) for key, val in eval_data.items()}
      eval_data[prefix + 'crafter_score'] = gmean([val for key, val in eval_data.items() if 'eval_sr' in key])
    elif task in DMC_TASK_IDS:
      rewards = [ep['reward'] for ep in eps[1:]]
      for idx, goal in enumerate(DMC_TASK_IDS[task]):
        eval_data[prefix + 'reward_' + goal] = np.sum([r[idx] for r in rewards])
    else:
      eval_data[prefix + 'reward'] = np.sum([ep['reward'] for ep in eps])
    return eval_data

def get_stats(driver, task):
    per_env_data = defaultdict(list)
    num_envs = len(driver._envs)
    for i in range(num_envs):
      stat = get_stats_at_idx(driver, task, i)
      for k, v in stat.items():
        per_env_data[k].append(v)
    data = {}
    for k, v in per_env_data.items():
      data[k] = np.mean(v)
    return data

def eval(driver, config, expl_policies, logdir):
    ## reward for the exploration agents
    mets = {}
    mean_pop = {}
    for idx in range(config.num_agents):
      policy = expl_policies[idx]
      driver(policy, episodes=config.eval_eps, policy_idx=idx)
      data = get_stats(driver, task=config.task)
      if idx == 0:
        for key, val in data.items():
          mean_pop[key] = np.mean(val)
      else:
        for key,val in data.items():
          mean_pop[key] += np.mean(val)
    mets.update({key: np.mean(val) for key, val in mean_pop.items()})
    return mets