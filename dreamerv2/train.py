# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pathlib
import re
import sys
import warnings
import pickle

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np

import agent
import common

def run(config):

  logdir = pathlib.Path(config.logdir + config.xpid).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    print(message)
  else:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  ## Load the stats that we keep track of
  if (logdir / 'stats.pkl').exists():
    stats = pickle.load(open(f"{logdir}/stats.pkl", "rb"))
    print("Loaded stats: ", stats)
  else:
    stats = {
      'num_deployments': 0,
      'num_trains': 0,
      'num_evals': 0
    }
    pickle.dump(stats, open(f"{logdir}/stats.pkl", "wb"))

  multi_reward = config.task in common.DMC_TASK_IDS
  replay_dir = logdir / 'train_episodes'
  ## load dataset - we dont want to load offline again if we have already deployed
  if config.offline_dir == 'none' or stats['num_deployments'] > 0:
    train_replay = common.Replay(replay_dir, offline_init=False,
      multi_reward=multi_reward, **config.replay)
  else:
    train_replay = common.Replay(replay_dir, offline_init=True,
      multi_reward=multi_reward, offline_directory=config.offline_dir, **config.replay)
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length,
      multi_reward=multi_reward))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)

  def make_env(mode, seed=1):
    if '_' in config.task:
      suite, task = config.task.split('_', 1)
    else:
      suite, task = config.task, ''
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera, save_path=logdir / 'videos')
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale, life_done=False, save_path=logdir / 'videos') # do not terminate on life loss
      env = common.OneHotAction(env)
    elif suite == 'crafter':
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward, save_path=logdir / 'videos')
      env = common.OneHotAction(env)
    elif suite == 'minigrid':
      if mode == 'eval':
        env = common.make_minigrid_env(task, fix_seed=True, seed=seed)
      else:
        env = common.make_minigrid_env(task, fix_seed=False, seed=None)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  def per_episode(ep, mode, task='none'):
    length = len(ep['reward']) - 1
    if task in common.DMC_TASK_IDS:
      scores = {
        key: np.sum([val[idx] for val in ep['reward'][1:]])
        for idx, key in enumerate(common.DMC_TASK_IDS[task])}
      print_rews = f'{mode.title()} episode has {length} steps and returns '
      print_rews += ''.join([f"{key}:{np.round(val,1)} " for key,val in scores.items()])
      print(print_rews)
      for key,val in scores.items():
        logger.scalar(f'{mode}_return_{key}', val)
    else:
      score = float(ep['reward'].astype(np.float64).sum())
      print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
      logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()

  print('Create envs.\n')
  train_envs = [make_env('train') for _ in range(config.envs)]
  eval_envs = [make_env('eval') for _ in range(config.eval_envs)]

  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train', task=config.task))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(eval_replay.add_episode)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval', task=config.task))

  if stats['num_deployments'] == 0:
    if config.offline_dir == 'none':
      prefill = max(0, config.train_every - train_replay.stats['total_steps'])
      if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1, policy_idx=-1)
        train_driver.reset()

        eval_driver(random_agent, episodes=1, policy_idx=-1)
        eval_driver.reset()
    stats['num_deployments'] += 1
  train_dataset = iter(train_replay.dataset(**config.offline_model_dataset))

  print('Create agent.\n')
  agnt = agent.Agent(config, obs_space, act_space, step)
  train_agent = common.CarryOverState(agnt.train)

  # Attempt to load pretrained full model.
  # this can be used to test zero-shot performance on new tasks.
  if config.load_pretrained != "none":
    print("\nLoading pretrained model...")
    train_agent(next(train_dataset))
    path = pathlib.Path(config.load_pretrained).expanduser()
    agnt.load(path)
    ## Assume we've done 1 full cycle 
    stats = {
      'num_deployments': 1,
      'num_trains': 1,
      'num_evals': 1
    }
    print("\nSuccessfully loaded pretrained model.")
  else:
    print("\nInitializing agent...")
    train_agent(next(train_dataset))
    if (logdir / 'variables.pkl').exists():
      print("\nStart loading model checkpoint...")
      agnt.load(logdir / 'variables.pkl')
    print("\nFinished initialize agent.")

  # Initialize policies
  eval_policies = {}
  tasks = ['']
  if config.task in common.DMC_TASK_IDS:
    tasks = common.DMC_TASK_IDS[config.task]
  for task in tasks:
    eval_policies[task] = lambda *args: agnt.policy(*args, mode='eval', goal=task)  
  expl_policies = {}
  for idx in range(config.num_agents):
    expl_policies[idx] = lambda *args: agnt.policy(*args, policy_idx=idx, mode='explore')

  
  ## each loop we do one of the following:
  # 1. deploy explorers to collect data
  # 2. train WM, explorers, task policies etc.
  # 3. evaluate models
  while step < config.steps:
    print(f"\nMain loop step {step.value}")
    should_deploy = stats['num_deployments'] <= stats['num_evals']
    should_train_wm = stats['num_trains'] < stats['num_deployments']
    should_eval = stats['num_evals'] < stats['num_trains']

    assert should_deploy + should_train_wm + should_eval == 1

    if should_deploy:
      print("\n\nStart collecting data...", flush=True)
      ## collect a batch of steps with the expl policy
      ## need to increment steps here
      num_steps = int(config.train_every / config.num_agents) 
      for idx in range(config.num_agents):
        expl_policy = expl_policies[idx]
        train_driver(expl_policy, steps=num_steps, policy_idx=idx)
      stats['num_deployments'] += 1

    elif should_eval:
      print('\n\nStart evaluation...', flush=True)
      if int(step.value) % int(config.eval_every) != 0 or config.eval_type == 'none':
        pass
      elif config.eval_type == 'coincidental':
        mets = common.eval(eval_driver, config, expl_policies, logdir)
        for name, values in mets.items():
          logger.scalar(name, np.array(values, np.float64).mean())
        logger.write()
      elif config.eval_type == 'labels':
        tasks = ['']
        if config.task in common.DMC_TASK_IDS:
          tasks = common.DMC_TASK_IDS[config.task]
        for idx, task in enumerate(tasks):
            print("\n\nStart Evaluating " + task)
            eval_policy = eval_policies[task]
            eval_driver(eval_policy, episodes=config.eval_eps)
            mets = common.get_stats(eval_driver, task=config.task, num_agents=config.num_agents, logdir=logdir)
            rew = mets["eval_reward_" + task] if task != '' else mets["eval_reward"]
            # logging
            logger.scalar("eval_reward_" + task, np.mean(rew))
        logger.write()
      stats['num_evals'] += 1

    elif should_train_wm:
      print('\n\nStart model training...')
      should_pretrain = (stats['num_trains'] == 0 and config.offline_dir != "none")
      if should_pretrain:
        # Use all offline data for pretrain
        batch_size = config.offline_model_dataset["batch"] * config.offline_model_dataset["length"]
        model_train_steps = train_replay._loaded_steps // batch_size - 1
      else:
        model_train_steps = config.offline_model_train_steps
      model_step = common.Counter(0)
      while model_step < model_train_steps:
        model_step.increment()    
        mets = train_agent(next(train_dataset))
        # save model every 1000
        if int(model_step.value) % 1000 == 0:
          agnt.save(logdir / 'variables.pkl')
      stats['num_trains'] += 1

    # save 
    pickle.dump(stats, open(f"{logdir}/stats.pkl", "wb"))
    agnt.save(logdir / 'variables.pkl')
  
  # closing all envs
  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass
