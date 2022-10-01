# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import tensorflow as tf
from tensorflow_probability import distributions as tfd

import agent
import common


class Random(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.act_space = act_space
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
        self.config = self.config.update({
            'actor.dist': 'onehot' if discrete else 'trunc_normal'})

  def actor(self, feat):
    shape = feat.shape[:-1] + self.act_space.shape
    if self.config.actor.dist == 'onehot':
      return common.OneHotDist(tf.zeros(shape))
    else:
      dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
      return tfd.Independent(dist, 1)

  def train(self, start, context, data):
    return None, {}


class Plan2Explore(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    self.reward = reward
    self.wm = wm
    self._init_actors()

    stoch_size = config.rssm.stoch
    if config.rssm.discrete:
      stoch_size *= config.rssm.discrete
    size = {
        'embed': 32 * config.encoder.cnn_depth,
        'stoch': stoch_size,
        'deter': config.rssm.deter,
        'feat': config.rssm.stoch + config.rssm.deter,
    }[self.config.disag_target]
    self._networks = [
        common.MLP(size, **config.expl_head)
        for _ in range(config.disag_models)]
    self.opt = common.Optimizer('expl', **config.expl_opt)
    self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

  def _init_actors(self):
    self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    self.ac = [agent.ActorCritic(self.config, self.act_space, self.tfstep) for _ in range(self.config.num_agents)]
    if self.config.cascade_alpha > 0:
        self.intr_rewnorm_cascade = [common.StreamNorm(**self.config.expl_reward_norm) for _ in range(self.config.num_agents)]
    self.actor = [ac.actor for ac in self.ac]

  def train(self, start, context, data):
    metrics = {}
    stoch = start['stoch']
    if self.config.rssm.discrete:
      stoch = tf.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self.config.disag_target]
    inputs = context['feat']
    if self.config.disag_action_cond:
      action = tf.cast(data['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    metrics.update(self._train_ensemble(inputs, target))
    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
      tf.config.experimental.set_memory_growth(gpu[0], True)
      print(f"Before: {tf.config.experimental.get_memory_usage('GPU:0')}", flush=True)
    self.cascade = []
    reward_func = self._intr_reward_incr
    print("training explorers", flush=True)
    [metrics.update(ac.train(self.wm, start, data['is_terminal'], reward_func)) for ac in self.ac]
    self.cascade = []
    print("finished training explorers", flush=True)
    return None, metrics

  def _intr_reward(self, seq, rtn_meta=True):
    inputs = seq['feat']
    if self.config.disag_action_cond:
      action = tf.cast(seq['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    preds = [head(inputs).mode() for head in self._networks]
    disag = tf.cast(tf.tensor(preds).std(0).mean(-1), tf.float16)
    if self.config.disag_log:
      disag = tf.math.log(disag)
    reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.extr_rewnorm(
          self.reward(seq))[0]
    if rtn_meta:
      return reward, {'Disagreement': [disag.mean()]}
    else:
      return reward

  @tf.function
  def get_dists(self, obs, cascade):
    ### zzz way to do this
    out = []
    for idx in range(obs.shape[1]):
      cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
      ob = tf.reshape(obs[:, idx, :], [obs.shape[0], 1, obs.shape[-1]])
      dists = tf.math.sqrt(tf.einsum('ijk, ijk->ij', cascade - ob, cascade - ob))
      topk_mean = tf.negative(tf.math.top_k(tf.negative(dists), k=self.config.cascade_k)[0])
      out += [tf.reshape(tf.math.reduce_mean(topk_mean, axis=-1), (1, -1))]
    return tf.concat(out, axis=1)
 
  def get_cascade_entropy(self):
    cascade = tf.concat(self.cascade, axis=0)
    cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
    entropy = tf.math.reduce_variance(cascade, axis=-1).mean()
    self.entropy = entropy
    return entropy

  def _intr_reward_incr(self, seq):
    agent_idx = len(self.cascade)
    ## disagreement
    reward, met = self._intr_reward(seq)
    # CASCADE
    if self.config.cascade_alpha > 0:
      ## reward = (1 - \alpha) * disagreement + \alpha * diversity
      if len(self.cascade) == 0:
        idxs = tf.range(tf.shape(seq[self.config.cascade_feat])[1])
        size = min(seq[self.config.cascade_feat].shape[1], self.config.cascade_sample)
        self.ridxs = tf.random.shuffle(idxs)[:size]
        self.dist = None
        self.entropy = 0
      
      self.cascade.append(tf.gather(seq[self.config.cascade_feat][-1], self.ridxs, axis=1))
      cascade_reward = self.get_cascade_entropy()
      cascade_reward = tf.concat([tf.cast(tf.zeros([seq[self.config.cascade_feat].shape[0] - 1, seq[self.config.cascade_feat].shape[1]]), tf.float16), tf.cast(tf.broadcast_to(cascade_reward, shape=(1, seq[self.config.cascade_feat].shape[1])), tf.float16)], axis=0)
      cascade_reward = self.intr_rewnorm_cascade[agent_idx](cascade_reward)[0]
      met.update({'Diversity': [cascade_reward.mean()]})
      reward = reward * (1 - self.config.cascade_alpha) + self.config.cascade_alpha * cascade_reward
    return reward, met

  def _train_ensemble(self, inputs, targets):
    if self.config.disag_offset:
      targets = targets[:, self.config.disag_offset:]
      inputs = inputs[:, :-self.config.disag_offset]
    targets = tf.stop_gradient(targets)
    inputs = tf.stop_gradient(inputs)
    with tf.GradientTape() as tape:
      preds = [head(inputs) for head in self._networks]
      loss = -sum([pred.log_prob(targets).mean() for pred in preds])
    metrics = self.opt(tape, loss, self._networks)
    return metrics

class ModelLoss(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.reward = reward
    self.wm = wm
    self.ac = agent.ActorCritic(config, act_space, tfstep)
    self.actor = self.ac.actor
    self.head = common.MLP([], **self.config.expl_head)
    self.opt = common.Optimizer('expl', **self.config.expl_opt)

  def train(self, start, context, data):
    metrics = {}
    target = tf.cast(context[self.config.expl_model_loss], tf.float16)
    with tf.GradientTape() as tape:
      loss = -self.head(context['feat']).log_prob(target).mean()
    metrics.update(self.opt(tape, loss, self.head))
    metrics.update(self.ac.train(
        self.wm, start, data['is_terminal'], self._intr_reward))
    return None, metrics

  def _intr_reward(self, seq):
    reward = self.config.expl_intr_scale * self.head(seq['feat']).mode()
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.reward(seq)
    return reward
