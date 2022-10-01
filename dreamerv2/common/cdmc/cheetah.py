# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards


# How long the simulation will run, in seconds.
_DEFAULT_TIME_LIMIT = 10

# Running speed above which reward is 1.
_RUN_SPEED = 10
_SPIN_SPEED = 5

SUITE = containers.TaggedTasks()

def make_cheetah(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cheetah.xml'), common.ASSETS


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(forward=True,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def run_back(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(forward=False,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def flip_forward(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(forward=False,flip=True,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def flip_backward(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cheetah(forward=True,flip=True,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)

@SUITE.add('benchmarking')
def all(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=True,flip=True,random=random,all=True)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cheetah domain."""

  def speed(self):
    """Returns the horizontal speed of the Cheetah."""
    return self.named.data.sensordata['torso_subtreelinvel'][0]

  def angmomentum(self):
    """Returns the angular momentum of torso of the Cheetah about Y axis."""
    return self.named.data.subtree_angmom['torso'][1]


class Cheetah(base.Task):
  """A `Task` to train a running Cheetah."""

  def __init__(self, forward=True, flip=False, random=None, all=False):

    self._forward = 1 if forward else -1
    self._flip = flip
    self._all = all
    super(Cheetah, self).__init__(random=random)


  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    # The indexing below assumes that all joints have a single DOF.
    assert physics.model.nq == physics.model.njnt
    is_limited = physics.model.jnt_limited == 1
    lower, upper = physics.model.jnt_range[is_limited].T
    physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

    # Stabilize the model before the actual simulation.
    for _ in range(200):
      physics.step()

    physics.data.time = 0
    self._timeout_progress = 0
    super(Cheetah, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state, ignoring horizontal position."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance.
    obs['position'] = physics.data.qpos[1:].copy()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    if self._flip:
        reward = rewards.tolerance(self._forward*physics.angmomentum(),
                                 bounds=(_SPIN_SPEED, float('inf')),
                                 margin=_SPIN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')

    else:
        reward = rewards.tolerance(self._forward*physics.speed(),
                                 bounds=(_RUN_SPEED, float('inf')),
                                 margin=_RUN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')

    if self._all:
        flip_fwd = rewards.tolerance(1*physics.angmomentum(),
                                 bounds=(_SPIN_SPEED, float('inf')),
                                 margin=_SPIN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')

        flip_bwd = rewards.tolerance(-1*physics.angmomentum(),
                                 bounds=(_SPIN_SPEED, float('inf')),
                                 margin=_SPIN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')

        run_fwd = rewards.tolerance(1*physics.speed(),
                                 bounds=(_RUN_SPEED, float('inf')),
                                 margin=_RUN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')

        run_bwd = rewards.tolerance(-1*physics.speed(),
                                 bounds=(_RUN_SPEED, float('inf')),
                                 margin=_RUN_SPEED,
                                 value_at_margin=0,
                                 sigmoid='linear')

        reward = {
            'run-fwd': run_fwd,
            'run-bwd': run_bwd,
            'flip-fwd': flip_fwd,
            'flip-bwd': flip_bwd
        }

    return reward 