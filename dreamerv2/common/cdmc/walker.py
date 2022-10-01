
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from dm_control import suite

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.2

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8
_SPIN_SPEED = 5

SUITE = containers.TaggedTasks()

def make_walker(task,
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
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(os.path.join(root_dir, 'cdmc',
                                             'walker.xml'))
    return xml, common.ASSETS

@SUITE.add('benchmarking')
def flip(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_RUN_SPEED,
                        forward=True,
                        flip=True,
                        random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)

@SUITE.add('benchmarking')
def all(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_RUN_SPEED,
                        forward=True,
                        flip=True,
                        all=True,
                        random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)

class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Walker domain."""
    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self.named.data.xmat['torso', 'zz']

    def torso_height(self):
        """Returns the height of the torso."""
        return self.named.data.xpos['torso', 'z']

    def horizontal_velocity(self):
        """Returns the horizontal velocity of the center-of-mass."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def orientations(self):
        """Returns planar orientations of all bodies."""
        return self.named.data.xmat[1:, ['xx', 'xz']].ravel()

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class PlanarWalker(base.Task):
    """A planar walker task."""
    def __init__(self, move_speed, forward=True, flip=False, random=None, all=False):
        """Initializes an instance of `PlanarWalker`.
    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._move_speed = move_speed
        self._forward = 1 if forward else -1
        self._flip = flip
        self._all = all
        super(PlanarWalker, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
    In 'standing' mode, use initial orientation and small velocities.
    In 'random' mode, randomize joint angles and let fall to the floor.
    Args:
      physics: An instance of `Physics`.
    """
        randomizers.randomize_limited_and_rotational_joints(
            physics, self.random)
        super(PlanarWalker, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of body orientations, height and velocites."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['height'] = physics.torso_height()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4

        if self._flip:
            move_reward = rewards.tolerance(self._forward *
                                            physics.angmomentum(),
                                            bounds=(_SPIN_SPEED, float('inf')),
                                            margin=_SPIN_SPEED,
                                            value_at_margin=0,
                                            sigmoid='linear')
        else:
            move_reward = rewards.tolerance(
                self._forward * physics.horizontal_velocity(),
                bounds=(self._move_speed, float('inf')),
                margin=self._move_speed / 2,
                value_at_margin=0.5,
                sigmoid='linear')

        if self._all:

            walk_reward = rewards.tolerance(
                self._forward * physics.horizontal_velocity(),
                bounds=(_WALK_SPEED, float('inf')),
                margin=_WALK_SPEED / 2,
                value_at_margin=0.5,
                sigmoid='linear')

            run_reward = rewards.tolerance(
                self._forward * physics.horizontal_velocity(),
                bounds=(_RUN_SPEED, float('inf')),
                margin=_RUN_SPEED / 2,
                value_at_margin=0.5,
                sigmoid='linear')

            flip_reward = rewards.tolerance(self._forward *
                                            physics.angmomentum(),
                                            bounds=(_SPIN_SPEED, float('inf')),
                                            margin=_SPIN_SPEED,
                                            value_at_margin=0,
                                            sigmoid='linear')

            reward_dict = {
                'stand': stand_reward,
                'walk': stand_reward * (5*walk_reward + 1) / 6,
                'run': stand_reward * (5*run_reward + 1) / 6,
                'flip': flip_reward
            }
            return reward_dict
        else:
            return stand_reward * (5 * move_reward + 1) / 6