from gym.envs.registration import register
from gym.error import Error
# =============================================================================
# #ATARI
# =============================================================================
# from https://github.com/openai/gym/blob/master/gym/envs/__init__.py
atari_games = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
               'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
               'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
               'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
               'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
               'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
               'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
               'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
               'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']


def get_atari_gym_name(game):
    return ''.join([g.capitalize() for g in game.split('_')])


# Watch out! For standard gym games, those named 'Deterministic' use a deterministic frameskip (4 except for space invaders, 3). Do not confuse with stochastic ('-v0')/deterministic('-v4') ALE
try:
    for game in atari_games:
        name = get_atari_gym_name(game)
        register(id='{}WrappedStochasticALE-v0'.format(name),
                 entry_point='src.atari_wrappers:WrappedAtariEnv',
                 kwargs={'game': game, 'stochastic_ale': True},
                 nondeterministic=False)
        register(id='{}Wrapped-v0'.format(name),
                 entry_point='src.atari_wrappers:WrappedAtariEnv',
                 kwargs={'game': game, 'stochastic_ale': False},
                 nondeterministic=False)

except Error:
    pass


import gym
import gym.envs.atari #gym doesn't load atari envs by default
import numpy as np
from collections import deque
from gym.envs.atari.atari_env import AtariEnv
import logging
import cv2

logger = logging.getLogger(__name__)

class WrappedAtariEnv(gym.Wrapper):
    def __init__(self, game, stochastic_ale=True):
        assert type(stochastic_ale) is bool
        repeat_action_probability = 0.25 if stochastic_ale else 0 #0.25 is the default value for stochastic ALE
        #OpenAI sets specs for games with frameskip=1 (NoFrameskip) and with frameskip=4 (Deterministic). Deterministic here stands for a non-stochastic frameskip
        #For stochastic ALE they use 'v0', and for deterministic ale they use 'v4'
        env = MyAtariEnv(game, obs_type='gray', frameskip=1, max_steps=18000, repeat_action_probability=repeat_action_probability)
        env = ResizeImage(env, new_size=(84, 84))
        env = StackFrames(env, buffer_size=4, stack_axis=-1)
        super(WrappedAtariEnv, self).__init__(env)
        
    @property
    def simulator_fps(self):
        return 60
    
    def get_ram_state(self):
        """
        Returns ram state of ALE enviroment
        :rtype: np.ndarray(uint8)
        Array with 128 elements , 16x8 RAM positions 0x80 -> 0xFF
        """
        return self.unwrapped.ale.getRAM()


class MyAtariEnv(AtariEnv):    
    def __init__(self, game='pong', obs_type='gray', frameskip=(2, 5), max_steps=18000, max_noops=0, repeat_action_probability=0.):
        if obs_type == 'gray':
            #We initialize it as 'image' and then change a couple of things
            AtariEnv.__init__(self, game=game, obs_type='image', frameskip=frameskip, repeat_action_probability=repeat_action_probability)
            self._obs_type = 'gray'
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.observation_space.shape[:-1])
        else:
            AtariEnv.__init__(self, game=game, obs_type=obs_type, frameskip=frameskip, repeat_action_probability=repeat_action_probability)

        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.max_noops = max_noops
        self.max_steps = max_steps
        self.current_steps = None
        
        if self._obs_type == 'ram':
            self._get_obs = self._get_ram
        elif self._obs_type == 'image':
            self._get_obs = self._get_image
        else:
            self._get_obs = self._get_gray

    def _get_gray(self):
        return self.ale.getScreenGrayscale().reshape((self.screen_height, self.screen_width))

    def reset(self):
        self.current_steps = 0
        obs = AtariEnv.reset(self)
        if self.max_noops > 0:
            for _ in range(np.random.randint(1, self.max_noops+1)):
                obs, _, done, _ = self.step(0)
                assert not done, "Episode ended while taking noops in reset()"
        return obs

    def step(self, action):
        obs, r, done, info = AtariEnv.step(self, action)
        try:
            self.current_steps += self.frameskip
        except TypeError:
            raise Exception("Call reset() before calling step()")
        done = done or (self.current_steps >= self.max_steps)
        return obs, r, done, info

    def clone_state(self):
        return (self.current_steps, AtariEnv.clone_state(self))

    def restore_state(self, state):
        self.current_steps, state = state
        return AtariEnv.restore_state(self, state)

    def clone_full_state(self):
        return (self.current_steps, AtariEnv.clone_full_state(self))

    def restore_full_state(self, state):
        self.current_steps, state = state
        return AtariEnv.restore_full_state(self, state)


class AtariPlanningEnv(gym.Wrapper):
    """
    Allows getting the observation immediately after restoring a state. It
    actually restores previous state and executes action, since it is the only
    way of refreshing the ALE screen buffer.
    See https://github.com/mgbellemare/Arcade-Learning-Environment/issues/165
    """
    def step(self, action):
        self.last_state = self.unwrapped.clone_full_state()
        self.last_action = action
        return self.env.step(action)
        
    def reset(self):
        obs = self.env.reset()
        obs, r, done, info = self.step(0) #noop
        assert not done, "End of episode at reset()."
        return obs
        
    def get_internal_state(self):
        return (self.last_state, self.last_action)
    
    def set_internal_state(self, internal_state):
        state, action = internal_state
        self.unwrapped.restore_full_state(state)
        self.step(action)
        
    def get_observation(self):
        return self.unwrapped._get_obs()


# Gym wrapper with clone/restore state
class Wrapper(gym.Wrapper):
    def clone_state(self):
        return self.env.clone_state()

    def restore_state(self, state):
        self.env.restore_state(state)


class MaxFilter(Wrapper):
    """
    Observation is elementwise maximum between last two observations
    """
    def __init__(self, env):
        super(MaxFilter, self).__init__(env)
        self.observations = deque(maxlen=2)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        return self._observation(obs), r, done, info

    def reset(self):
        observation = self.env.reset()
        self.observations.append(observation)
        self.observations.append(observation)
        return observation
        
    def _observation(self, observation):
        self.observations.append(observation)
        return np.max(self.observations, axis=0)


class MaxFilterFrameskip(Wrapper):
    def __init__(self, env, frameskip=4):
        super(MaxFilterFrameskip, self).__init__(env)
        self.observations = deque(maxlen=2)
        self.frameskip = frameskip
        
    def step(self, action):
        reward = 0.0
        for _ in range(self.frameskip):
            obs, r, done, info = self.env.step(action)
            self.observations.append(obs)
            reward += r
            if done:
                break
        return np.max(self.observations, axis=0), reward, done, info
    
    def reset(self):
        observation = self.env.reset()
        self.observations.append(observation)
        self.observations.append(observation)
        return observation


class ResizeImage(Wrapper):
    """
    Resizes frame to new_size
    """
    def __init__(self, env, new_size=(84,84)):
        super(ResizeImage, self).__init__(env)
        self.resize_fn = lambda obs: cv2.resize(obs, dsize=new_size, interpolation = cv2.INTER_LINEAR)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_size)

    def reset(self):
        observation = self.env.reset()
        return self.resize_fn(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.resize_fn(observation), reward, done, info


class Grayscale(Wrapper):
    """
    Ignores observation from previous wrapper/environment and returns ale.getScreenGrayscale().
    """
    def __init__(self, env):
        super(Grayscale, self).__init__(env)
        self.screen_width, self.screen_height = self.unwrapped.ale.getScreenDims()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.observation_space.shape[:-1])

    def reset(self):
        _ = self.env.reset()
        return self._observation()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._observation(), reward, done, info

    def _observation(self):
        return self.unwrapped.ale.getScreenGrayscale().reshape((self.screen_height, self.screen_width)) #faster than converting the input RGB observation


class OneLifeEpisode(Wrapper):
    """
    Emits terminal signal when loosing a life, causing the agent to reset the
    environment. This reset is 'fake' though, so that the whole state space can
    be explored. From https://github.com/openai/baselines/blob/699919f1cf2527b184f4445a3758a773f333a1ba/baselines/common/atari_wrappers.py#L50
    """
    def __init__(self, env):
        super(OneLifeEpisode, self).__init__(env)
        self.lives = self.env.unwrapped.ale.lives()
        self.done = True
    
    def step(self, action):
        observation, reward, self.done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        done = True if lives < self.lives else self.done
        self.lives = lives
        return observation, reward, done, info
    
    def reset(self):
        if not self.done:
            observation, _, done, _ = self.env.step(0) #perform 1 noop step
            if not done:
                return observation
        observation = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return observation


class StackFrames(gym.Wrapper):
    def __init__(self, env, buffer_size=4, stack_axis=-1):
        assert(buffer_size > 0)
        assert stack_axis in (0, -1)
        super(StackFrames, self).__init__(env)
        self.buffer_size = buffer_size
        self.observations = deque(maxlen=buffer_size)
        if stack_axis == 0:
            shape = [self.buffer_size] + list(self.observation_space.shape)
        else:
            shape = list(self.observation_space.shape) + [self.buffer_size]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape)
        
    def step(self, a):
        observation, reward, done, info = self.env.step(a)
        self.observations.append(observation)
        return self._observation(), reward, done, info
    
    def reset(self):
        initial_frame = self.env.reset()
        for _ in range(self.buffer_size):
            self.observations.append(initial_frame)
        return self._observation()

    def _observation(self):
        return list(self.observations)

    def clone_state(self):
        return (tuple(self.observations), self.env.clone_state())

    def restore_state(self, state):
        assert len(state[0]) == len(self.observations)
        self.observations.extend(state[0])
        return self.env.restore_state(state[1])


class ResetAndFire(Wrapper):
    def __init__(self, env):
        super(ResetAndFire, self).__init__(env)
        if env.unwrapped.get_action_meanings()[1] == 'FIRE': #0 is always noop, 1 is fire if the action is allowed
            def reset_and_fire():
                self.env.reset()
                obs, _, done, _ = self.env.step(1) #whatch out: if using stochastic ale we might actually not fire
                assert not done, "Episode ended after reset + fire"
                return obs
            self._reset = reset_and_fire
