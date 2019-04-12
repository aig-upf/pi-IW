import time, timeit
import cv2
from .utils import node_fill_network_data
from .session_manager import session_manager

class Agent:
    def __init__(self):
        self._done = True
        self.viewer = None

    def _action(self):
        raise NotImplementedError()

    def step(self):
        if self._done:
            self._obs = self._reset()
        a = self._action()
        self._obs, r, self._done, info = self._env.step(a)
        return a, r, self._done

    def _reset(self):
        return self._env.reset()

    def get_step_fn(self, env, verbose=False, render=False, render_fps=6, render_size=None):
        self._env = env
        if render or verbose:
            spf = 1/render_fps
            def step_render():
                t = timeit.default_timer()
                a, r, done = self.step()
                if verbose:
                    print("a = %d, r = %d, done = %d" % (a, r, done), flush=True)
                if render:
                    self.render(size=render_size)
                    sleep_time = spf - (timeit.default_timer() - t)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                return a, r, done
            return step_render
        return self.step #step function with no overhead

    def _get_image(self, size=None):
        img = self._obs[-1] if type(self._obs) is list else self._obs
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def render(self, size=None, close=False):
        # render the game as the agent sees it
        # Source: https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self._get_image(size)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            time.sleep(2) #first time we wait a little, otherwise we'll miss the first frames
        self.viewer.imshow(img)

    def __del__(self):
        self.render(close=True) #close render window

class RandomAgent(Agent):
    def _action(self):
        return self._env.action_space.sample()

class PolicyAgent(Agent):
    def __init__(self, policy):
        self.policy = policy
        Agent.__init__(self)

    def _reset(self):
        return Agent._reset(self)

    def _action(self):
        current_data = {"obs": self._obs}
        node_fill_network_data(session_manager.session, self.policy.network, ["policy_head"], current_data)
        return self.policy.get_action(current_data)

class LookaheadAgent(Agent):
    def __init__(self, lookahead):
        self.lookahead = lookahead
        Agent.__init__(self)

    def step(self):
        trajectory, _ = self.lookahead.step()
        d = trajectory[-1]
        self._obs = d["obs"]
        return d["a"], d["r"], d["done"]

    def _reset(self):
        raise Exception("Lookahead takes care of that.")

    def get_step_fn(self, env, verbose=False, render=False, render_fps=4, render_size=None):
        assert self.lookahead.planner.actor.env is env
        return Agent.get_step_fn(self, env, verbose=verbose, render=render, render_fps=render_fps, render_size=render_size)


def run_episode(step_fn):
    # Initialization
    episode_reward = 0
    episode_steps = 0
    done = False

    # Run episode
    while not done:
        a, r, done = step_fn()
        episode_reward += r
        episode_steps += 1

    return episode_reward, episode_steps


def run_episodes(step_fn, num_episodes=1):
    episode_rewards = list()
    episode_steps = list()
    for i in range(num_episodes):
        reward, steps = run_episode(step_fn)
        episode_rewards.append(reward)
        episode_steps.append(steps)
    return episode_rewards, episode_steps