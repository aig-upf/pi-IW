"""
Comparison between algorithms that interleave planning and learning steps:
pi-IW-BASIC, pi-IW-dynamic and AlphaZero
"""

import numpy as np
import tensorflow as tf
from utils import softmax, sample_pmf
from online_planning import softmax_Q_tree_policy
from planning_step import gridenvs_BASIC_features


# "observe" function will be executed at each interaction with the environment
def get_observe_funcion(algorithm, model, preproc_obs_fn=None):
    if preproc_obs_fn is None:
        preproc_obs_fn = lambda x: np.asarray(x)  #

    def observe_alphazero(env, node):
        x = tf.constant(preproc_obs_fn([node.data["obs"]]).astype(np.float32)) #TODO: maybe change dtype to tf.constant and take out np.asarray from preproc_obs_fn
        logits, value = model(x, output_features=False)
        node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
        node.data["v"] = value.numpy().squeeze()

    def observe_pi_iw_BASIC(env, node):
        x = tf.constant(preproc_obs_fn([node.data["obs"]]).astype(np.float32))
        logits = model(x)
        node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
        gridenvs_BASIC_features(env, node)  # compute BASIC features for gridenvs

    def observe_pi_iw_dynamic(env, node):
        x = tf.constant(preproc_obs_fn([node.data["obs"]]).astype(np.float32))
        logits, features = model(x, output_features=True)
        node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
        node.data["features"] = features.numpy().ravel()

    if algorithm == "AlphaZero": return observe_alphazero
    elif algorithm == "pi-IW-BASIC": return observe_pi_iw_BASIC
    elif algorithm == "pi-IW-dynamic": return observe_pi_iw_dynamic
    raise ValueError('Wrong algorithm? Options are: "AlphaZero", "pi-IW-BASIC" and "pi-IW-dynamic"')


def policy_counts(node, temp):
    assert "N" in node.data.keys(), "Counts not present in tree. Use a planner that computes counts."
    if temp > 0:
        aux = node.data["N"] ** (1 / temp)
        return aux / np.sum(aux)
    else:
        assert temp == 0
        return softmax(node.data["N"], temp=0)


def compute_returns(rewards, discount_factor):
    R = 0
    returns = []
    for i in range(len(rewards) - 1, -1, -1):
        R = rewards[i] + discount_factor * R
        returns.append(R)
    return list(reversed(returns))


class Counter:
    def __init__(self):
        self.cnt = 0
    def inc(self):
        self.cnt += 1
        return self.cnt


# AlphaZero planning step function with given hyperparameters
def get_alphazero_planning_step_fn(actor, planner, tree_budget, first_moves_temp, temp):
    def alphazero_planning_step(episode_transitions):
        cnt = Counter()
        budget_fn = lambda: cnt.inc() == tree_budget
        planner.plan(tree=actor.tree,
                     successor_fn=actor.generate_successor,
                     stop_condition_fn=budget_fn)
        if episode_transitions <= first_moves_temp:
            tree_policy = policy_counts(actor.tree.root, temp)
        else:
            tree_policy = policy_counts(actor.tree.root, temp=0)
        return tree_policy
    return alphazero_planning_step


# pi-IW planning step function with given hyperparameters
def get_pi_iw_planning_step_fn(actor, planner, policy_fn, tree_budget, discount_factor):
    def pi_iw_planning_step(episode_tranistions):
        nodes_before_planning = len(actor.tree)
        budget_fn = lambda: len(actor.tree) - nodes_before_planning == tree_budget
        planner.plan(tree=actor.tree,
                     successor_fn=actor.generate_successor,
                     stop_condition_fn=budget_fn,
                     policy_fn=policy_fn)
        return softmax_Q_tree_policy(actor.tree, actor.tree.branching_factor, discount_factor, temp=0)
    return pi_iw_planning_step


# Given either pi-IW or AlphaZero planning step functions, run an entire episode performing planning and learning steps
def run_episode(plan_step_fn, learner, dataset, cache_subtree, add_returns, preproc_obs_fn=None, render=False):
    episode_done = False
    actor.reset()
    episode_rewards = []
    aux_replay = ExperienceReplay()  # New auxiliary buffer to save current episode transitions
    while not episode_done:
        # Planning step
        tree_policy = plan_step_fn(len(episode_rewards))

        # Execute action (choose one node as the new root from depth 1)
        a = sample_pmf(tree_policy)
        prev_root_data, current_root_data = actor.step(a, cache_subtree, render, render_size=(512,512))
        aux_replay.append({"observations": prev_root_data["obs"],
                           "target_policy": tree_policy})
        episode_rewards.append(current_root_data["r"])
        episode_done = current_root_data["done"]

        # Learning step
        if learner is not None:
            batch = dataset.sample(batch_size)
            if preproc_obs_fn is not None:
                batch["observations"] = preproc_obs_fn(batch["observations"])
            obs = tf.constant(batch["observations"], dtype=tf.float32)
            target_policy = tf.constant(batch["target_policy"], dtype=tf.float32)
            if add_returns:
                returns = tf.constant(batch["returns"], dtype=tf.float32)
                loss, _ = learner.train_step(obs, target_policy, returns)
            else:
                loss, _ = learner.train_step(obs, target_policy)

    # Add episode to the dataset
    if add_returns:
        returns = compute_returns(episode_rewards, discount_factor)  # Backpropagate rewards
        aux_replay.add_column("returns", returns)  # Add them to the dataset
    dataset.extend(aux_replay)  # Add transitions to the buffer that will be used for learning

    return episode_rewards


class TrainStats:
    def __init__(self):
        self.last_interactions = 0
        self.steps = 0
        self.episodes = 0

    def report(self, episode_rewards, total_interactions):
        self.episodes += 1
        self.steps += len(episode_rewards)
        print("Episode: %i."%self.episodes,
              "Reward: %.2f"%np.sum(episode_rewards),
              "Episode interactions %i."%(total_interactions-self.last_interactions),
              "Episode steps %i."%len(episode_rewards),
              "Total interactions %i."%total_interactions,
              "Total steps %i."%self.steps)
        self.last_interactions = total_interactions


if __name__ == "__main__":
    import gym, gym.wrappers
    import argparse
    import logging
    from mcts import MCTSAlphaZero
    from rollout_iw import RolloutIW
    from tree import TreeActor
    from supervised_policy import Mnih2013, SupervisedPolicy, SupervisedPolicyValue
    from experience_replay import ExperienceReplay
    from utils import remove_env_wrapper, env_has_wrapper
    from atari_wrappers import is_atari_env, wrap_atari_env
    from distutils.util import strtobool
    import gridenvs.examples  # load simple envs


    # HYPERPARAMETERS
    tree_budget = 50
    discount_factor = 0.99
    puct_factor = 0.5 # AlphaZero
    first_moves_temp = np.inf # AlphaZero
    policy_temp = 1 # AlphaZero
    cache_subtree = True
    batch_size = 32
    learning_rate = 0.0007
    replay_capacity = 1000
    replay_min_transitions = 100
    max_simulator_steps = 1000000
    regularization_factor = 0.001
    clip_grad_norm = 40
    rmsprop_decay = 0.99
    rmsprop_epsilon = 0.1
    frameskip_atari = 15


    logger = logging.getLogger(__name__)

    # Compatibility with tensorflow 2.0
    tf.enable_eager_execution()
    tf.enable_resource_variables()

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", type=str, default="pi-IW-dynamic",
                        choices=["AlphaZero", "pi-IW-BASIC", "pi-IW-dynamic"])
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-e", "--env", type=str, default="GE_MazeKeyDoor-v2")
    parser.add_argument("--render", type=strtobool, default="False")
    args, _ = parser.parse_known_args()


    # Set random seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Create the gym environment. When creating it with gym.make(), gym usually puts a TimeLimit wrapper around an env.
    # We'll take this out since we will most likely reach the step limit (when restoring the internal state of the
    # emulator the step count of the wrapper will not reset)
    env = gym.make(args.env)
    if env_has_wrapper(env, gym.wrappers.TimeLimit):
        env = remove_env_wrapper(env, gym.wrappers.TimeLimit)
        logger.warning("TimeLimit environment wrapper removed.")

    # If the environment is an Atari game, the observations will be the last four frames stacked in a 4-channel image
    if is_atari_env(env):
        env = wrap_atari_env(env, frameskip_atari)
        logger.warning("Atari environment modified: observation is now a 4-channel image of the last four non-skipped frames in grayscale. Frameskip set to %i." % frameskip_atari)
        preproc_obs_fn = lambda obs_batch: np.moveaxis(obs_batch, 1, -1)  # move channels to the last dimension
    else:
        preproc_obs_fn = None

    # Define model and optimizer
    model = Mnih2013(num_logits=env.action_space.n, add_value=(args.algorithm=="AlphaZero"))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=rmsprop_decay,
                                          epsilon=rmsprop_epsilon)

    # TreeActor provides equivalent functions to env.step() and env.reset() for on-line planning: it creates a tree,
    # adds nodes to it and allows us to take steps (maybe keeping the subtree)
    actor = TreeActor(env, observe_fn=get_observe_funcion(args.algorithm, model, preproc_obs_fn))

    # Define, depending on the algorithm, the planner and the step functions for planning and learning
    if args.algorithm == "AlphaZero":
        planner = MCTSAlphaZero(branching_factor=env.action_space.n,
                                discount_factor=discount_factor,
                                puct_factor=puct_factor)
        plan_step_fn = get_alphazero_planning_step_fn(actor=actor,
                                                      planner=planner,
                                                      tree_budget=tree_budget,
                                                      first_moves_temp=first_moves_temp,
                                                      temp=policy_temp)
        learner = SupervisedPolicyValue(model, optimizer, regularization_factor=regularization_factor, use_graph=True)
    else:
        assert args.algorithm in ("pi-IW-BASIC", "pi-IW-dynamic")
        planner = RolloutIW(branching_factor=env.action_space.n, ignore_cached_nodes=True)
        network_policy = lambda node, bf: node.data["probs"]  # Policy to guide the planner: NN output probabilities
        plan_step_fn = get_pi_iw_planning_step_fn(actor=actor,
                                                  planner=planner,
                                                  policy_fn=network_policy,
                                                  tree_budget=tree_budget,
                                                  discount_factor=discount_factor)
        learner = SupervisedPolicy(model, optimizer, regularization_factor=regularization_factor, use_graph=True)

    # Initialize experience replay: run complete episodes until we exceed both batch_size and dataset_min_transitions
    print("Initializing experience replay", flush=True)
    train_stats = TrainStats()
    experience_replay = ExperienceReplay(capacity=replay_capacity)
    while len(experience_replay) < batch_size or len(experience_replay) < replay_min_transitions:
        episode_rewards = run_episode(plan_step_fn=plan_step_fn,
                                      learner=None,
                                      dataset=experience_replay,
                                      cache_subtree=cache_subtree,
                                      add_returns=(args.algorithm=="AlphaZero"),
                                      preproc_obs_fn=preproc_obs_fn,
                                      render=args.render)
        train_stats.report(episode_rewards, actor.nodes_generated)

    # Interleave planning and learning steps
    print("\nInterleaving planning and learning steps.", flush=True)
    while actor.nodes_generated < max_simulator_steps:
        episode_rewards = run_episode(plan_step_fn=plan_step_fn,
                                      learner=learner,
                                      dataset=experience_replay,
                                      cache_subtree=cache_subtree,
                                      add_returns=(args.algorithm=="AlphaZero"),
                                      preproc_obs_fn=preproc_obs_fn,
                                      render=args.render)
        train_stats.report(episode_rewards, actor.nodes_generated)
