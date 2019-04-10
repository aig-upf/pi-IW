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
def get_observe_funcion(algorithm, model):
    def observe_alphazero(env, node):
        x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
        logits, value = model(x, output_features=False)
        node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
        node.data["v"] = value.numpy().squeeze()

    def observe_pi_iw_BASIC(env, node):
        x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
        logits = model(x)
        node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
        gridenvs_BASIC_features(env, node)  # compute BASIC features for gridenvs

    def observe_pi_iw_dynamic(env, node):
        x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
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
    return returns


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
def run_episode(plan_step_fn, learner, dataset, cache_subtree, add_returns):
    episode_done = False
    actor.reset()
    episode_rewards = []
    aux_replay = ExperienceReplay()  # New auxiliary buffer to save current episode transitions
    while not episode_done:
        tree_policy = plan_step_fn(len(episode_rewards))
        a = sample_pmf(tree_policy)
        prev_root_data, current_root_data = actor.step(a, cache_subtree=cache_subtree)
        aux_replay.append({"observations": prev_root_data["obs"],
                        "target_policy": tree_policy})
        episode_rewards.append(current_root_data["r"])
        episode_done = current_root_data["done"]

        if learner is not None:
            batch = dataset.sample(batch_size)
            obs = tf.constant(batch["observations"], dtype=tf.float32)
            target_policy = tf.constant(batch["target_policy"], dtype=tf.float32)
            if add_returns:
                returns = tf.constant(batch["returns"], dtype=tf.float32)
                loss, _ = learner.train_step(obs, target_policy, returns)
            else:
                loss, _ = learner.train_step(obs, target_policy)

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
    import gym
    import argparse
    from mcts import MCTSAlphaZero
    from rollout_iw import RolloutIW
    from tree import TreeActor
    from supervised_policy import Mnih2013, SupervisedPolicy, SupervisedPolicyValue
    from experience_replay import ExperienceReplay
    import gridenvs.examples  # load simple envs


    # Compatibility with tensorflow 2.0
    tf.enable_eager_execution()
    tf.enable_resource_variables()


    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", type=str, default="pi-IW-dynamic",
                        choices=["AlphaZero", "pi-IW-BASIC", "pi-IW-dynamic"])
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-e", "--env", type=str, default="GE_MazeKeyDoor-v2")
    args, _ = parser.parse_known_args()

    # HYPERPARAMETERS
    tree_budget = 20
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


    # Set random seed
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Define environment, model and optimizer and tree actor
    env = gym.make(args.env)
    model = Mnih2013(num_logits=env.action_space.n, add_value=(args.algorithm=="AlphaZero"))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=rmsprop_decay,
                                          epsilon=rmsprop_epsilon)
    actor = TreeActor(env, observe_fn=get_observe_funcion(args.algorithm, model))

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
        network_policy = lambda node, bf: node.data["probs"]
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
                                      add_returns=(args.algorithm=="AlphaZero"))
        train_stats.report(episode_rewards, actor.nodes_generated)

    # Interleave planning and learning steps
    print("\nInterleaving planning and learning steps.", flush=True)
    while actor.nodes_generated < max_simulator_steps:
        episode_rewards = run_episode(plan_step_fn=plan_step_fn,
                                      learner=learner,
                                      dataset=experience_replay,
                                      cache_subtree=cache_subtree,
                                      add_returns=(args.algorithm=="AlphaZero"))
        train_stats.report(episode_rewards, actor.nodes_generated)
