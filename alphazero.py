"""
Example of pi-IW: interleaving planning and learning.
"""

import numpy as np
import tensorflow as tf
from utils import softmax


# Function that will be executed at each interaction with the environment
def observe(env, node):
    x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
    logits, value = model(x, output_features=False)
    node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
    node.data["v"] = value.numpy().squeeze()


def policy_counts(node, temp):
    assert "N" in node.data.keys(), "Counts not present in tree. Use a planner that computes counts."
    if temp > 0:
        aux = node.data["N"] ** (1 / temp)
        return aux / np.sum(aux)
    else:
        assert temp == 0
        return softmax(node.data["N"], temp=0)


class Counter:
    def __init__(self):
        self.cnt = 0
    def inc(self):
        self.cnt += 1
        return self.cnt

def planning_step(actor, planner, dataset, tree_budget, cache_subtree, episode_transitions, first_moves_temp, temp):
    cnt = Counter()
    budget_fn = lambda: cnt.inc() == tree_budget
    planner.plan(tree=actor.tree,
                 successor_fn=actor.generate_successor,
                 stop_condition_fn=budget_fn)
    if episode_transitions <= first_moves_temp:
        tree_policy = policy_counts(actor.tree.root, temp)
    else:
        tree_policy = policy_counts(actor.tree.root, temp=0)
    a = sample_pmf(tree_policy)
    prev_root_data, current_root_data = actor.step(a, cache_subtree=cache_subtree)
    dataset.append({"observations": prev_root_data["obs"],
                    "target_policy": tree_policy})
    return current_root_data["r"], current_root_data["done"]


def compute_returns(rewards, discount_factor):
    R = 0
    returns = []
    for i in range(len(rewards) - 1, -1, -1):
        R = rewards[i] + discount_factor * R
        returns.append(R)
    return returns


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
    from mcts import MCTSAlphaZero
    from tree import TreeActor
    from supervised_policy import SupervisedPolicyValue, Mnih2013
    from utils import sample_pmf
    from experience_replay import ExperienceReplay
    import gridenvs.examples  # load simple envs


    # Compatibility with tensorflow 2.0
    tf.enable_eager_execution()
    tf.enable_resource_variables()


    # HYPERPARAMETERS
    seed = 0
    env_id = "GE_MazeKeyDoor-v0"
    tree_budget = 50
    discount_factor = 0.99
    puct_factor = 0.5
    first_moves_temp = np.inf
    policy_temp = 1
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
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
    env = gym.make(env_id)
    actor = TreeActor(env, observe_fn=observe)
    planner = MCTSAlphaZero(branching_factor=env.action_space.n,
                            discount_factor=discount_factor,
                            puct_factor=puct_factor)

    model = Mnih2013(num_logits=env.action_space.n, add_value=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=rmsprop_decay,
                                          epsilon=rmsprop_epsilon)
    learner = SupervisedPolicyValue(model, optimizer, regularization_factor=regularization_factor, use_graph=True)
    experience_replay = ExperienceReplay(capacity=replay_capacity)

    # Initialize experience replay: run complete episodes until we exceed both batch_size and dataset_min_transitions
    print("Initializing experience replay", flush=True)
    train_stats = TrainStats()
    while len(experience_replay) < batch_size or len(experience_replay) < replay_min_transitions:
        episode_done = False
        actor.reset()
        aux_replay = ExperienceReplay()  # Auxiliary buffer to save current episode transitions
        episode_rewards = []
        while not episode_done:
            r, episode_done = planning_step(actor,
                                            planner,
                                            dataset=aux_replay, # Add transitions to a separate episode buffer
                                            tree_budget=tree_budget,
                                            cache_subtree=cache_subtree,
                                            episode_transitions=len(episode_rewards),
                                            first_moves_temp=first_moves_temp,
                                            temp=policy_temp)
            episode_rewards.append(r)

        train_stats.report(episode_rewards, actor.nodes_generated)
        returns = compute_returns(episode_rewards, discount_factor)  # Backpropagate rewards
        aux_replay.add_column("returns", returns)  # Add them to the dataset
        experience_replay.extend(aux_replay)  # Add transitions to the buffer that will be used for learning

    # Once initialized, interleave planning and learning steps
    print("\nInterleaving planning and learning steps.", flush=True)
    tree = actor.reset()
    aux_replay = ExperienceReplay()  # Auxiliary buffer to save current episode transitions
    episode_rewards = []
    while actor.nodes_generated < max_simulator_steps:
        r, episode_done = planning_step(actor,
                                        planner,
                                        dataset=aux_replay,  # Add transitions to a separate episode buffer
                                        tree_budget=tree_budget,
                                        cache_subtree=cache_subtree,
                                        episode_transitions=len(episode_rewards),
                                        first_moves_temp=first_moves_temp,
                                        temp=policy_temp)
        episode_rewards.append(r)

        # Add transitions to the experience replay buffer once the episode is over
        if episode_done:
            train_stats.report(episode_rewards, actor.nodes_generated)
            returns = compute_returns(episode_rewards, discount_factor)  # Backpropagate rewards
            aux_replay.add_column("returns", returns)  # Add them to the dataset
            experience_replay.extend(aux_replay)  # Add transitions to the buffer that will be used for learning
            aux_replay = ExperienceReplay()
            episode_rewards = []
            actor.reset()

        # Learning step
        batch = experience_replay.sample(batch_size)
        loss, _ = learner.train_step(tf.constant(batch["observations"], dtype=tf.float32),
                                     tf.constant(batch["target_policy"], dtype=tf.float32),
                                     tf.constant(batch["returns"], dtype=tf.float32))
        # print(actor.tree.root.data["s"][0], "Simulator steps:", actor.nodes_generated, "\tPlanning steps:", cnt, "\tLoss:", loss.numpy())
