"""
Example of pi-IW: guided Rollout-IW, interleaving planning and learning.
"""

import numpy as np
import tensorflow as tf
from planning_step import gridenvs_BASIC_features
from online_planning import softmax_Q_tree_policy


# Function that will be executed at each interaction with the environment
def observe_pi_iw_dynamic(env, node):
    x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
    logits, features = model(x, output_features=True)
    node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
    node.data["features"] = features.numpy().ravel()


def observe_pi_iw_BASIC(env, node):
    x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
    logits = model(x)
    node.data["probs"] = tf.nn.softmax(logits).numpy().ravel()
    gridenvs_BASIC_features(env, node)  # compute BASIC features


def planning_step(actor, planner, dataset, policy_fn, tree_budget, cache_subtree, discount_factor):
    nodes_before_planning = len(actor.tree)
    budget_fn = lambda: len(actor.tree) - nodes_before_planning == tree_budget
    planner.plan(tree=actor.tree,
                 successor_fn=actor.generate_successor,
                 stop_condition_fn=budget_fn,
                 policy_fn=policy_fn)
    tree_policy = softmax_Q_tree_policy(actor.tree, actor.tree.branching_factor, discount_factor, temp=0)
    a = sample_pmf(tree_policy)
    prev_root_data, current_root_data = actor.step(a, cache_subtree=cache_subtree)
    dataset.append({"observations": prev_root_data["obs"],
                    "target_policy": tree_policy})
    return current_root_data["r"], current_root_data["done"]


if __name__ == "__main__":
    import gym
    from rollout_iw import RolloutIW
    from tree import TreeActor
    from supervised_policy import SupervisedPolicy, Mnih2013
    from utils import sample_pmf
    from experience_replay import ExperienceReplay
    import gridenvs.examples  # load simple envs


    # Compatibility with tensorflow 2.0
    tf.enable_eager_execution()
    tf.enable_resource_variables()


    # HYPERPARAMETERS
    seed = 0
    env_id = "GE_PathKeyDoor-v0"
    use_dynamic_feats = False # otherwise BASIC features will be used
    n_episodes = 5
    tree_budget = 20
    discount_factor = 0.99
    cache_subtree = True
    batch_size = 32
    learning_rate = 0.0007
    replay_capacity = 1000
    regularization_factor = 0.001
    clip_grad_norm = 40
    rmsprop_decay = 0.99
    rmsprop_epsilon = 0.1


    # Set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
    env = gym.make(env_id)
    observe_fn = observe_pi_iw_dynamic if use_dynamic_feats else observe_pi_iw_BASIC
    actor = TreeActor(env, observe_fn=observe_fn)
    planner = RolloutIW(branching_factor=env.action_space.n, ignore_cached_nodes=True)

    model = Mnih2013(num_logits=env.action_space.n, add_value=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=rmsprop_decay,
                                          epsilon=rmsprop_epsilon)
    learner = SupervisedPolicy(model, optimizer, regularization_factor=regularization_factor, use_graph=True)
    experience_replay = ExperienceReplay(capacity=replay_capacity)

    def network_policy(node, branching_factor):
        return node.data["probs"]

    # Initialize experience replay: run some steps until we have enough examples to form one batch
    print("Initializing experience replay", flush=True)
    actor.reset()
    while len(experience_replay) < batch_size:
        r, episode_done = planning_step(actor=actor,
                                        planner=planner,
                                        dataset=experience_replay,
                                        policy_fn=network_policy,
                                        tree_budget=tree_budget,
                                        cache_subtree=cache_subtree,
                                        discount_factor=discount_factor)
        if episode_done: actor.reset()

    # Interleave planning and learning steps
    print("\nInterleaving planning and learning steps.", flush=True)
    actor.reset()
    steps_cnt = 0
    episode_steps = 0
    episodes_cnt = 0
    while episodes_cnt < n_episodes:
        r, episode_done = planning_step(actor=actor,
                                        planner=planner,
                                        dataset=experience_replay,
                                        policy_fn=network_policy,
                                        tree_budget=tree_budget,
                                        cache_subtree=cache_subtree,
                                        discount_factor=discount_factor)

        # Learning step
        batch = experience_replay.sample(batch_size)
        loss, _ = learner.train_step(tf.constant(batch["observations"], dtype=tf.float32),
                                     tf.constant(batch["target_policy"], dtype=tf.float32))

        steps_cnt += 1
        episode_steps +=1
        print(actor.tree.root.data["s"][0], "Reward: ", r, "Simulator steps:", actor.nodes_generated,
              "Planning steps:", steps_cnt, "Loss:", loss.numpy(), "\n")
        if episode_done:
            print("Problem solved in %i steps (min 13 steps)."%episode_steps)
            actor.reset()
            episodes_cnt += 1
            episode_steps = 0
            if episodes_cnt < n_episodes: print("\n------- New episode -------")