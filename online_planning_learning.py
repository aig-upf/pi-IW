
if __name__ == "__main__":
    import gym
    from rollout_iw import RolloutIW
    from tree import TreeActor
    from supervised_policy import SupervisedPolicy, Mnih2013
    from utils import sample_pmf
    from online_planning import softmax_Q_tree_policy
    from experience_replay import ExperienceReplay
    import tensorflow as tf
    import numpy as np

    tf.enable_eager_execution()
    tf.enable_resource_variables()

    import gridenvs.examples #load simple envs
    from plan_step import gridenvs_BASIC_features

    env_id = "GE_PathKeyDoor-v0"
    max_tree_nodes = 30
    discount_factor = 0.99
    cache_subtree = True

    seed = 0
    batch_size = 32
    learning_rate = 0.0007
    replay_capacity = 1000
    replay_min_transitions = 100
    max_simulator_steps = 10000

    # Set random seed
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
    env = gym.make(env_id)
    actor = TreeActor(env, observe_fn=gridenvs_BASIC_features)
    planner = RolloutIW(branching_factor=env.action_space.n)

    model = Mnih2013(num_logits=env.action_space.n)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    learner = SupervisedPolicy(model, optimizer)

    experience_replay = ExperienceReplay(capacity=replay_capacity)

    def planning_step(actor, planner, dataset):
        planner.plan(tree=actor.tree,
                     successor_fn=actor.generate_successor,
                     stop_condition_fn=lambda: len(actor.tree) == max_tree_nodes)

        tree_policy = softmax_Q_tree_policy(actor.tree, env.action_space.n, discount_factor, temp=0)
        a = sample_pmf(tree_policy)
        prev_root_data, current_root_data = actor.step(a, cache_subtree=cache_subtree)
        dataset.append({"observations": prev_root_data["obs"],
                        "target_policy": tree_policy})

        return current_root_data["done"]

    # Initialize experience replay: run complete episodes until we exceed both batch_size and dataset_min_transitions
    while len(experience_replay) < batch_size or len(experience_replay) < replay_min_transitions:
        episode_done = False
        actor.reset()
        while not episode_done:
            episode_done = planning_step(actor, planner, dataset=experience_replay)

    # Once initialized, interleave planning and learning steps
    tree = actor.reset()
    episode_transitions = ExperienceReplay()
    cnt = 0
    while actor.nodes_generated < max_simulator_steps:
        episode_done = planning_step(actor, planner, dataset=episode_transitions)

        # Add transitions to dataset once the episode is over
        if episode_done:
            experience_replay.extend(episode_transitions)
            episode_transitions = ExperienceReplay()
            actor.reset()

        # Learning step
        batch = experience_replay.sample(batch_size)
        loss, _ = learner.train_step(tf.constant(batch["observations"], dtype=tf.float32),
                                     tf.constant(batch["target_policy"], dtype=tf.float32))
        cnt += 1
        print("Simulator steps:", actor.nodes_generated, "\tPlanning steps:", cnt, "\tLoss:", loss.numpy())
