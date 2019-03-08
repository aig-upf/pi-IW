
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
    import timeit

    tf.enable_eager_execution()
    tf.enable_resource_variables()

    import gridenvs.examples #load simple envs
    from plan_step import gridenvs_BASIC_features

    env_id = "GE_MazeKeyDoor-v0"
    tree_budget = 50
    discount_factor = 0.99
    cache_subtree = True

    seed = 0
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
    actor = TreeActor(env, observe_fn=gridenvs_BASIC_features)
    planner = RolloutIW(branching_factor=env.action_space.n, ignore_cached_nodes=True)

    model = Mnih2013(num_logits=env.action_space.n, add_value=False)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                          decay=rmsprop_decay,
                                          epsilon=rmsprop_epsilon)
    learner = SupervisedPolicy(model, optimizer, regularization_factor=regularization_factor, use_graph=True)

    experience_replay = ExperienceReplay(capacity=replay_capacity)

    def fill_nn(node):
        x = tf.constant(np.expand_dims(node.data["obs"], axis=0).astype(np.float32))
        logits, features = model(x, output_features=True)
        node.data["probs"] = tf.nn.softmax(logits).numpy().flatten()
        node.data["features"] = features.numpy().flatten()

    def network_policy(node, branching_factor):
        fill_nn(node)
        return node.data["probs"]

    def planning_step(actor, planner, dataset, policy):
        nodes_before_planning = len(actor.tree)
        budget_fn = lambda: len(actor.tree) - nodes_before_planning == tree_budget
        planner.plan(tree=actor.tree,
                     successor_fn=actor.generate_successor,
                     stop_condition_fn=budget_fn,
                     policy_fn = policy)

        tree_policy = softmax_Q_tree_policy(actor.tree, env.action_space.n, discount_factor, temp=0)
        a = sample_pmf(tree_policy)
        prev_root_data, current_root_data = actor.step(a, cache_subtree=cache_subtree)
        dataset.append({"observations": prev_root_data["obs"],
                        "target_policy": tree_policy})

        return current_root_data["done"]

    # Initialize experience replay: run complete episodes until we exceed both batch_size and dataset_min_transitions
    print("Initializing experience replay", flush=True)
    while len(experience_replay) < batch_size or len(experience_replay) < replay_min_transitions:
        episode_done = False
        actor.reset()
        while not episode_done:
            start = timeit.default_timer()
            episode_done = planning_step(actor, planner, dataset=experience_replay, policy=network_policy)
            print(actor.tree.root.data["s"], "\ttime:", timeit.default_timer()-start)
            print("", flush=True)

    # Once initialized, interleave planning and learning steps
    print("\nInterleaving planning and learning steps.", flush=True)
    tree = actor.reset()
    episode_transitions = ExperienceReplay()
    cnt = 0
    while actor.nodes_generated < max_simulator_steps:
        start = timeit.default_timer()
        episode_done = planning_step(actor, planner, dataset=episode_transitions, policy=network_policy)

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
        print(actor.tree.root.data["s"], "\ttime:", timeit.default_timer() - start)
        print("", flush=True)
