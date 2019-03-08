# Define how we will extract features
def gridenvs_BASIC_features(env, node):
    features_array = env.unwrapped.world.get_colors().flatten()
    node.data["features"] = list(enumerate(features_array))

if __name__ == "__main__":
    import gym
    from rollout_iw import RolloutIW
    from tree import TreeActor
    import gridenvs.examples # register GE environments to gym

    env_id = "GE_PathKeyDoor-v0"
    max_tree_nodes = 30

    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
    env = gym.make(env_id)
    actor = TreeActor(env, observe_fn=gridenvs_BASIC_features)

    planner = RolloutIW(branching_factor=env.action_space.n)

    tree = actor.reset()
    planner.plan(tree=tree,
                 successor_fn=actor.generate_successor,
                 stop_condition_fn=lambda: len(tree) == max_tree_nodes)

    # Print tree function
    n = 0
    def str_node_data(data):
        global n
        s = str(n) + "-> "
        n+=1
        if "r" in data.keys():
            s += "r: " + str(data["r"])
        else:
            s += "ROOT"
        return s

    print(actor.tree.str_tree(str_node_data))

