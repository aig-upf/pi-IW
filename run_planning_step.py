
if __name__ == "__main__":
    import gridenvs.examples # load GE environments
    from src.utils import create_env
    from src.registry import get_planner
    from src.tree_actor import TreeActor
    from src.settings import settings


    env = create_env(settings["env_id"], settings["frameskip"])
    actor = TreeActor(env)

    planner, feature_extractor = get_planner(actor=actor,
                                             obs_shape=env.observation_space.shape,
                                             n_actions=env.action_space.n,
                                             planner_class=settings["planner"],
                                             features_class=settings["features"],
                                             NN_planner=settings["trainedNN_planner"],
                                             NN_features=settings["trainedNN_features"])

    actor.reset_env()
    planner.plan()

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

