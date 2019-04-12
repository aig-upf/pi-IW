
if __name__ == "__main__":
    import gridenvs.examples #load simple envs
    import numpy as np
    import timeit
    from src.utils import configure_root_logger, create_env
    from src.registry import get_lookahead
    from src.agent import LookaheadAgent, run_episodes
    from src.tree_actor import TreeActor
    from src.settings import settings, parse_args


    parse_args()

    # LOGGING
    configure_root_logger(log_to_stdout=True, filename=None, enable_logs=True, log_level=settings["log_level"])

    env = create_env(settings["env_id"], settings["frameskip"])

    actor = TreeActor(env)

    lookahead, feature_extractor = get_lookahead(actor=actor,
                                                 obs_shape=env.observation_space.shape,
                                                 n_actions=env.action_space.n,
                                                 lookahead_class=settings["lookahead"],
                                                 planner_class=settings["planner"],
                                                 features_class=settings["features"],
                                                 NN_planner=settings["trainedNN_planner"],
                                                 NN_features=settings["trainedNN_features"])

    start = timeit.default_timer()

    agent = LookaheadAgent(lookahead)
    step_fn = agent.get_step_fn(env, verbose=True, render=True, render_size=(512, 512), render_fps=100)

    res = run_episodes(step_fn, 1)

    print("Avg reward: %.2f  Avg steps: %.2f. Avg time: %.2fs" % (
    np.mean(res[0]), np.mean(res[1]), (timeit.default_timer() - start) / len(res[0])), flush=True)
