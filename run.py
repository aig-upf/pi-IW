# Run a single agent locally
if __name__ == "__main__":
    import os
    import argparse
    from src.utils import configure_root_logger
    from src.settings import settings, parse_args
    from src.session_manager import session_manager
    from src.learner import LookaheadLearner
    import gridenvs.examples


    parser = argparse.ArgumentParser(add_help=False) # disabling help here won't show these next 3 arguments but all of them
    parser.add_argument('log_path')
    parser.add_argument('exp_id')
    args, unused_args = parser.parse_known_args() # parse num workers and port (and seed so that it does not appear in unused_args)

    parse_args()

    log_dir = os.path.join(args.log_path, args.exp_id)  # All logs and checkpoints will go to this directory
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, "log.txt")
    configure_root_logger(log_to_stdout=True, filename=filename, enable_logs=True, log_level="INFO")

    with session_manager.configure(log_dir = log_dir,
                                   task_id = 0,
                                   num_workers = 1,
                                   write_summaries = True,
                                   save_checkpoints = False):
        actor_learner = LookaheadLearner(log_dir=log_dir)
        actor_learner.run()
