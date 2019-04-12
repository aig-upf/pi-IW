from src.settings import settings, parse_args
from src.session_manager import session_manager
import gridenvs.examples # import GE envs

def launch_worker(task_id, num_workers, synchronous, server, log_dir, ps_shutdown=False):
    try:
        logger.info('Worker %i created.' % task_id)
        from src.learner import LookaheadLearner

        session_manager.initialize(log_dir=log_dir,
                                   task_id=task_id,
                                   num_workers=num_workers,
                                   target=server.target,
                                   worker_device='/job:worker/task:%d' % task_id,
                                   ps_device='/job:ps/task:0',
                                   synchronous=synchronous,
                                   write_summaries=True,
                                   save_checkpoints=False)
        actor_learner = LookaheadLearner(log_dir=log_dir)
        actor_learner.run()
    finally:
        if ps_shutdown:
            #send shutdown signal
            logger.info('Worker %i: Sending shutdown signal to parameter server and closing.' % task_id)
            g = tf.Graph() #MonitoredTrainingSession finalizes the default graph :( We add new ops in another graph
            with g.as_default():
                queue = tf.FIFOQueue(num_workers, tf.int32, shared_name="shutdown_queue")
                shutdown_signal = queue.enqueue(1)
            sess = tf.Session(server.target, graph=g)
            sess.run(shutdown_signal) #send something to tell the PS we are done here
            sleep(2) #wait a little, otherwise the parameter server hangs waiting for this worker
        logger.info('Closing worker %i.' % task_id)

def launch_ps(num_workers, server, ps_shutdown=False):
    """
    for automatic ps shutdown:
    instead of server.join() to wait for other processes to send tf ops, we'll
    block waiting for tokens in a queue, to know when to shut down the server.
    https://stackoverflow.com/questions/39810356/shut-down-server-in-tensorflow
    """
    logger.info('Parameter server created.')
    try:
        if ps_shutdown:
            queue = tf.FIFOQueue(num_workers, tf.int32, shared_name="shutdown_queue")
            sess = tf.Session(server.target)
            finished_workers = 0
            while finished_workers < num_workers:
                sess.run(queue.dequeue()) #block until receiving something from the queue
                finished_workers += 1
        else:
            server.join()
    finally:
        logger.info('Closing parameter server.')

if __name__ == "__main__":
    import logging
    import argparse
    import tensorflow as tf
    from time import sleep
    import os
    from src.utils import configure_root_logger_cluster
    from distutils.util import strtobool
    
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(add_help=False) #disabling help here won't show these next 3 arguments but all of them
    parser.add_argument('job_name')
    parser.add_argument('task_id', type=int)
    parser.add_argument('num_workers', type=int)
    parser.add_argument('start_port', type=int)
    parser.add_argument('synchronous', type=strtobool)
    parser.add_argument('log_path')
    parser.add_argument('exp_id')
    args, _ = parser.parse_known_args()

    parse_args()
    assert args.job_name in ('ps', 'worker'), "Job name has to be either 'ps', for parameter server, or 'worker'."
        
    #LOGGING
    log_dir = os.path.join(args.log_path, args.exp_id) #All logs and checkpoints will go to this directory
    os.makedirs(log_dir, exist_ok=True)
    enable_logs = (settings["logs"] == "all" or (args.job_name == "worker" and args.task_id == 0))
    filename = os.path.join(log_dir, "log.txt")
    configure_root_logger_cluster(args.job_name, args.task_id, log_to_stdout=True, filename=filename, enable_logs=enable_logs, log_level="INFO")
  
    #Create cluster spec: 1 ps and num_workers workers
    ps = ['localhost:{}'.format(args.start_port)]
    assert args.num_workers > 0
    workers = ['localhost:{}'.format(args.start_port+i+1) for i in range(args.num_workers)]
    cluster_spec = tf.train.ClusterSpec({'ps': ps, 'worker': workers})
    
    #Create server for local task
    server = tf.train.Server(cluster_spec, job_name=args.job_name, task_index=args.task_id)
    
    #Run ps / worker
    if args.job_name == 'ps':
        launch_ps(args.num_workers, server)
    else:
        launch_worker(args.task_id, args.num_workers, args.synchronous, server, log_dir)
