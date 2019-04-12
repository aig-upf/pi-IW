import subprocess
import os
import launch_task
import argparse
import timeit
from distutils.util import strtobool
from src.settings import settings, parse_args


def task(job_name, task_id, args, unused_args):
    task = os.path.abspath(launch_task.__file__)
    task_args = [job_name, str(task_id), str(args.num_workers), str(args.start_port), str(args.synchronous), str(args.log_path), str(args.exp_id)]
    cmd = ['python3', task] + task_args + ['--seed', str(args.seed + task_id)] + unused_args
    return subprocess.Popen(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # disabling help here won't show these next 3 arguments but all of them
    parser.add_argument('num_workers', type=int)
    parser.add_argument('start_port', type=int)
    parser.add_argument('synchronous', type=strtobool)
    parser.add_argument('log_path')
    parser.add_argument('exp_id')
    parser.add_argument('--seed', type=int, default=settings["seed"]) # We parse it here to give each worker seed+worker_id
    args, unused_args = parser.parse_known_args() # parse num workers and port (and seed so that it does not appear in unused_args)

    parse_args() # to show help for all other arguments (unused here)

    try:
        #Launch parameter server
        ps = task('ps', 0, args, unused_args)
        
        #Launch workers
        workers = []
        for i in range(args.num_workers):
            workers.append(task('worker', i, args, unused_args))
        for w in workers:
            w.wait()
        print("All workers finished / died.", flush=True)
        ps.wait()
        print("PS died.", flush=True)

    finally:
        try:
            #Send signal
            print("Stopping all processes...", flush=True)
            for worker in workers:
                worker.send_signal(subprocess.signal.SIGINT)
            ps.send_signal(subprocess.signal.SIGINT)        
            
            #Wait up to total_time
            total_time = 30
            workers_procs = list(enumerate(workers))
            i = 0
            start = timeit.default_timer()
            while timeit.default_timer() - start < total_time:
                idx, worker = workers_procs[i]
                if worker.poll() is not None:
                    print("Worker %i stopped correctly."%idx, flush=True)
                    workers_procs.remove(workers_procs[i])
                if len(workers_procs) == 0:
                    break
                i = (i + 1) % len(workers_procs)
    
            #If there are still processes alive report them
            for idx, worker in workers_procs:
                print("ERROR: Worker %i didn't stop. Killing it..."%idx, flush=True)
            if ps.poll() is None:
                print("PS stopped correctly.", flush=True)
            else:
                print("ERROR: PS didn't stop. Killing it...", flush=True)
        finally:    
            for worker in workers:
                worker.kill()
            ps.kill()
            print("Done!", flush=True)