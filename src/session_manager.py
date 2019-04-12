import tensorflow as tf
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@contextmanager
def variable_name_scope(name, reuse=False):
    """
    Trying to unify variable and name scopes. Variable scopes are reusable in
    the sense of ending up with the same name (unless we pass None as argument)
    These names only apply to variables created with the get_variable() method
    though, and all other operations (including creating a variable with
    tf.Variable) will end up with a different name everytime we enter a with
    statement (those use the name scope instead of the variable scope, which is
    also set by tf.variable_scope, but it is not reused across different calls)
    Name scopes only end up with the same name if they are set as absolute.
    Variable scopes cannot be absolute, check
    https://github.com/tensorflow/tensorflow/issues/9557.

    Desired effect:
        with variable_name_scope("a"):
            tf.Variable(0, name="var1") # 'a/var1:0'
            tf.get_variable("var2", ()) # 'a/var2:0'
            b = a*2                     # 'a/mul:0'
        do_something()
        with variable_name_scope("a"):
            tf.Variable(0, name="var1") # 'a/var1_1:0'
            c = a*2                     # 'a/mul_1:0'
            tf.get_variable("var2", ()) # raises exception
        do_something()
        with variable_name_scope("a", reuse=True):
            tf.Variable(0, name="var1") # 'a/var1_2:0'
            tf.get_variable("var2", ()) # 'a/var2:0' (same variable as before!)

    Note that only variables with tf.get_variable will be reused, all other
    operations will be unique in the graph. Having the same name scope makes
    them be grouped when inspecting the graph with tensorboard, and in practice
    only moves the index that makes the name unique towards the end, (i.e
    'a/b/var1_1' instead of 'a/b_1/var1' or 'a_1/b/var1', for instance). See
    the discussion https://github.com/tensorflow/tensorflow/issues/6007 for
    other solutions.
    """
    assert not name.endswith('/'), "Absolute scopes disallowed."
    assert name is not None, "A name is required. Also, cannot set the root scope for variable_scopes"
    assert name != "", "An empty string will result in a name with two '/', it doesn't set the root variable scope as in name scopes."
    current_name_scope = tf.get_default_graph().get_name_scope()
    if len(current_name_scope) > 0:
        current_name_scope += "/"  # if its not the root scope, we add '/'
    name_scope = current_name_scope + name + "/"  # +"/" to make it absolute and avoid adding suffixes "_1", "_2"...
    with tf.variable_scope(name, reuse=reuse) as var_scope:
        # Let's set the name_scope of the variable_scope to the one we want
        var_scope._name_scope = name_scope  # HACK: may break in the future
        with tf.name_scope(name_scope):
            yield var_scope


@contextmanager
def absolute_name_scope(name):
    """
    Just to make explicit that a name_scope is absolute (appends '/').
    Note that tf.get_variable() ignores name_scopes and uses variable_scopes
    instead.
    """
    if not name.endswith('/') and len(name) > 0:
        name += '/'
    with tf.name_scope(name) as scope:
        yield scope


@contextmanager
def variables_in_collection(collection):
    """
    When exiting the context manager, all variables that have been created (and
    added to the global variables collection) will be added to the specified
    collection.
    """
    old_variables = set(tf.global_variables())
    yield
    new_variables = [x for x in tf.global_variables() if x not in old_variables]  # order is preserved
    for var in new_variables:
        tf.add_to_collection(collection, var)


class SessionManager():
    def __init__(self):
        self._optimizer = None
        self._initialized = False
        self._session = None
        self.summary_writer = None
        self.summaries = dict()

    @property
    def is_chief(self):
        assert self._initialized
        return self.task_id == 0

    def initialize(self, log_dir=None, task_id=0, num_workers=1, target='', worker_device="/job:localhost/task:0/", ps_device="/job:localhost/task:0/", synchronous=True, write_summaries=False, save_checkpoints=False, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1):
        if self._initialized:
            logger.warning("Reinitializing session manager. Reseting tensorflow graph...")
            self.close()

        self.task_id = task_id
        self.num_workers = num_workers
        assert task_id < num_workers
        self.target = target
        self.log_dir = log_dir
        self.synchronous = synchronous
        self.intra_op_parallelism_threads = intra_op_parallelism_threads
        self.inter_op_parallelism_threads = inter_op_parallelism_threads
        self.save_checkpoints = save_checkpoints
        self.write_summaries = write_summaries
        self.worker_device = worker_device
        self.ps_device = ps_device

        tf.reset_default_graph()
        self._initialized = True
        with tf.device("cpu:0"), self.ps():
            self.global_step = tf.train.get_or_create_global_step() #puts the variable in GLOBAL_STEP collection for later use! optimizer may increase it and use it for synchronization

    def _check_initialized(self):
        if not self._initialized:
            logger.warning(
                "Session manager has not been initialized. Creating new LOCAL session with default configuration (won't be closed automatically).")
            self.initialize()

    @contextmanager
    def worker(self):
        self._check_initialized()
        with tf.device(self.worker_device), variable_name_scope("worker_scope"), variables_in_collection("worker_variables"):
            yield
        
    @contextmanager
    def ps(self):
        self._check_initialized()
        with tf.device(self.ps_device), variable_name_scope("ps_scope"), variables_in_collection("ps_variables"):
            yield

    @property
    def execution_is_distributed(self):
        self._check_initialized()
        return not (self.num_workers == 1 and self.worker_device == self.ps_device)

    @property
    def session(self):
        if not self._session:
            self._check_initialized()
            assert not tf.get_default_graph().finalized, "TF graph is finalized."
            with tf.device("cpu:0"), tf.device(self.worker_device), variable_name_scope("worker_scope"):
                ps_vars_init_op = tf.variables_initializer(tf.get_collection("ps_variables"), "ps_variables_init") #it actually goes to ps because variables have to be assigned in their devices
                ps_vars_ready_op = tf.report_uninitialized_variables(tf.get_collection("ps_variables"), "report_uninitialized_ps_variables")
                worker_vars_init_op = tf.variables_initializer(tf.get_collection("worker_variables"), "worker_variables_init")
                all_vars_ready_op = tf.report_uninitialized_variables(tf.global_variables(), "report_uninitialized_variables")

            class FastSaver(tf.train.Saver): #Seen in OpenAI's implementation
                # Disables write_meta_graph argument, which freezes entire process and is mostly useless.
                # Taken from: https://github.com/openai/universe-starter-agent/blob/master/worker.py
                #TODO: check if this is ok and necessary
                def save(self, sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix="meta", write_meta_graph=True):
                    super(FastSaver, self).save(sess, save_path, global_step, latest_filename, meta_graph_suffix, False)

            if self.log_dir and self.write_summaries:
                if self.is_chief:
                    self.summary_writer = tf.summary.FileWriterCache.get(logdir = self.log_dir)
                else:
                    self.summary_writer = tf.summary.FileWriter(logdir = self.log_dir + '/worker%i' % self.task_id, graph=None)
            else:
                self.summary_writer = None

            saver = FastSaver(var_list = tf.get_collection("ps_variables"), max_to_keep = 1) if self.log_dir and self.save_checkpoints else None

            #We define initializers and set saver
            scaffold = tf.train.Scaffold(init_op = ps_vars_init_op,
                                         ready_for_local_init_op= ps_vars_ready_op,
                                         local_init_op = worker_vars_init_op,
                                         ready_op = all_vars_ready_op,
                                         saver = saver)

            #Hooks
            hooks = []
            if self._optimizer and self.synchronous and self.num_workers > 1:
                hooks.append(self._optimizer.make_session_run_hook(self.is_chief, num_tokens=0)) #Important! num_tokens has to be 0 here to have an effective barrier.

            #Config proto:
            #https://stackoverflow.com/questions/43084960/tensorflow-variables-are-not-initialized-using-between-graph-replication
            #https://github.com/MatheusMRFM/A3C-LSTM-with-Tensorflow/pull/1
            #https://github.com/openai/universe-starter-agent/blob/f16f37d9d3bc8146cf68a75557e1ba89824b7e54/worker.py#L143
            #https://stackoverflow.com/questions/41233635/tensorflow-inter-and-intra-op-parallelism-configuration
            config = tf.ConfigProto(device_filters=[self.ps_device, self.worker_device], intra_op_parallelism_threads=self.intra_op_parallelism_threads, inter_op_parallelism_threads=self.inter_op_parallelism_threads)

            self._session = tf.train.MonitoredTrainingSession(master = self.target,
                                                              is_chief = self.is_chief,
                                                              checkpoint_dir = self.log_dir,
                                                              scaffold = scaffold,
                                                              hooks = hooks,
                                                              save_checkpoint_secs = 600 if self.save_checkpoints else None,
                                                              save_summaries_steps = None, #disable auto-summaries
                                                              save_summaries_secs = None, #disable auto-summaries
                                                              config = config,
                                                              stop_grace_period_secs=10) #SyncReplicasOptimizer's threads are not stopped, don't know why https://github.com/tensorflow/tensorflow/issues/13779
        return self._session

    @contextmanager
    def configure(self, log_dir=None, task_id=0, num_workers=1, target='', worker_device="/job:localhost/task:0/", ps_device="/job:localhost/task:0/", synchronous=True, write_summaries=False, save_checkpoints=False, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1):
        self.initialize(log_dir, task_id, num_workers, target, worker_device, ps_device, synchronous, write_summaries, save_checkpoints, intra_op_parallelism_threads, inter_op_parallelism_threads)
        try:
            yield
        finally:
            self.close()

    def __del__(self):
        if self._initialized:
            self.close()

    def close(self):
        assert self._initialized
        if self.summary_writer:
            self.summary_writer.close()
        if self._session:
            self._session.close()  # sess.close() last because it fails to stop tensorflow threads with SyncReplicasOptimizer
            self._session = None
        self._initialized = False

    def set_optimizer(self, optimizer):
        assert self._initialized
        assert self._optimizer is None, "Trying to set optimizer twice."
        if self.synchronous and self.num_workers > 1:
            self._optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=self.num_workers, total_num_replicas=self.num_workers)
        else:
            self._optimizer = optimizer

    def apply_grads(self, grads_and_vars):
        #Tensorflow collocates variables created by apply_gradients in the same device as the variables to optimize.
        #Thus, it is not possible to have separate statistics for each worker.
        #Also, the new optimizer variables will be in the same namespace as the variables to optimize
        assert self._initialized
        assert self._optimizer is not None, "Optimizer needs to be set."
        with variables_in_collection("ps_variables"):
            apply_grads_op = self._optimizer.apply_gradients(grads_and_vars, self.global_step) #global step will be automatically increased. Also, it will be used to synchronize workers.
        return apply_grads_op

    def add_scalar_summaries(self, name_or_names, dtype=tf.float32, scope=""):
        if type(name_or_names) is not list: name_or_names = [name_or_names]
        for summary_name in name_or_names:
            assert summary_name not in self.summaries.keys()
            with absolute_name_scope(scope):
                ph = tf.placeholder(dtype, (), name=summary_name + '_ph')
                summary_op = tf.summary.scalar(summary_name, ph)
            self.summaries[summary_name] = (ph, summary_op)

    def compute_summaries(self, names_and_values, step):
        if not self.session.should_stop():
            if type(names_and_values) is not list: names_and_values = [names_and_values]
            for name, value in names_and_values:
                ph, summary_op = self.summaries[name]
                summary = self.session.run(summary_op, feed_dict={ph: value})
                self.summary_writer.add_summary(summary, step)


session_manager = SessionManager()