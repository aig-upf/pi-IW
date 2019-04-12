import numpy as np
import logging
import sys
from contextlib import contextmanager
from collections import defaultdict
import timeit
import gym, gym.wrappers

logger = logging.getLogger(__name__)

#GYM
def create_env(env_id, frameskip=1):
    env = gym.make(env_id)
    if type(env) is gym.wrappers.TimeLimit:
        env = env.env #take out TimeLimit (gym adds it by default)
        
    if hasattr(env.unwrapped, 'frameskip'):
        assert type(frameskip) is int and frameskip >= 1
        env.unwrapped.frameskip = frameskip
        logger.info("Frameskip set to %i."% frameskip)
    else:
        logger.warning("Can't set frameskip.")
    return env


#ARGPARSE
class ArgsManager:
    """
    Small wrapper around argparse. Makes sure booleans are used properly and
    allows setting a value to None.
    All arguments will be optional and will have a default value.
    """

    def __init__(self):
        import argparse
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def _boolean(self, x):
        from distutils.util import strtobool  # allows strings such as 'y', 'yes', 'true' for True, same goes for False
        if type(x) is bool:
            return x
        elif type(x) is str:
            return strtobool(x)
        else:
            raise TypeError("Input must be either a string or a boolean.")

    def _allowNone(self, type_constructor):
        def constructor(x):
            if x in (None, 'None', 'none'):
                return None
            else:
                return type_constructor(x)

        return constructor

    def add_arg(self, name, default, type_constructor, description, short_name=None, choices=None):
        assert not name.startswith('-'), "'--' will be automatically added"
        assert short_name is None or not short_name.startswith('-'), "'-' will be automatically added"
        if type_constructor is bool: type_constructor = self._boolean
        type_constructor = self._allowNone(type_constructor)
        assert default == type_constructor(default), "wrong default's type for " + name + str(type(default)) + "  " + str(type(type_constructor(default)))

        names = ("-" + short_name, "--" + name) if short_name else ("--" + name,)  # include short_name or not
        self.parser.add_argument(*names, default=default, type=type_constructor, help=description, choices=choices)

    def change_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    def parse_args(self):
        return self.parser.parse_known_args()[0]


#TIMING
class TimeSnippets:
    def __init__(self):
        self.snippet_times = defaultdict(int)
        self.snippet_counts = defaultdict(int)

    @contextmanager
    def time_snippet(self, name):
        start = timeit.default_timer()
        yield
        self.snippet_times[name] += (timeit.default_timer() - start)
        self.snippet_counts[name] += 1

    def get_average_time(self, name):
        try:
            return self.snippet_times[name] / self.snippet_counts[name]
        except ZeroDivisionError:
            return 0

    def reset(self, name_or_names):
        if type(name_or_names) not in (list, tuple):
            name_or_names = [name_or_names]
        for name in name_or_names:
            self.snippet_times[name] = 0
            self.snippet_counts[name] = 0

    def print_times(self):
        s = ["Snippet times:"]
        for name in self.snippet_times.keys():
            s.append("%s: %s" % (name, self.get_average_time(name)))
        s.append("-----------")
        print("\n".join(s))


#NUMPY    
def softmax(x, temp=1, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    if temp == 0:
        res = (x == np.max(x, axis=-1))
        return res/np.sum(res, axis=-1)
    x = x/temp
    e_x = np.exp( (x - np.max(x, axis=axis, keepdims=True)) ) #subtracting the max makes it more numerically stable, see http://cs231n.github.io/linear-classify/#softmax and https://stackoverflow.com/a/38250088/4121803
    return e_x / e_x.sum(axis=axis, keepdims=True)


def sample_cdf(cum_probs, size=None):
    s = cum_probs[-1]
    assert s > 0.99999 and s < 1.00001, "Probabilities do not sum to 1: %"%cum_probs #just to check our input looks like a probability distribution, not 100% sure though.
    if size is None:
        # if rand >=s, cumprobs > rand would evaluate to all False. In that case, argmax would take the first element argmax([False, False, False]) -> 0.
        # This may happen still if probabilities sum to 1:
        # cumsum > rand is computed in a vectorized way, and in our machine (looks like) these operations are done in 32 bits.
        # Thus, even if our probabilities sum to exactly 1.0 (e.g. [0. 0.00107508 0.00107508 0.0010773 0.2831216 1.]), when rand is really close to 1 (e.g. 0.999999972117424),
        # when computing cumsum > rand in a vectorized way it will consider it in float32, which turns out to be cumsum > 1.0 -> all False.
        # This is why we check that (float32)rand < s:
        while True:
            rand = np.float32(np.random.rand())
            if rand < s:
                break
        res = (cum_probs > rand)
        return res.argmax()

    if type(size) is int:
        rand = np.random.rand(size).reshape((size,1))
    else:
        assert type(size) in (tuple, list), "Size can either be None for scalars, an int for vectors or a tuple/list containing the size for each dimension."
        assert len(size) > 0, "Use None for scalars."
        rand = np.random.rand(*size).reshape(size +(1,))

    # Again, we check that (float32)rand < s (easier to implement)
    mask = rand.astype(np.float32) >= s
    n = len(rand[mask])
    while n > 0:
        rand[mask] = np.random.rand(n)
        mask = rand.astype(np.float32) >= s
        n = len(rand[mask])
    return (cum_probs > rand).argmax(axis=-1)


def sample_pmf(probs, size=None):
    return sample_cdf(probs.cumsum(), size)


def random_index(array_len, size=None, replace=False, probs=None, cumprobs=None):
    """
    Similar to np.random.choice, but slightly faster.
    """
    if probs is None and cumprobs is None:
        res = np.random.randint(0, array_len, size)
        one_sample = lambda: np.random.randint(0, array_len)
    else:
        assert probs is None or cumprobs is None, "Either both probs and cumprobs is None (uniform probability distribution used) or only one of them is not None, not both."
        if probs is not None:
            cumprobs = probs.cumsum()
        assert array_len == len(cumprobs)
        res = sample_cdf(cumprobs, size)
        one_sample = lambda: sample_cdf(cumprobs)

    if not replace and size is not None:
        assert size <= array_len
        s = set()
        for i in range(size):
            l = len(s)
            s.add(res[i])
            while len(s) == l:
                res[i] = one_sample()
                s.add(res[i])
    return res


#PICKLE
def save_pickle(pickle_file, save_obj):
    import pickle
    assert type(pickle_file) is str
    with open(pickle_file, 'wb') as f:
        pickle.dump(save_obj, f, pickle.HIGHEST_PROTOCOL) #-1 would be equivalent to pickle.HIGHEST_PROTOCOL (much more efficient)

def load_pickle(pickle_file):
    import pickle
    with open(pickle_file, 'rb') as f:
        obj = pickle.load(f)
    return obj

    
#LOGGING
def configure_root_logger(log_to_stdout=True, filename=None, enable_logs=True, log_level=logging.INFO, fmt="[%(asctime)s] %(levelname)-.1s %(message)s"):
    import sys
    
    if enable_logs:
        handlers = list()
        if filename is not None:
            handlers.append(logging.FileHandler(filename=filename))
        if log_to_stdout:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        root_logger = logging.getLogger()
        root_logger.handlers = [] #root_logger.removeHandler() does not always work
        
        root_logger.setLevel(log_level)
        formatter = logging.Formatter(fmt=fmt)
        for handler in handlers:
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
    
def configure_root_logger_cluster(job_name, task_id, log_to_stdout=True, filename=None, enable_logs=True, log_level=logging.INFO):
    configure_root_logger(log_to_stdout, filename, enable_logs, log_level, "[%(asctime)s] {}:{} %(levelname)-.1s %(message)s".format(job_name, task_id))


def get_size(obj):
    #For a more complete function see https://stackoverflow.com/a/30316760
    if type(obj) is np.ndarray:
        return obj.nbytes
    if type(obj) in (tuple, list, set):
        return np.sum([get_size(elem) for elem in obj])
    return sys.getsizeof(obj)
    

def preprocess_obs_list(obs_list):
    if type(obs_list[0]) is list:
        return [np.stack(obs, axis=-1) for obs in obs_list]
    return obs_list


def node_fill_network_data(session, network, layer_names, node_data):
    names = list(layer_names)
    output_tensors = [network.layers[name] for name in names]

    net_data = session.run(output_tensors,
                           feed_dict={network.layers["input"]: preprocess_obs_list([node_data["obs"]])})
    net_data = [out.squeeze(axis=0) for out in net_data]
    node_data[network] = dict(list(zip(names, net_data)))


def is_int(x):
    return np.issubdtype(type(x), np.integer)


def is_float(x):
    return np.issubdtype(type(x), np.floating)
