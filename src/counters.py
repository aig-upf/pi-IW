from collections import defaultdict
from src.session_manager import session_manager
import tensorflow as tf

class Ticker:
    """
    Executes a sequence of functions every tick() call.
    """
    ticker_fns = defaultdict(list)
    
    @classmethod
    def tick(cls, ticker_name):
        fns = cls.ticker_fns[ticker_name]
        for fn in fns:
            fn()
        
    @classmethod
    def add(cls, fn, ticker_name):
        assert callable(fn)
        cls.ticker_fns[ticker_name].append(fn)

class Counter:
    def __init__(self, name):
        self.name = name
        self._value = 0

    def increment(self, inc=1):
        self._value += inc
        
    def reset(self):
        self._value = 0
        
    @property
    def value(self):
        return self._value

class GlobalCounter(Counter):
    def __init__(self, name, shape=(), dtype=tf.int64, use_locking=True):
        Counter.__init__(self, name)
        self.shape = shape
        self.dtype = dtype
        self.use_locking = use_locking
        
        self.inc_ph = None
        self.inc_op = None
        self.var = None
        
        self._inc = 0
        self.need_update = True
        
    def build(self):
        with tf.device("cpu:0"), session_manager.ps():
            self.var = tf.get_variable(self.name+"_var",
                                       shape=self.shape,
                                       dtype=self.dtype,
                                       initializer=tf.constant_initializer(value=0, dtype=self.dtype),
                                       trainable=False)
            self.inc_ph = tf.placeholder(dtype=self.dtype, shape=self.shape, name="inc_"+self.name+"_ph")
            self.inc_op = self.var.assign_add(self.inc_ph, use_locking=self.use_locking)
    
    def get_inc_op(self, feed_dict={}):
        assert self.inc_op is not None, "%s: First call build method."%self.name
        feed_dict.update({self.inc_ph: self._inc})
        self._inc = 0
        self.need_update=True
        return self.inc_op, feed_dict
    
    def increment(self, amount):
        self._inc += amount

    def update(self, value):
        assert self.need_update or self._inc == 0, "Value incremented between fill_feed_dict and update calls."
        self.need_update = False
        self._value = value
    
    def reset(self):
        raise NotImplementedError("Can't reset a global counter.")
    
    @property
    def value(self):
        return self._value + self._inc

class CounterGroup:
    def __init__(self, allow_increment, allow_reset):
        self.counters = list()
        self.allow_increment = allow_increment
        self.allow_reset = allow_reset
    
    def __iter__(self):
        for counter in self.counters:
            yield counter
    
    def increment(self, amount=1):
        assert self.allow_increment
        for counter in self.counters:
            counter.increment(amount)
    
    def reset(self):
        assert self.allow_reset
        for counter in self.counters:
            counter.reset()
            
    @property
    def value(self):
        raise Exception("A group of counters has no value. Wrong counter?")
            
class CounterHandler:
    def __init__(self):
        self.counters = dict()
        self._global_counters = CounterGroup(allow_increment=False, allow_reset=False)
        self._global_counters.global_vars = None
        self._global_counters.inc_op = None
    
    def new_group(self, name, allow_increment=True, allow_reset=True):
        assert not name in self.counters.keys(), "Group or counter with name %s already exists."%name
        self.counters[name] = CounterGroup(allow_increment, allow_reset)
    
    def new_counter(self, name, add_to_groups=[], is_global=False):
        assert not name in self.counters.keys(), "Group or counter with name %s already exists."%name
        
        counter = GlobalCounter(name) if is_global else Counter(name)
        self.counters[name] = counter
        
        if is_global:
            assert self._global_counters.global_vars is None, "Already built. Can't add more global counters."
            self._global_counters.counters.append(counter)
        
        for group_name in add_to_groups:
            assert group_name in self.counters.keys(), "Group %s does not exist."%group_name
            group = self.counters[group_name]
            assert type(group) is CounterGroup
            group.counters.append(counter)
        
    def increment(self, counter_or_group, amount=1):
        self.counters[counter_or_group].increment(amount)
    
    def reset(self, counter_or_group):
        self.counters[counter_or_group].reset()
        
    def __getitem__(self, counter_name):
        try:
            return self.counters[counter_name].value
        except KeyError:
            raise KeyError("Counter with name %s does not exist."%counter_name)
        
    def build_global_counters(self):
        assert self._global_counters.global_vars is None, "Already built."
        self._global_counters.global_vars = list()
        inc_ops = list()
        for global_counter in self._global_counters:
            global_counter.build()
            self._global_counters.global_vars.append(global_counter.var)
            inc_ops.append(global_counter.inc_op)
        self._global_counters.inc_op = tf.group(*inc_ops)

    def get_inc_op(self, feed_dict={}):
        for global_counter in self._global_counters:
            _, feed_dict = global_counter.get_inc_op(feed_dict)
        return self._global_counters.inc_op, feed_dict

    def update_global_counters(self):
        res = session_manager.session.run(self._global_counters.global_vars)
        for ref, value in zip(self._global_counters, res):
            ref.update(value)


# =============================================================================
#   Counters: global variables
# =============================================================================
TrainCounters = CounterHandler()
TrainCounters.new_group("transitions", allow_increment=True, allow_reset=False)
TrainCounters.new_group("interactions", allow_increment=True, allow_reset=False)
TrainCounters.new_group("episode", allow_increment=False, allow_reset=True)

TrainCounters.new_counter("local_transitions", add_to_groups=["transitions"]) #number of RL steps (not including planning steps)
TrainCounters.new_counter("local_episodes") #number of terminal transitions
TrainCounters.new_counter("local_interactions", add_to_groups=["interactions"]) #number of interactions with the environment (RL steps and/or planning steps)

TrainCounters.new_counter("global_transitions", add_to_groups=["transitions"], is_global=True)
TrainCounters.new_counter("global_interactions", add_to_groups=["interactions"], is_global=True)

TrainCounters.new_counter("episode_transitions", add_to_groups=["episode", "transitions"])
TrainCounters.new_counter("episode_reward",  add_to_groups=["episode"])
