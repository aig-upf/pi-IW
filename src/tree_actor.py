from collections import defaultdict
from .data_structures import Tree
from .counters import TrainCounters, Ticker
from .utils import node_fill_network_data
from .session_manager import session_manager


class TreeActor:
    """
    Interacts with an environment while adding nodes to a tree.
    """
    def __init__(self, env):
        self.env = env
        self.tree = None
        self.observe_fns = []
        self.observe_NNs = defaultdict(set)

    def make_root(self, node, keep_subtree):
        self.tree.new_root(node, keep_subtree)
        if self.last_node is not self.tree.root:
            self.last_node = None #just in case
        
    def reset_env(self):
        obs = self.env.reset()
        self.tree = Tree(self.env.action_space.n, {"obs": obs, "done": False})
        self.observe(self.tree.root)

    def add_observe_fn(self, fn):
        self.observe_fns.append(fn)

    def request_nn_output(self, nn, layer_name):
        assert layer_name in nn.layers.keys()
        self.observe_NNs[nn].add(layer_name)

    def expand(self, node, action):
        if self.last_node is not node:
            self.env.restore_state(node.data["s"])
        
        #Perform step
        next_obs, r, end_of_episode, info = self.env.step(action)
        node_data = {"a": action, "r": r, "done": end_of_episode, "obs": next_obs}
        node_data.update(info)
        child = self.tree.add(node, node_data)
        TrainCounters.increment("interactions")
        Ticker.tick("interaction")
        
        #if not child.data["done"]:
        self.observe(child)

        return child

    def observe(self, node):
        node.data["s"] = self.env.clone_state()

        for nn, layer_names in self.observe_NNs.items():
            node_fill_network_data(session_manager.session, nn, layer_names, node.data)

        for obs_fn in self.observe_fns:
            obs_fn(self.env, node)

        self.last_node=node