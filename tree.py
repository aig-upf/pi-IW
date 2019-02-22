import numpy as np
from collections import defaultdict

class Node:
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        if self.parent:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        self.children = []

    def size(self):
        return np.sum([c.size() for c in self.children]) + 1

    def breadth_first(self):
        current_nodes = [self]
        while len(current_nodes) > 0:
            children = []
            for node in current_nodes:
                yield node
                children.extend(node.children)
            current_nodes = children

    def depth_first(self):
        yield self
        for child in self.children:
            for node in child.depth_first():
                yield node

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def add(self, data):
        return Node(data, parent=self)

    def make_root(self):
        if not self.is_root():
            self.parent.children.remove(self)  # just to be consistent
            self.parent = None
            old_depth = self.depth
            for node in self.breadth_first():
                node.depth -= old_depth

    def str_node(self, str_data_fn=lambda data: str(data)):
        tab = '   '
        s = str_data_fn(self.data) + '\n'
        for node in self.depth_first():
            d = node.depth - self.depth
            if d > 0:
                s += "".join([tab] * d + ['|', str_data_fn(node.data), '\n'])
        return s


class Tree:
    """
    Although a Node is a tree by itself, this class provides more iterators and quick access to the different depths of
    the tree, and keeps track of the root node
    """

    def __init__(self, branching_factor, root_data):
        self.branching_factor = branching_factor
        self.new_root(Node(root_data))

    def __len__(self):
        return len(self.nodes)

    def str_tree(self, str_data_fn=lambda data: str(data)):
        return (self.root.str_node(str_data_fn))

    def iter_depth_first(self, include_root=False, include_leaves=True):
        iterator = self.root.depth_first()
        try:
            root = next(iterator)
            if include_root:
                yield root
            while True:
                node = next(iterator)
                if include_leaves or not node.is_leaf():
                    yield node
        except StopIteration:
            pass

    def iter_breadth_first(self, include_root=False, include_leaves=True):
        if include_root:
            yield self.root
        for d in range(1, self.max_depth + 1):
            for node in self.depth[d]:
                if include_leaves or not node.is_leaf():
                    yield node

    def iter_breadth_first_reverse(self, include_root=False, include_leaves=True):
        for d in range(self.max_depth, 0, -1):
            for node in self.depth[d]:
                if include_leaves or not node.is_leaf():
                    yield node
        if include_root:
            yield self.root

    def new_root(self, node, keep_subtree=True):
        node.make_root()
        self.root = node
        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)
        if not keep_subtree:
            node.children = list()  # remove children
        for n in self.root.breadth_first():
            self._add(n)  # iterate through children nodes and add them to the depth list

    def _add(self, node):
        self.depth[node.depth].append(node)
        self.nodes.append(node)
        if node.depth > self.max_depth: self.max_depth = node.depth

    def add(self, parent_node, data):
        child = parent_node.add(data)
        self._add(child)
        return child

    def extract_trajectory(self, node):
        trajectory = [node.data]
        while not node.is_root():
            node = node.parent
            trajectory.append(node.data)
        return list(reversed(trajectory))


class TreeActor:
    """
    Interacts with an environment while adding nodes to a tree.
    """

    def __init__(self, env, allow_restore, observe_fn=None):
        self.env = env
        self.allow_restore = allow_restore
        self.tree = None
        self.observe_fn = observe_fn if observe_fn is not None else lambda x: x

    def make_root(self, node, keep_subtree):
        self.tree.new_root(node, keep_subtree)
        if self.last_node is not self.tree.root:
            self.last_node = None  # just in case, we'll restore before expanding

    def reset_env(self):
        obs = self.env.reset()
        self.tree = Tree(self.env.action_space.n, {"obs": obs, "done": False})
        self.observe(self.tree.root)
        return self.tree

    def step(self, node, action):
        if self.last_node is not node:
            assert self.allow_restore
            self.env.unwrapped.restore_state(node.data["s"])

        # Perform step
        next_obs, r, end_of_episode, info = self.env.step(action)
        node_data = {"a": action, "r": r, "done": end_of_episode, "obs": next_obs}
        node_data.update(info) # add extra info e.g. atari lives
        child = self.tree.add(node, node_data)
        self.observe(child)
        return child

    def observe(self, node):
        if self.allow_restore:
            node.data["s"] = self.env.unwrapped.clone_state()
        self.observe_fn(self.env, node)
        self.last_node = node
