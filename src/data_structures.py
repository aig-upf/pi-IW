import heapq
import numpy as np
from collections import defaultdict, deque
from .utils import load_pickle, save_pickle, get_size, is_int, random_index

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
            self.parent.children.remove(self) #just to be consistent
            self.parent = None
            old_depth = self.depth
            for node in self.breadth_first():
                node.depth -= old_depth
                
    def str_node(self, str_data_fn = lambda data:str(data)):
        tab = '   '
        s = str_data_fn(self.data) + '\n'
        for node in self.depth_first():
            d = node.depth - self.depth
            if d > 0:
                s += "".join([tab]*d + ['|', str_data_fn(node.data), '\n'])
        return s

class Tree:
    """
    Although a Node is a tree by itself, this class provides more iterators,
    keeps track of the root node and quick access to the different depths of
    the tree.
    """
    def __init__(self, branching_factor, root_data):
        self.branching_factor = branching_factor
        self.new_root(Node(root_data))

    def __len__(self):
        return len(self.nodes)
    
    def str_tree(self, str_data_fn=lambda data: str(data)):
        return(self.root.str_node(str_data_fn))
    
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
        for d in range(1, self.max_depth+1):
            for node in self.depth[d]:
                if include_leaves or not node.is_leaf():
                    yield node

    def iter_breadth_first_reverse(self, include_root=False, include_leaves=True):
        for d in range(self.max_depth, 0 , -1):
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
            node.children = list() #remove children
        for n in self.root.breadth_first():
            self._add(n) #iterate through children nodes and add them in the depth list, when creating a new RLTree there are no children
            
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

class Queue():
    def __init__(self):
        self.q = []
    
    def push(self, item):
        self.q.append(item)

    def push_many(self, items):
        self.q.extend(list(items))
    
    def pop(self, i=0):
        return self.q.pop(i)
    
    def __len__(self):
        return len(self.q)
    
    def pop_random(self):
        return self.pop(np.random.randint(0, len(self)))
    
class PriorityQueue(Queue):
    """
    Note: heappush does not perfectly sort the list, heapsort does. However,
    items will come out as expected when using heappop.
    Also, removing an item from the queue and not heapifying it may alter the
    result of heappop.
    """
    def __init__(self):
        Queue.__init__(self)
        self.cnt = 0

    def push(self, priority, item):
        heapq.heappush(self.q, ((priority, self.cnt), item)) #cnt will be a unique index so that we always break ties with this, because given items may not be comparable
        self.cnt += 1

    def push_many(self, items):
        raise NotImplementedError()

    def pop(self, i=0):
        if i == 0:
            return heapq.heappop(self.q)
        else:
            self.q[i] = self.q[-1] #found somewhere that this is a better way than pop(i)
            x = self.q.pop()
            heapq.heapify(self.q)
            return x

def load_dataset(filename):
    metadata, info, data = load_pickle(filename)
    dataset = Dataset(metadata['names'], None, info)
    dataset._data = data
    return dataset

def resample_dataset(dataset, new_len):
    idx = np.random.randint(0, len(dataset), size=new_len)
    new_cols = list()
    for c, column_name in enumerate(dataset.column_names):
        new_col = deque(maxlen = new_len)
        for r in idx:
            new_col.append(dataset._data[c][r])
        new_cols.append(new_col)
    dataset._data = tuple(new_cols)


class Dataset:
    def __init__(self, column_names, max_len=None, info={}):
        assert len(column_names) > 0
        self.info = info
        self.column_names = column_names
        self._data = tuple(deque(maxlen=max_len) for _ in self.column_names)
        self._row_size = 0
        self.cumprobs = []  # in case we use linear annealing sampling

    def __len__(self):
        return len(self._data[0])

    @property
    def size(self):
        if self._row_size == 0:
            if len(self) == 0:
                return 0
            self._row_size = get_size(self[0])
        return self._row_size * len(self)

    def __repr__(self):
        return "Dataset of %i rows\nColumn names: %s\nInfo:\n%s" % (len(self), repr(self.column_names), repr(self.info))

    def _iter_row(self, row):
        assert type(row) in (dict, list, tuple)
        assert len(row) == len(self.column_names)
        if type(row) is dict:
            for i, column in enumerate(self.column_names):
                try:
                    yield i, row[column]
                except KeyError:
                    raise KeyError("Row dict does not contain %s" % column)
        else:
            for i, elem in enumerate(row):
                yield i, elem

    def __iter__(self):
        for i in range(len(self)):
            yield self._data[i]

    def __getitem__(self, idx):
        if type(idx) in (list, tuple):
            if type(idx[0] is str):
                assert all(type(i) is str for i in idx)
                return [self.get_column(i) for i in idx]
            else:
                assert all(is_int(i) for i in idx)
                return [self.get_row(i) for i in idx]
        else:
            if type(idx is str):
                return self.get_column(idx)
            else:
                assert is_int(idx), "Bad index"
                return self.get_row(idx)

    def get_row(self, i):
        if i < 0:
            i = len(self) + i
        return [self._data[j][i] for j in range(len(self.column_names))]

    def get_column(self, name):
        return self._data[name]

    def add(self, row):
        for column_idx, new_element in self._iter_row(row):
            self._data[column_idx].append(new_element)

    def sample_indices(self, size):
        return random_index(len(self), size, replace=False)

    def sample(self, size):
        idx = self.sample_indices(size)
        res = list()
        for n, name in enumerate(self.column_names):
            data_n = self._data[n]
            res.append((name, [data_n[i] for i in idx]))
        return dict(res)

    def save(self, filename):
        save_pickle(filename, ({'names': self.column_names}, self.info, self._data))