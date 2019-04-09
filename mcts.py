
import numpy as np
from utils import softmax, sample_pmf


class MCTSAlphaZero:
    """
    Implementation of Monte Carlo Tree Search as in AlphaZero.
    Expands nodes depending on the following metric:
        U(s,a) = c * P(s,a) * (sqrt(sum_b( N(s,b) )) / ( 1+N(s,a) ))
    """
    def __init__(self, branching_factor, discount_factor=0.99, puct_factor=1, noise_eps=0.25, noise_alpha=0.03):
        self.branching_factor = branching_factor
        self.discount_factor = discount_factor
        self.puct_factor = puct_factor
        self.noise_eps = noise_eps
        self.noise_alpha = noise_alpha

    def plan(self, tree, successor_fn, stop_condition_fn=lambda:False):
        if "N" not in tree.root.data.keys():
            assert len(tree) == 1  # We just created a new tree (new episode started), there's only one (root) node
            self.init_counters(tree.root)

        should_stop = False
        while not should_stop:
            # Select
            node, a = self.select(tree.root)

            # Expand and evaluate
            if node.data["done"]:
                assert a is None
                child = node  # we cannot expand, but we need to back up -> counts should increase
            else:
                child = successor_fn(node, a)
                self.init_counters(child)

            # Backup
            self.backup(child)
            should_stop = stop_condition_fn() # increase counter / check time

    def select(self, node):
        """
        Selects a node and an action to expand in a tree.
        Returns (node, action) to expand, or (None, None) if the subtree has
        been solved.
        """
        while True:
            if node.data["done"]:
                return node, None

            policy = self._get_policy(node)
            a = sample_pmf(policy)

            if node.is_leaf():
                # return action to expand
                return node, a

            not_in_tree = True
            for child in node.children:
                if child.data["a"] == a:
                    node = child
                    not_in_tree = False
                    break
            if not_in_tree:
                return node, a

    def backup(self, node):
        R = node.data["r"] if node.data["done"] else node.data["r"] + self.discount_factor * node.data["v"]
        while True:
            a = node.data["a"]
            node = node.parent
            node.data["N"][a] += 1
            # here we consider we are interacting with an environment
            # for self play we should multiply R by 1/-1 (player1, player2)
            node.data["W"][a] += R

            if node.is_root():
                return
            else:
                R = node.data["r"] + self.discount_factor * R

    def compute_U(self, node, add_noise):
        prior = node.data["probs"]
        if add_noise:
            noise = np.random.dirichlet([self.noise_alpha] * self.branching_factor)
            prior = (1 - self.noise_eps) * prior + self.noise_eps * noise

        N = node.data["N"] + 1
        sqrt_sum_counts = np.sqrt(np.sum(N))
        return self.puct_factor * prior * sqrt_sum_counts / (N + 1)

    def init_counters(self, node):
        if not node.data["done"]:
            node.data["W"] = np.zeros((self.branching_factor,))
            node.data["N"] = np.zeros((self.branching_factor,))

    def _get_policy(self, node):
        U = self.compute_U(node, node.is_root())  # we'll add noise to the root node
        Q = (node.data["W"] + node.data["v"]) / (node.data["N"] + 1)
        return softmax(Q + U, temp=0)