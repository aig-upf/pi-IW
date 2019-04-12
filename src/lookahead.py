import numpy as np
import logging
from collections import deque
from .utils import sample_pmf, softmax
from .counters import TrainCounters
from .settings import settings

logger = logging.getLogger(__name__)
    
class Lookahead:
    REQUIRES_NN = False

    def __init__(self, planner, save_trajectory=False):
        self.planner = planner
        self.save_trajectory = save_trajectory
        self._done = True

    def get_next_node(self, tree):
        assert not tree.root.is_leaf()
        lookahead_policy, target_policy = self.get_policies(tree)
        a = sample_pmf(lookahead_policy)
        tree.root.data["target_policy"] = target_policy

        next_node = None
        for child in tree.root.children:
            if a == child.data["a"]:
                next_node = child
        assert next_node is not None, "Selected action not in tree. Something wrong with the tree policy?"

        return next_node

    def step(self):
        if self._done:
            self.planner.actor.reset_env()
            maxlen = None if self.save_trajectory else 2
            self.trajectory = deque([self.planner.actor.tree.root.data], maxlen=maxlen)

        # generate tree
        expanded_nodes = self.planner.plan()

        next_node = self.get_next_node(self.planner.actor.tree)
        self.trajectory.append(next_node.data)

        #observe transition
        self._done = next_node.data["done"]

        # "take a step" (actually remove other branches and make selected child root)
        self.planner.actor.make_root(next_node, settings["cache_subtree"])

        return self.trajectory, expanded_nodes

    def get_policies(self, tree):
        raise NotImplementedError()


class LookaheadReturns(Lookahead):
    def compute_cumulative_rewards(self, tree):
        for node in tree.iter_breadth_first_reverse(include_root=False, include_leaves=True):
            if node.is_leaf():
                R = node.data["r"]
            else:
                R = node.data["r"] + settings["discount_factor"] * np.max([child.data["R"] for child in node.children])
            node.data["R"] = R

    def get_policies(self, tree):
        Q = self._compute_Q(tree)
        policy = softmax(Q, temp=0)
        return policy, policy

    def _compute_Q(self, tree):
        self.compute_cumulative_rewards(tree)
        Q = np.empty(self.planner.actor.env.action_space.n)
        Q.fill(-np.inf)
        for child in tree.root.children:
            Q[child.data["a"]] = child.data["R"]
        return Q


class LookaheadCounts(Lookahead):
    def _tree_policy(self, node, temp):
        assert "N" in node.data.keys(), "Counts not present in tree. Use a planner that computes counts."
        if temp > 0:
            aux = node.data["N"] ** (1 / temp)
            return aux / np.sum(aux)
        else:
            assert temp == 0
            return softmax(node.data["N"], temp=0)

    def get_policies(self, tree):
        if settings["alphazero_firstmoves_temp"] is None or TrainCounters["episode_transitions"] <= settings["alphazero_firstmoves_temp"]:
            tree_policy = self._tree_policy(tree.root, settings["alphazero_target_policy_temp"])
        else:
            tree_policy = self._tree_policy(tree.root, temp=0)
        return tree_policy, tree_policy
