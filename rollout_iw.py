import numpy as np
from collections import defaultdict
from utils import sample_pmf

class RolloutNovelty1Table():
    def __init__(self, ignore_cached_nodes):
        self.atom_depth = defaultdict(lambda: np.inf)
        self.ignore_cached_nodes = ignore_cached_nodes  # only features from new nodes in the tree will be added to the novelty table

    def check(self, atoms, depth, node_is_new):
        for atom in atoms:
            if depth < self.atom_depth[atom] or (not node_is_new and depth == self.atom_depth[atom]):
                return 1  # at least one atom is either case 1 or 4
        return np.inf  # all atoms are either case 2 or 3

    #TODO: make update() function, and remove ignore_cached_nodes from init -> check_and_update(ignore_cached_nodes=True, ...) and remove node_is_new??

    def check_and_update(self, atoms, depth, node_is_new):
        is_novel = False
        for atom in atoms:
            if depth < self.atom_depth[atom]:
                if self.ignore_cached_nodes:
                    if node_is_new:
                        # here node_is_new controls that existing nodes (already in the tree) are not added to the table (and not pruned)
                        self.atom_depth[atom] = depth
                else:
                    # all nodes
                    self.atom_depth[atom] = depth
                is_novel = True  # case 1, novel
            # else if node_is_new, case 2, not novel
            elif not node_is_new and depth == self.atom_depth[atom]:
                is_novel = True  # case 4, was novel before and is still novel
            # else, case 3, was novel before but not anymore
        return 1 if is_novel else np.inf

class RolloutIW:
    """
    Rollout IW as in Improving width-based planning with compact policies (Junyent et al. 2018)
    """

    def __init__(self, branching_factor, max_novelty=1, ignore_cached_nodes=False, ignore_terminal_nodes=False, min_cum_prob=0):
        self.branching_factor = branching_factor
        self.max_novelty = max_novelty
        assert self.max_novelty == 1, "Novelty > 1 not implemented"
        self.ignore_cached_nodes = ignore_cached_nodes
        self.ignore_terminal_nodes = ignore_terminal_nodes
        self.min_cum_prob = min_cum_prob # Prune when the cumulative probability of the remaining (not solved) actions is lower than this threshold

    def plan(self, tree, expand_fn, stop_condition_fn=lambda:False):
        self.novelty_table = RolloutNovelty1Table(self.ignore_cached_nodes)

        #Online planning only:
        self.initialize(tree) # To deal with an existing tree (maybe initialize novelty table with existing nodes, etc)

        while not stop_condition_fn() and not tree.root.solved:
            #Select
            node, a = self.select(tree.root)
            if a is not None:
                self.rollout(node, a, expand_fn, stop_condition_fn)
        
    def initialize(self, tree):
        #Set solved
        for node in tree.iter_breadth_first(include_root=True, include_leaves=True):
            assert "features" in node.data.keys(), "IW planners require the state to be factored into features"
            node.solved = False
            if node.data["done"]:
                if not self.ignore_terminal_nodes and not self.ignore_cached_nodes:
                    # We add features of terminal nodes to novelty table (Bandres et al. (2018) implementation doesn't do that!)
                    novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=True)
                self.solve_and_propagate_label(node)
            else:
                if not self.ignore_cached_nodes:
                    # add cached nodes to novelty table, and maybe prune them
                    novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, True)
                    if novelty > self.max_novelty:
                        self.solve_and_propagate_label(node)
                    
    def _get_policy(self, node):
        #Uniform policy
        p = np.empty((self.branching_factor,))
        p.fill(1/self.branching_factor)
        return p
    
    def select(self, node):
        """
        Selects a node and an action to expand in a tree.
        Returns (node, action) to expand, or (None, None) if the subtree has
        been solved.
        """
        assert not node.data["done"]
        # if node.data["done"]:
        #     self.solve_and_propagate_label(node)
        #     return None, None

        while True:
            assert not node.solved and not node.data["done"], "Solved: %s.  Done: %s.  Depth: %s"%(str(node.solved), str(node.data["done"]), str(node.depth))
            novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=False)
            if novelty > self.max_novelty:
                self.solve_and_propagate_label(node)
                return None, None # Prune node
    
            a, child = self.select_action_following_policy(node)
            assert child is None or (not child.solved and not child.data["done"]), "Solved: %s.  Done: %s.  Depth: %s"%(str(child.solved), str(child.data["done"]), str(child.depth))

            if a is None:
                return None, None # All actions recommended by the policy have been already solved for this node
            else:
                assert a < self.branching_factor, "Ilegal action recommended by the policy (action a=%i >= action_space.n=%i)." % (a, self.branching_factor)
                if child is None:
                    return node, a # Action a needs to be expanded for this node
                else:
                    node = child # Continue traversing the tree
            
    def select_action_following_policy(self, node):
        policy = self._get_policy(node)
        if node.is_leaf():
            #return action to expand
            assert not node.solved and not node.data["done"], "Solved: %s.  Done: %s.  Depth: %s"%(str(node.solved), str(node.data["done"]), str(node.depth))
            return sample_pmf(policy), None

        node_children = [None]*self.branching_factor
        available_actions = (policy > 0)
        for child in node.children:
            node_children[child.data["a"]] = child
            if child.solved:
                available_actions[child.data["a"]] = False

        #Take out actions that have been solved
        p = (policy*available_actions)
        ps = p.sum()
            
        #No actions available?
        if ps <= self.min_cum_prob:
            #All actions recommended by the policy (i.e. with prob >0) have been (or should be considered) solved. Solve node.
            #It is posible that some nodes in the subtree are not marked as solved. That's not a problem, and it's because the policy gives those branches low probability (less than min_prob)
            self.solve_and_propagate_label(node)
            return None, None
        
        #Select action not solved
        p = p/ps
        assert all((p>0) == available_actions), "p: %s;  avail_a: %s;   ps:%s"%(str(p), str(available_actions), str(ps))
        a = sample_pmf(p)

        child = node_children[a]
        if child:
            assert not child.solved and not child.data["done"], "a: %i, Solved: %s.  Done: %s.  Depth: %s.  policy: %s.  avail_actions: %s.  p: %s.  ps: %s.  children: %s."%(a, str(child.solved), str(child.data["done"]), str(child.depth), str(policy), str(available_actions), str(p), str(ps), str([(c.data["a"], c.solved) for c in node.children]))

        return a, child
    
    def rollout(self, node, a, expand_fn, stop_condition_fn):
        while not stop_condition_fn():
            node = expand_fn(node, a)
            node.solved = False

            if node.data["done"]:
                if not self.ignore_terminal_nodes:
                    # We add features of terminal nodes to novelty table (Blai's implementation doesn't do that!)
                    novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=True)
                self.solve_and_propagate_label(node)
                return
            else:
                novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=True)
                if novelty > self.max_novelty:
                    self.solve_and_propagate_label(node)
                    return

            a, child = self.select_action_following_policy(node)
            assert a is not None and child is None, "Action: %s, child: %s"%(str(a), str(child))
        return

    def solve_and_propagate_label(self, node):
        node.solved = True
        while not node.is_root():
            node = node.parent
            if self.check_all_children_solved(node):
                node.solved = True
            else:
                break
            
    def check_all_children_solved(self, node):
        if len(node.children) == self.branching_factor and all(child.solved for child in node.children):
            assert len(set([child.data["a"] for child in node.children])) == self.branching_factor
            return True
        return False
