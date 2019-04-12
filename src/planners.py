import numpy as np
import logging
from .data_structures import Queue
from .novelty import Novelty1Table, NoveltyTable, RolloutNovelty1Table
from .utils import softmax, sample_pmf
from .settings import settings

logger = logging.getLogger(__name__)

class Planner:
    REQUIRES_NN = False
    REQUIRES_FEATURES = False

    def __init__(self, actor, n_actions):
        self.actor = actor
        self.n_actions = n_actions
        self.within_budget, self.reset_budget = self._budget_functions()
        
    def _budget_functions(self):
        assert settings["tree_budget_type"] in ('interactions', 'seconds')
        if settings["tree_budget_type"] == 'interactions':
            assert settings["tree_budget"] == int(settings["tree_budget"]), "Tree budget must be an integer when specifying the number of interactions"
            
        if settings["tree_budget_type"] == 'interactions':
            def reset_budget_interactions():
                self.step_cnt = 0
            def within_budget_interactions():
                self.step_cnt += 1
                return self.step_cnt < settings["tree_budget"] and (settings["max_tree_size"] is None or len(self.actor.tree) < settings["max_tree_size"])
            return within_budget_interactions, reset_budget_interactions
        else:
            import timeit
            def within_budget_secs():
                return timeit.default_timer() - self.start < settings["tree_budget"] and (settings["max_tree_size"] is None or len(self.actor.tree) < settings["max_tree_size"])
            def reset_budget_secs():
                self.start = timeit.default_timer()
            return within_budget_secs, reset_budget_secs
        
    def expand(self, node, a):
        child = self.actor.expand(node, a)
        return child, self.within_budget()
        
    def plan(self):
        self.reset_budget()
        len_before = len(self.actor.tree)
        self._plan()
        # return amount of expanded nodes
        return len(self.actor.tree) - len_before
        
    def _plan(self):
        raise NotImplementedError()


def actions_already_used(node):
    actions = list()
    for child in node.children:
        actions.append(child.data["a"])
    return actions

class Random(Planner):
    def _plan(self):
        #Add nodes of the tree to the queue
        actions = set(range(self.actor.env.action_space.n))
        nodes = Queue()
        for node in self.actor.tree.iter_breadth_first(include_root=True, include_leaves=True):
            not_used_actions = actions - set(actions_already_used(node))
            if len(not_used_actions) >= 0 and not node.data["done"]:
                nodes.push_many(zip([node]*len(not_used_actions), not_used_actions))
        
        #Expand nodes randomly
        within_budget = True
        while len(nodes) != 0 and within_budget:
            node, a = nodes.pop_random()
            child, within_budget = self.expand(node, a)
            if not child.data["done"]:
                nodes.push_many(zip([child]*len(actions), actions))

class BFS(Planner):
    def _plan(self):
        #Add nodes of the tree to the queue
        actions = list(range(self.actor.env.action_space.n))
        queue = Queue()
        for node in self.actor.tree.iter_breadth_first(include_root=True, include_leaves=True):
            if node.is_leaf() and not node.data["done"]:
                queue.push(node)
        
        #Expand nodes
        within_budget = True
        while len(queue) != 0 and within_budget:
            node = queue.pop()
            np.random.shuffle(actions)
            for a in actions:
                child, within_budget = self.expand(node, a)
                if not child.data["done"]:
                    queue.push(child)
                if not within_budget:
                    break


class IW(Planner):
    REQUIRES_FEATURES = True

    def _plan(self):
        assert settings["iw_reset_table"] == "transition", "Other options are not implemented."
        if settings["iw_max_novelty"] == 1:
            novelty_table = Novelty1Table()
        else:
            novelty_table = NoveltyTable(settings["iw_max_novelty"])

        # Add nodes of the tree to the queue
        actions = list(range(self.actor.env.action_space.n))
        queue = Queue()
        if settings["iw_consider_cached_nodes_novelty"]:
            # add cached nodes to novelty table, and maybe prune them
            for node in self.actor.tree.iter_breadth_first(include_root=True, include_leaves=True):
                if not node.data["done"]:
                    novelty = novelty_table.check_and_update(node.data["features"])
                    if node.is_leaf() and novelty <= settings["iw_max_novelty"]:
                        queue.push(node)
        else:
            # do not add cached nodes to novelty table, thus no pruning
            for node in self.actor.tree.iter_breadth_first(include_root=True, include_leaves=True):
                if node.is_leaf() and not node.data["done"]:
                    queue.push(node)

        # Expand nodes
        within_budget = True
        while len(queue) != 0 and within_budget:
            node = queue.pop()
            np.random.shuffle(actions)
            for a in actions:
                child, within_budget = self.expand(node, a)
                if not child.data["done"]:
                    novelty = novelty_table.check_and_update(child.data["features"])
                    if novelty <= settings["iw_max_novelty"]:
                        queue.push(child)
                if not within_budget:
                    break


class MCTSAlphaZero(Planner):
    """
    Implementation of Monte Carlo Tree Search as in AlphaZero.
    Expands nodes depending on the following metric:
        U(s,a) = c * P(s,a) * (sqrt(sum_b( N(s,b) )) / ( 1+N(s,a) ))
    """

    REQUIRES_NN = True

    def __init__(self, actor, n_actions, neural_net):
        Planner.__init__(self, actor, n_actions)
        self.nn = neural_net
        self.actor.request_nn_output(self.nn, "policy_head")
        self.actor.request_nn_output(self.nn, "value_head")

    def _plan(self):
        self._initialize()

        within_budget = True
        while within_budget:
            # Select
            node, a = self._select(self.actor.tree.root)

            # Expand and evaluate
            if node.data["done"]:
                assert a is None
                child = node  # do not expand! But back up -> count should increase
                within_budget = self.within_budget()
            else:
                child, within_budget = self.expand(node, a)
                self._init_counters(child)

            # Backup
            self._backup(child)

    def _initialize(self):
        if not "N" in self.actor.tree.root.data.keys():
            # We just created a new tree (new episode started)
            assert len(self.actor.tree) == 1
            self._init_counters(self.actor.tree.root)
            self.actor.tree.root.data["probs"] = softmax(self.actor.tree.root.data[self.nn]["policy_head"])

    def _select(self, node):
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

    def _backup(self, node):
        if not node.data["done"]:
            node.data["probs"] = softmax(node.data[self.nn]["policy_head"])
        R = node.data["r"] if node.data["done"] else node.data["r"] + settings["discount_factor"] * \
                                                     node.data[self.nn]["value_head"]
        while True:
            a = node.data["a"]
            node = node.parent
            node.data["N"][a] += 1
            node.data["W"][
                a] += R  # we consider here we are interacting with an environment, but we should multiply R by 1/-1 (player1, player2) if doing self play!

            if node.is_root():
                return
            else:
                R = node.data["r"] + settings["discount_factor"] * R

    def compute_U(self, node, add_noise):
        prior = node.data["probs"]
        if add_noise:
            noise = np.random.dirichlet([settings["alphazero_noise_alpha"]] * self.actor.env.action_space.n)
            prior = (1 - settings["alphazero_noise_eps"]) * prior + settings["alphazero_noise_eps"] * noise

        N = node.data["N"] + 1
        sqrt_sum_counts = np.sqrt(np.sum(N))
        return settings["alphazero_puct_factor"] * prior * sqrt_sum_counts / (N + 1)

    def _init_counters(self, node):
        if not node.data["done"]:
            node.data["W"] = np.zeros((self.actor.env.action_space.n,))
            node.data["N"] = np.zeros((self.actor.env.action_space.n,))

    def _get_policy(self, node):
        U = self.compute_U(node, node.is_root())  # we'll add noise to the root node
        Q = (node.data["W"] + node.data[self.nn]["value_head"]) / (node.data["N"] + 1)
        return softmax(Q + U, temp=0)



class OriginalRolloutIW(Planner):
    """
    Rollout-IW as in https://arxiv.org/abs/1801.03354.
    The only difference should be that the four different cases explained in
    the paper are covered in the novelty table class.
    """
    REQUIRES_FEATURES = True

    def _plan(self):
        assert settings["iw_reset_table"] == "transition", "Other options are not implemented."
        assert settings[
                   "iw_max_novelty"] == 1, "General RolloutNoveltyTable not implemented, max novelty must be 1."
        novelty_table = RolloutNovelty1Table(settings["iw_consider_cached_nodes_novelty"])

        # Reset solved
        if settings["iw_consider_cached_nodes_novelty"]:
            # add cached nodes to novelty table, and maybe prune them
            for node in self.actor.tree.iter_breadth_first(include_root=True, include_leaves=True):
                node.solved = False
                if node.data["done"]:
                    if settings["iw_consider_terminal_nodes_novelty"] and settings[
                        "iw_consider_cached_nodes_novelty"]:
                        # We add features of terminal nodes to novelty table (Blai's implementation doesn't do that!)
                        novelty = self.novelty_table.check_and_update(node.data["features"], node.depth,
                                                                      node_is_new=True)
                    self.solve_and_propagate_label(node)
                else:
                    if settings["iw_consider_cached_nodes_novelty"]:
                        novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, True)
                        if novelty > settings["iw_max_novelty"]:
                            self.solve_and_propagate_label(node)

        # Expand nodes
        within_budget = True
        while within_budget and not self.actor.tree.root.solved:
            within_budget = self._rollout(novelty_table)

    def _rollout(self, novelty_table):
        new_in_tree = False  # flag to indicate if we have to restore the environment state. Once we start generating new nodes (new in the tree) we won't need to restore for the whole rollout. It will also determine if a node is novel or not.
        node = self.actor.tree.root
        within_budget = True
        while within_budget:
            node, new_in_tree, within_budget = self._pick_random_unsolved_child(node, new_in_tree)
            if node.data["done"]:
                # We don't add this node's features to the novelty table, because Blai's implementation doesn't add them...
                self.solve_and_propagate_label(node)
                break
            else:
                novelty = novelty_table.check_and_update(node.data["features"], node.depth, new_in_tree)
                if novelty > settings["iw_max_novelty"]:
                    self.solve_and_propagate_label(node)
                    break
        return within_budget

    def _pick_random_unsolved_child(self, node, last_node_is_new):
        if node.is_leaf():
            a = self.actor.env.action_space.sample()  # random action
        else:
            assert not last_node_is_new
            solved_actions = [child.data["a"] for child in node.children if child.solved]
            unsolved_actions = list(set(range(self.actor.env.action_space.n)) - set(solved_actions))
            assert len(unsolved_actions) > 0
            a = np.random.choice(unsolved_actions)

            # Already in tree?
            for child in node.children:
                if child.data["a"] == a:
                    return child, False, True  # node not new, let's assume we are still within budget (traversing the tree should be quick)

        # Generate node
        child, within_budget = self.expand(node, a)
        child.solved = False
        return child, True, within_budget

    def solve_and_propagate_label(self, node):
        node.solved = True
        while True:
            node = node.parent
            if self.check_all_children_solved(node):
                node.solved = True
                if node.is_root():
                    break
            else:
                break

    def check_all_children_solved(self, node):
        if len(node.children) == self.actor.env.action_space.n and all(child.solved for child in node.children):
            assert set([child.data["a"] for child in node.children]) == set(range(self.actor.env.action_space.n))
            return True
        return False


class RolloutIW(Planner):
    """
    Another implementation of RolloutIW. Instead of "picking" a node that can
    be either generated or already in the tree, we traverse the tree in a
    select function, similar to MCTS, and then take a rollout from the selected
    node. It may solve nodes while traversing the tree, causing it to call the
    select function several times before executing a rollout.
    It is equivalent to OriginalRolloutIW.
    """
    REQUIRES_FEATURES = True

    def __init__(self, actor, n_actions):
        Planner.__init__(self, actor, n_actions)
        assert settings["iw_reset_table"] == "transition", "Other options are not implemented."
        assert settings[
                   "iw_max_novelty"] == 1, "General RolloutNoveltyTable not implemented, max novelty must be 1."

    def _plan(self):
        self.novelty_table = RolloutNovelty1Table(settings["iw_consider_cached_nodes_novelty"])
        self._initialize()

        within_budget = True
        while within_budget and not self.actor.tree.root.solved:
            # Select
            node, a = self.select(self.actor.tree.root)
            if a is not None:
                within_budget = self.rollout(node, a)

    def _initialize(self):
        # Set solved
        for node in self.actor.tree.iter_breadth_first(include_root=True, include_leaves=True):
            node.solved = False
            if node.data["done"]:
                if settings["iw_consider_terminal_nodes_novelty"] and settings[
                    "iw_consider_cached_nodes_novelty"]:
                    # We add features of terminal nodes to novelty table (Blai's implementation doesn't do that!)
                    novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=True)
                self.solve_and_propagate_label(node)
            else:
                if settings["iw_consider_cached_nodes_novelty"]:
                    # add cached nodes to novelty table, and maybe prune them
                    novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, True)
                    if novelty > settings["iw_max_novelty"]:
                        self.solve_and_propagate_label(node)

    def _get_policy(self, node):
        # Uniform policy
        p = np.empty((self.actor.env.action_space.n,))
        p.fill(1 / self.actor.env.action_space.n)
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
            assert not node.solved and not node.data["done"], "Solved: %s.  Done: %s.  Depth: %s" % (
            str(node.solved), str(node.data["done"]), str(node.depth))
            novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=False)
            if novelty > settings["iw_max_novelty"]:
                self.solve_and_propagate_label(node)
                return None, None  # Prune node

            a, child = self.select_action_following_policy(node)
            assert child is None or (
                        not child.solved and not child.data["done"]), "Solved: %s.  Done: %s.  Depth: %s" % (
            str(child.solved), str(child.data["done"]), str(child.depth))

            if a is None:
                return None, None  # All actions recommended by the policy have been already solved for this node
            else:
                assert a < self.actor.env.action_space.n, "Ilegal action recommended by the policy (action a=%i >= action_space.n=%i)." % (
                a, self.n_actions)
                if child is None:
                    return node, a  # Action a needs to be expanded for this node
                else:
                    node = child  # Continue traversing the tree

    def select_action_following_policy(self, node):
        policy = self._get_policy(node)
        if node.is_leaf():
            # return action to expand
            assert not node.solved and not node.data["done"], "Solved: %s.  Done: %s.  Depth: %s" % (
            str(node.solved), str(node.data["done"]), str(node.depth))
            return sample_pmf(policy), None

        node_children = [None] * self.actor.env.action_space.n
        available_actions = (policy > 0)
        for child in node.children:
            node_children[child.data["a"]] = child
            # available_actions[child.data["a"]] = not child.solved
            if child.solved:
                available_actions[child.data["a"]] = False

        # Take out actions that have been solved
        p = (policy * available_actions)
        ps = p.sum()

        # No actions available?
        if ps <= settings["RIW_min_prob"]:
            # All actions recommended by the policy (i.e. with prob >0) have been (or should be considered) solved. Solve node.
            # It is posible that some nodes in the subtree are not marked as solved. That's not a problem, and it's because the policy gives those branches low probability (less than iw_solved_prob)
            self.solve_and_propagate_label(node)
            return None, None

        # Select action not solved
        p = p / ps
        assert all((p > 0) == available_actions), "p: %s;  avail_a: %s;   ps:%s" % (
        str(p), str(available_actions), str(ps))
        a = sample_pmf(p)

        child = node_children[a]
        if child:
            assert not child.solved and not child.data[
                "done"], "a: %i, Solved: %s.  Done: %s.  Depth: %s.  policy: %s.  avail_actions: %s.  p: %s.  ps: %s.  children: %s." % (
            a, str(child.solved), str(child.data["done"]), str(child.depth), str(policy), str(available_actions),
            str(p), str(ps), str([(c.data["a"], c.solved) for c in node.children]))

        return a, child

    def rollout(self, node, a):
        within_budget = True
        while within_budget:
            node, within_budget = self.expand(node, a)
            node.solved = False

            if node.data["done"]:
                if settings["iw_consider_terminal_nodes_novelty"]:
                    # We add features of terminal nodes to novelty table (Blai's implementation doesn't do that!)
                    novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=True)
                self.solve_and_propagate_label(node)
                return within_budget
            else:
                novelty = self.novelty_table.check_and_update(node.data["features"], node.depth, node_is_new=True)
                if novelty > settings["iw_max_novelty"]:
                    self.solve_and_propagate_label(node)
                    return within_budget

            a, child = self.select_action_following_policy(node)
            assert a is not None and child is None, "Action: %s, child: %s" % (str(a), str(child))
            # if a is None:
            #     return within_budget
        assert not within_budget
        return within_budget  # budget exhausted

    def solve_and_propagate_label(self, node):
        node.solved = True
        while not node.is_root():
            node = node.parent
            if self.check_all_children_solved(node):
                node.solved = True
            else:
                break

    def check_all_children_solved(self, node):
        if len(node.children) == self.actor.env.action_space.n and all(child.solved for child in node.children):
            assert len(set([child.data["a"] for child in node.children])) == self.actor.env.action_space.n
            return True
        return False


class PolicyGuidedIW(RolloutIW):
    """
    RolloutIW with policy as heuristic for node selection (instead of random).
    """
    REQUIRES_NN = True

    def __init__(self, actor, n_actions, neural_net):
        Planner.__init__(self, actor, n_actions)
        self.nn = neural_net
        self.actor.request_nn_output(self.nn, "policy_head")

    def _get_policy(self, node):
        return softmax(node.data[self.nn]["policy_head"], temp=settings["policy_iw_temp"])
