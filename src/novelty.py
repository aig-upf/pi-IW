import numpy as np
from collections import defaultdict
from itertools import combinations

def set_insertion(s, elem):
    """
    Try to add the given element into the given set and return whether the
    element was actually inserted, i.e. because it was not in the set before.
    """
    l = len(s)
    s.add(elem)
    return len(s) != l

class Novelty1Table():
    def __init__(self):
        self.table = set()

    def check(self, atoms):
        is_novel = any(atom for atom in atoms if atom not in self.table)
        return 1 if is_novel else np.inf

    def check_and_update(self, atoms):
        is_novel = False
        for atom in atoms:
            is_novel = set_insertion(self.table, atom) or is_novel
        return 1 if is_novel else np.inf

class NoveltyTable:
    def __init__(self, max_width):
        self.max_width = max_width

        # We'll have one novelty table for each width value; for instance, tables[2] will contain all
        # tuples of size 2 that have been seen in the search so far.
        self.tables = defaultdict(Novelty1Table)
        
    def check(self, atoms):
        novelty = np.inf
        # Iterate for each value of k, and process all tuples of size k to check for novel ones.
        # NOTE that even if we find that a state has novelty e.g. 1, we still iterate through all tuples
        # of larger sizes so that they can be recorded in the novelty tables.
        for k in range(1, self.max_width + 1):
            if self.tables[k].check(combinations(atoms, k)):
                novelty = min(novelty, k)
        return novelty
        
    def check_and_update(self, atoms):
        """
        Evaluates the novelty of a state up to the pre-set max-width.
        """
        novelty = np.inf
        for k in range(1, self.max_width + 1):
            if self.tables[k].check_and_update(combinations(atoms, k)):
                novelty = min(novelty, k)
        return novelty
    
class RolloutNovelty1Table():
    def __init__(self, add_cached_nodes_to_novelty_table):
        self.atom_depth = defaultdict(lambda : np.inf)
        self.ignore_cached_nodes = not add_cached_nodes_to_novelty_table #only features from new nodes in the tree will be added to the novelty table
        
    def check(self, atoms, depth, node_is_new):
        for atom in atoms:
            if depth < self.atom_depth[atom] or (not node_is_new and depth == self.atom_depth[atom]):
                return 1 #at least one atom is either case 1 or 4
        return np.inf #all atoms are either case 2 or 3
        
    def check_and_update(self, atoms, depth, node_is_new):
        is_novel = False
        for atom in atoms:
            if depth < self.atom_depth[atom]:
                if self.ignore_cached_nodes:
                    if node_is_new:
                        #here node_is_new controls that existing nodes (already in the tree) are not added to the table (and not pruned)
                        self.atom_depth[atom] = depth
                else:
                    #all nodes
                    self.atom_depth[atom] = depth
                is_novel = True #case 1, novel
            #else if node_is_new, case 2, not novel
            elif not node_is_new and depth == self.atom_depth[atom]:
                is_novel = True # case 4, was novel before and is still novel
            #else, case 3, was novel before but not anymore
        return 1 if is_novel else np.inf