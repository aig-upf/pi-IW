"""
Example of online planning with Rollout IW (without any learning or guidance) using the set of BASIC features in the
corridor example (the agent has to go pick the key, and undo the path to open a door). Note that this problem, with this
set of features, has width 2, and is therefore not solvable in one (off-line) planning step.
"""


import numpy as np
from utils import sample_pmf, softmax


def compute_return(tree, discount_factor):
    for node in tree.iter_breadth_first_reverse(include_root=False, include_leaves=True):
        if node.is_leaf():
            R = node.data["r"]
        else:
            R = node.data["r"] + discount_factor * np.max([child.data["R"] for child in node.children])
        node.data["R"] = R

def softmax_Q_tree_policy(tree, n_actions, discount_factor, temp=0):
    compute_return(tree, discount_factor)
    Q = np.empty(n_actions, dtype=np.float32)
    Q.fill(-np.inf)
    for child in tree.root.children:
        Q[child.data["a"]] = child.data["R"]
    return softmax(Q, temp=temp)


if __name__ == "__main__":
    import gym
    from rollout_iw import RolloutIW
    from tree import TreeActor
    from planning_step import gridenvs_BASIC_features
    import gridenvs.examples #load simple envs


    # HYPERPARAMETERS
    seed = 0
    env_id = "GE_PathKeyDoor-v0"
    max_tree_nodes = 30
    discount_factor = 0.99
    cache_subtree = True


    # Set random seed
    np.random.seed(seed)

    env = gym.make(env_id)
    actor = TreeActor(env, observe_fn=gridenvs_BASIC_features)
    planner = RolloutIW(branching_factor=env.action_space.n, ignore_cached_nodes=True)

    tree = actor.reset()
    episode_done = False
    steps_cnt = 0
    while not episode_done:
        planner.plan(tree=tree,
                     successor_fn=actor.generate_successor,
                     stop_condition_fn=lambda: len(tree) == max_tree_nodes)

        p = softmax_Q_tree_policy(tree, env.action_space.n, discount_factor, temp=0)
        a = sample_pmf(p)
        prev_root_data, current_root_data = actor.step(a, cache_subtree=cache_subtree)

        episode_done = current_root_data["done"]
        steps_cnt += 1
        print("\n".join([" ".join(row) for row in env.unwrapped.get_char_matrix(actor.tree.root.data["s"])]),
              "Action: ", current_root_data["a"], "Reward: ", current_root_data["r"],
              "Simulator steps:", actor.nodes_generated, "Planning steps:", steps_cnt, "\n")

    print("It took %i steps but the problem can be solved in 13." % steps_cnt)
