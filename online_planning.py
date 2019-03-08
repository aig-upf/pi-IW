from utils import sample_pmf, softmax
import numpy as np


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
    from plan_step import gridenvs_BASIC_features
    import gridenvs.examples #load simple envs

    env_id = "GE_PathKeyDoor-v0"
    max_tree_nodes = 30
    discount_factor = 0.99
    cache_subtree = True

    env = gym.make(env_id)
    actor = TreeActor(env, observe_fn=gridenvs_BASIC_features)
    planner = RolloutIW(branching_factor=env.action_space.n)

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
        print("Action: %i. Reward: %.1f" % (current_root_data["a"], current_root_data["r"])) # obs is in prev_root_data
        steps_cnt += 1

    print("It took %i steps" % steps_cnt)
