from utils import sample_pmf
import numpy as np

def softmax(x, temp=1, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    if temp == 0:
        res = (x == np.max(x, axis=-1))
        return res/np.sum(res, axis=-1)
    x = x/temp
    e_x = np.exp( (x - np.max(x, axis=axis, keepdims=True)) ) #subtracting the max makes it more numerically stable, see http://cs231n.github.io/linear-classify/#softmax and https://stackoverflow.com/a/38250088/4121803
    return e_x / e_x.sum(axis=axis, keepdims=True)

def compute_return(tree, discount_factor):
    for node in tree.iter_breadth_first_reverse(include_root=False, include_leaves=True):
        if node.is_leaf():
            R = node.data["r"]
        else:
            R = node.data["r"] + discount_factor * np.max([child.data["R"] for child in node.children])
        node.data["R"] = R

def max_return_policy(tree, n_actions, discount_factor):
    compute_return(tree, discount_factor)
    Q = np.empty(n_actions, dtype=np.float32)
    Q.fill(-np.inf)
    for child in tree.root.children:
        Q[child.data["a"]] = child.data["R"]
    return softmax(Q, temp=0)


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


    # Instead of env.step() and env.reset(), we'll use TreeActor helper class, which creates a tree and adds nodes to it
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

        p = max_return_policy(tree, env.action_space.n, discount_factor)
        a = sample_pmf(p)
        prev_root_data, current_root_data = actor.step(a, cache_subtree=cache_subtree)

        episode_done = current_root_data["done"]
        print("Action: %i. Reward: %.1f" % (current_root_data["a"], current_root_data["r"])) # obs is in prev_root_data
        steps_cnt += 1

    print("It took %i steps" % steps_cnt)