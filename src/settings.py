from keyword import iskeyword
from .utils import ArgsManager


def is_valid_variable_name(name):
    return name.isidentifier() and not iskeyword(name)


class Param:
    def __init__(self, default, type_constructor, description, short_name=None, choices=None):
        self.default = default
        self.type_constructor = type_constructor
        self.description = description
        self.short_name = short_name
        self.choices = choices


class Parameters:
    def __init__(self):
        self._args = dict()

    def add(self, name, default, type_constructor, description, short_name=None, choices=None):
        assert name not in self._args
        assert is_valid_variable_name(name)
        self._args[name] = Param(default, type_constructor, description, short_name, choices)
        setattr(self, name, default)

    def get_defaults(self):
        return dict([(name, param.default) for name, param in self._args.items()])

    def parse(self, arg_values):
        #Add parameters to argparse
        arg_manager = ArgsManager()
        for arg_name, arg in self._args.items():
            arg_name_console = arg_name.replace('_', '-')
            if arg_values:
                default = arg_values[arg_name]
            else:
                default = arg.default
            arg_manager.add_arg(arg_name_console, default, arg.type_constructor, arg.description, arg.short_name, arg.choices)

        parsed_args = arg_manager.parse_args()
        return dict(parsed_args._get_kwargs())


# =============================================================================
# Default settings
# =============================================================================
def get_default_params():
    from datetime import datetime
    import numpy as np

    params = Parameters()

    seed = np.random.randint(0, 2 ** 30)  # it can be up to 2**32-1 but we leave a margin (we'll add task-id to this for every worker)
    params.add("seed", seed, int, "Random seed for reproducibility. Worker i will use seed+i.")

    #Environment
    params.add("env_id", 'PongWrapped-v0', str, "OpenAI Gym environment id", 'e')
    params.add("render", False, bool, "Render environment while training")
    params.add("frameskip", 15, int, "Amount of frames to skip at each action decision (just for atari games). Last action will be repated")

    #Agent
    params.add("clip_reward", True, bool, "Whether to clip rewards to [-1,1] or not")
    params.add("discount_factor", 0.99, float, "Discount factor")

    #Neural network
    params.add("network_architecture", 'Mnih2013', str, "Neural network architecture to be used.")
    params.add("NN_params_from_pkl", None, str, "Use existing network parameters instead of random initialization")
    params.add("Mnih2013_num_outputs", 256, int, "Number of outputs of the last hidden layer for architecture Mnih2013")

    #Learner
    params.add("summaries", 'chief', str, "Either only the chief worker computes summaries or all of them", choices=('chief', 'all'))
    params.add("logs", 'chief', str, "Either only the chief worker prints logs or all of them", choices=('chief', 'all'))
    params.add("algorithm", 'SupervisedPolicy', str, 'Algorithm to use (defines loss)', 'a')
    params.add("max_train_interactions", 1000000, int, 'Global interactions to be done by all the workers')
    params.add("save_eval_interactions", 10000, int, 'Save neural network parameters every x interactions')
    params.add("save_model", False, bool, "Whether to save the model or not")
    params.add("eval_episodes", 0, int, 'Evaluation episodes')
    params.add("eval_max_episode_transitions", 18000, int, 'Maximum steps per evaluation episode')
    params.add("local_batch_size", 32, int, "Maximum local steps to be done by the worker before updating the global neural network with the accumulated gradients")
    params.add("save_model_episodes", False, bool, "Save model after each episode")

    #Loss weights
    params.add("loss_regularization_weight", 0.001, float, "Regularization factor.")
    params.add("loss_value_weight", 1, float, "Value loss factor.")

    #Optimizer
    params.add("optimizer", 'rmsprop', str, 'Optimizer to use', choices=('rmsprop', 'adam'))
    params.add("learning_rate", 0.0005, float, "Learning rate", 'lr')
    params.add("learning_rate_annealing", 'constant', str, 'Learning rate annealing', choices=('linear', 'constant'))
    params.add("gradient_accumulator", 'mean', str, 'Accumulate gradients of a batch by adding them up or performing an average. The accumulated gradients of all workers will be averaged.', choices=('sum', 'mean'))
    params.add("clip_grad_norm", 40.0, float, "If not 'None', gradients will be clipped using GLOBAL norm")
    params.add("rmsprop_decay", 0.99, float, "RMSProp gradient running average factor. Gradient will be multiplied by 1-decay")
    params.add("rmsprop_momentum", 0.0, float, "RMSProp momentum")
    params.add("rmsprop_epsilon", 0.1, float, "RMSProp epsilon")
    params.add("rmsprop_centered", False, float, 'RMSProp as in "Generating Sequences With Recurrent Neural Networks", A. Graves 2013')

    #Tree algorithms
    params.add("lookahead", "LookaheadReturns", str, "Lookeahead class that will select nodes after each tree expansion in order to perform a step")
    params.add("planner", 'BFS', str, "Planner that will generate the tree")
    params.add("tree_budget", 50, float, "Maximum number of steps or seconds (see tree-budget-type) that the planner can take.")
    params.add("tree_budget_type", 'interactions', str, "Units of tree-budget.", choices=('seconds', 'interactions'))
    params.add("max_tree_size", None, int, "Planning algorithms will stop expanding nodes if there are max_tree_size nodes in the tree (counting cached nodes)")
    params.add("cache_subtree", True, bool, "Use already generated tree in subsequent steps")

    #IW
    params.add("iw_max_novelty", 1, int, "Max width.")
    params.add("iw_reset_table", "transition", str, "When to reset the novelty table. Choices are every step, when an episode ends or never (global)")
    params.add("iw_consider_cached_nodes_novelty", False, bool, "If True, nodes of the cached subtree will be added to the novelty table and the leaves of the tree that are not novel will be pruned")
    params.add("iw_consider_terminal_nodes_novelty", True, bool, "If True, features of terminal nodes will also be added to the novelty table")
    params.add("policy_iw_temp", 1, float, "Policy softmax temperature for expanding nodes.")
    params.add("RIW_min_prob", 0, float, "Probability at which an action is considered pruned.")

    #MTCS Alpha Zero
    params.add("alphazero_noise_alpha", 0.03, float, "Alpha for Dirichlet noise added to the prior probabilities of the root node")
    params.add("alphazero_noise_eps", 0.25, float, "Dirichlet noise weight")
    params.add("alphazero_puct_factor", 1, float, "Controls exploration")
    params.add("alphazero_target_policy_temp", 1, float, "Temperature for policy from counts")
    params.add("alphazero_firstmoves_temp", 30, int, "Moves with temperature = alphazero-policy-temp, then temperature will be 0 for the rest of the game.")

    #Features
    params.add("features", "NNLastHiddenBool", str, "How to extract features for computing the novelty")
    params.add("trainedNN_features", None, str, "Location of the already trained NN to use for feature extraction. It will search for a settings file in the same path to construct the network properly")
    params.add("trainedNN_planner", None, str, "Location of the already trained NN to guide the planner (used by Planner and Lookahead). It will search for a settings file in the same path to construct the network properly")

    #Dataset generation
    params.add("dataset_min_transitions", 100, int, "Minimum transitions that the dataset will contain.")
    params.add("dataset_max_transitions", 1000, int, "Maximum number of transitions to include in the dataset.")

    return params


settings = get_default_params().get_defaults()

def parse_args():
    global settings
    settings.update(get_default_params().parse(settings))
