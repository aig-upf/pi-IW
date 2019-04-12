import logging
from .session_manager import session_manager

logger = logging.getLogger(__name__)


_registry = {key: dict() for key in ['Algorithms', 'Planners', 'Features', 'Lookaheads', 'Networks']}
def register(registry_name, obj_name, obj):
    global _registry
    assert registry_name in _registry.keys(), "Wrong registry name. Try 'Algorithms', 'Planners', 'Features', ..."
    assert obj_name not in _registry[registry_name].keys(), "%s already in registry %s" %(obj_name, registry_name)
    _registry[registry_name][obj_name] = obj

def register_class(registry_name, class_obj):
    register(registry_name, class_obj.__name__, class_obj)

def get_from_registry(registry_name, obj_name):
    global _registry
    assert registry_name in _registry.keys(), "Wrong registry name. Try 'Algorithms', 'Planners', 'Features', ..."
    assert obj_name in _registry[registry_name].keys(), "%s not present in registry %s: %s" %(obj_name, registry_name, str(_registry[registry_name].keys()))
    return _registry[registry_name][obj_name]

from .planners import *
from .lookahead import *
from .features import *
from .algorithms import SupervisedPolicy, SupervisedPolicyValue
from .networks import Mnih2013

register_class("Algorithms", SupervisedPolicy)
register_class("Algorithms", SupervisedPolicyValue)
register_class("Planners", BFS)
register_class("Planners", Random)
register_class("Planners", IW)
register_class("Planners", RolloutIW)
register_class("Planners", OriginalRolloutIW)
register_class("Planners", PolicyGuidedIW)
register_class("Planners", MCTSAlphaZero)

register_class("Lookaheads", LookaheadReturns)
register_class("Lookaheads", LookaheadCounts)

register_class("Features", SimpleBASIC)
register_class("Features", NNLastHiddenBins)
register_class("Features", NNLastHiddenBool)
register_class("Features", NNLastHiddenFloat)
register_class("Features", AtariRAMFeatures)
register_class("Features", NNLastHiddenAdaptativeNBins)

register_class("Networks", Mnih2013)


#Helper methods to create planner and lookahead instances. We can use

def create_nn_as(name, obs_shape, n_actions, params_filename):
    import os.path
    from .utils import load_pickle
    settings_filename = os.path.join("/".join(params_filename.split('/')[:-1]), "args.pkl")
    aux_settings = load_pickle(settings_filename)

    algorithm_class = get_from_registry('Algorithms', aux_settings["algorithm"])
    network_class  = get_from_registry("Networks", aux_settings["network_architecture"])

    nn = network_class(name=name,
                       params_file=params_filename,
                       input_shape=(None,) + obs_shape,
                       num_outputs=n_actions,
                       add_value_head=algorithm_class.USES_VALUE)
    return nn


def get_instance_with_nn(cls, obs_shape, n_actions, network_or_filename, kwargs={}):
    if cls.REQUIRES_NN:
        assert network_or_filename is not None, "%s requires a NN. Provide a network instance or a filename to load it from."%cls.__name__
        from src.networks import NeuralNetwork
        if isinstance(network_or_filename, NeuralNetwork):
            neural_net = network_or_filename
        elif type(network_or_filename) is str:
            with session_manager.worker():
                neural_net = create_nn_as("%s_net"%cls.__name__, obs_shape, n_actions, network_or_filename)
        else:
            raise Exception("Bad network_or_filename input. %s requires a NN. Provide a network instance or a filename to load it from."%cls.__name__)
        return cls(neural_net=neural_net, **kwargs)
    return cls(**kwargs)


def get_planner(actor, obs_shape, n_actions, planner_class, features_class=None, NN_planner=None, NN_features=None):
    from src.tree_actor import TreeActor
    feature_extractor = None
    assert type(actor) is TreeActor
    if type(planner_class) is str:
        planner_class = get_from_registry('Planners', planner_class)
    planner = get_instance_with_nn(planner_class, obs_shape, n_actions, NN_planner, {"actor": actor, "n_actions": n_actions})

    if planner_class.REQUIRES_FEATURES:
        assert features_class is not None, planner_class + " requires features. Please specify a feature extractor."
        if type(features_class) is str:
            features_class = get_from_registry('Features', features_class)
        feature_extractor = get_instance_with_nn(features_class, obs_shape, n_actions, NN_features, {"actor": actor})
    return planner, feature_extractor


def get_lookahead(actor, obs_shape, n_actions, lookahead_class, planner_class, features_class=None, NN_planner=None, NN_features=None):
    planner, feature_extractor = get_planner(actor, obs_shape, n_actions, planner_class, features_class, NN_planner, NN_features)
    if type(lookahead_class) is str:
        lookahead_class = get_from_registry('Lookaheads', lookahead_class)
    nn = planner.nn if planner.REQUIRES_NN else NN_planner
    return get_instance_with_nn(lookahead_class, obs_shape, n_actions, nn, {"planner": planner}), feature_extractor

