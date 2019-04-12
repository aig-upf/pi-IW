
import numpy as np
from src.data_structures import Queue
import bisect
import math
from src.settings import settings

class FeatureExtractor:
    REQUIRES_NN = False

    def __init__(self, actor):
        self.actor = actor
        self.actor.add_observe_fn(self.compute)

    def feature_vector_to_atoms(self, feature_vector):
        """
        Receives a fixed-size vector of feature values and returns a list of
        (idx, value) tuples to make them (unique) atoms.
        """
        return list(enumerate(feature_vector))

    def get_feature_vector(self, env, node):
        raise NotImplementedError("Abstract class")

    def compute(self, env, node):
        node.data["features"] = self.feature_vector_to_atoms(self.get_feature_vector(env, node))

# =============================================================================
# Gridworld environments
# =============================================================================
class SimpleBASIC(FeatureExtractor):
    def get_feature_vector(self, env, node):
        return env.unwrapped.world.get_colors().flatten()

# =============================================================================
# Atari features
# =============================================================================
from src.atari_wrappers import get_atari_gym_name, atari_games
class AtariRAMFeatures(FeatureExtractor):
    def __init__(self, actor):
        super(AtariRAMFeatures, self).__init__(actor)
        assert not 'Simple' in settings["env_id"], "Feature type not suitable for this environment."
        assert any(get_atari_gym_name(game) in settings["env_id"] for game in
                   atari_games), "Feature type not suitable for this environment."

    def get_feature_vector(self, env, node):
        return env.unwrapped.ale.getRAM()

# =============================================================================
# Neural Network features
# =============================================================================
class NNLastHidden(FeatureExtractor):
    REQUIRES_NN = True
    def __init__(self, actor, neural_net):
        super(NNLastHidden, self).__init__(actor)
        self.nn = neural_net
        self.last_hidden_layer = self.nn.hidden[-1]
        self.actor.request_nn_output(self.nn, self.last_hidden_layer)

class NNLastHiddenBool(NNLastHidden):
    def get_feature_vector(self, env, node):
        return node.data[self.nn][self.last_hidden_layer].astype(np.bool)

class NNLastHiddenFloat(NNLastHidden):
    def get_feature_vector(self, env, node):
        return node.data[self.nn][self.last_hidden_layer]

class NNLastHiddenBins(NNLastHidden):
    def get_feature_vector(self, env, node):
        # The vector is discretized by computing the bin index corresponding to the each feature value such that: 0, (0,bin_size], (bin_size, 2*bin_size], ...
        # e.g. [2, 0.1, 5, 6, 6.001] with bin_size=2 -> [1, 1, 3, 3, 4]
        return np.ceil(node.data[self.nn][self.last_hidden_layer] / settings["NN_features_bin_size"]).astype(np.int64)


class NNLastHiddenAdaptativeNBins(NNLastHidden):
    def __init__(self, actor, neural_net):
        super(NNLastHiddenAdaptativeNBins, self).__init__(actor, neural_net)
        self._sliding_window = Queue()
        self._sorted_feature_values = None
        self._n = settings["N_adaptative_bins"] - 1 #-1 because first bin is always 0
        assert self._n >= 2
        self._p = 1 / self._n

    def get_feature_vector(self, env, node):
        last_hidden = node.data[self.nn][self.last_hidden_layer]
        self._update_sliding_window(last_hidden)
        quantiles = [self._quantile(i*self._p) for i in range(1, self._n)]
        return [np.argmax( f <= np.array([0, *qs, np.inf]) ) for f, qs in zip(last_hidden, zip(*quantiles))]

    def _update_sliding_window(self, last_hidden):
        if len(self._sliding_window) > settings["sliding_window_bins_len"]:
            x = self._sliding_window.pop()
            for i, f in enumerate(x):
                self._sorted_feature_values[i].remove(f)

        self._sliding_window.push(last_hidden)
        if self._sorted_feature_values is None:
            self._sorted_feature_values = [[f] for f in last_hidden]
        else:
            for i, f in enumerate(last_hidden):
                bisect.insort_left(self._sorted_feature_values[i], f)

    def _quantile(self, p):
        k = (len(self._sorted_feature_values[0]) - 1) * p #TODO: precompute?
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return [f_list[int(k)] for f_list in self._sorted_feature_values]
        return [(c-k)*f_list[int(f)] + (k-f)*f_list[int(c)] for f_list in self._sorted_feature_values]