
import numpy as np
import tensorflow as tf
from .utils import load_pickle, save_pickle
from .session_manager import variable_name_scope
from .settings import settings

"""
Initializer used by deepmind in DQN (torch, lua) implementation.
Torch uses this initialization for all weights and biases, both for convolutional and fully connected layers
We need to account for the factor in:
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/layers/python/layers/initializers.py#L142
Source:
https://github.com/torch/nn/blob/7762e143lstm_celld86e1664a2675065420d57a7a4195d07/SpatialConvolution.lua#L38
"""
def torch_initializer():
    return tf.contrib.layers.variance_scaling_initializer(factor=1/3, mode = "FAN_IN", uniform=True)

def predefined_values_initializer(param_values):
    def _initializer(shape, dtype=None, partition_info=None):
        assert tuple(shape) == tuple(param_values.shape), "Expected shape: " + str(shape) + ".  Given shape: " + str(param_values.shape)
        #TODO: check dtype, maybe convert param_values?
        return param_values
    return _initializer

class NeuralNetwork:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

        initializers = None
        if "params_file" in self.kwargs and self.kwargs["params_file"] is not None:
            param_values = load_pickle(self.kwargs["params_file"])
            initializers = iter([predefined_values_initializer(pv) for pv in param_values])
            
        with variable_name_scope(self.name):
            self.scope = tf.get_variable_scope()
            self.build(initializers)
            self._ph_assign_op() #to be able to set parameter values
    
    def build(self, params_iterator=None):
        self.layers, self.inputs, self.hidden, self.outputs = self._build(params_iterator)
        assert all([x in self.layers.keys() for x in (self.inputs+self.hidden+self.outputs)])
            
    def get_params(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope.name)

    @property
    def size(self):
        n_params = 0
        for param in self.get_params():
            n_params += np.prod([shape.value for shape in param.shape])
        return n_params

    def get_assign_op(self, source_net):
        assign_ops = list()
        for var, source_var in zip(self.get_params(), source_net.get_params()):
            assign_ops.append(var.assign(source_var))
        return tf.group(*assign_ops)

    def get_param_values(self, session):
        return session.run(self.get_params())
    
    def set_param_values(self, session, param_values):
        feed_dict = dict(zip(self.param_placeholders, param_values))
        session.run(self.assign_ph_op, feed_dict=feed_dict)

    def deepcopy(self, name=None):
        if name is None:
            name = self.name
        return type(self)(name, **self.kwargs)

    def save(self, session, filename=None):
        if filename is None: filename = self.name + ".pkl"
        param_values = self.get_param_values(session)
        save_pickle(filename, param_values)
    
    def load(self, session, filename=None):
        if filename is None: filename = self.name + ".pkl"
        param_values = load_pickle(filename)
        self.set_param_values(session, param_values)

    def _ph_assign_op(self):
        assign_ops = list()
        self.param_placeholders = list()
        with variable_name_scope("param_placeholders"): #although there are no variables here, we use variable_name_scope to be consistent
            for var in self.get_params():
                name = var.op.name.split('/')[-1] + "_ph"
                param_ph = tf.placeholder(dtype=var.dtype, shape=var.shape, name=name)
                assign_ops.append(var.assign(param_ph))
                self.param_placeholders.append(param_ph)
        self.assign_ph_op = tf.group(*assign_ops)

    def _build(self, params_iterator=None):
        """
        if params_iterator is not None, those params will be used to initialize the network

        :return: layers: dict {layer_name: layer}
                 input_layers: list of layer names (strings)
                 hidden_layers: list of layer names
                 output_layers: list of layer names
        """
        raise NotImplementedError()


class Mnih2013(NeuralNetwork):
    def _build(self, params_iterator=None):
        network_input = tf.placeholder(tf.float32, shape=self.kwargs['input_shape'], name="input")
        hidden_layers = self._encoder(network_input, params_iterator)
        layers = dict(hidden_layers)
        layers["input"] = network_input
        hidden = [n for n, _ in hidden_layers]

        policy_logits_layer = self._policy_head(layers[hidden[-1]], params_iterator)  # policy logits or action values
        layers["policy_head"] = policy_logits_layer
        outputs = ["policy_head"]

        if self.kwargs['add_value_head']:
            value_layer = self._value_head(layers[hidden[-1]], params_iterator)
            layers["value_head"] = value_layer
            outputs.append("value_head")

        return layers, ["input"], hidden, outputs

    def _encoder(self, input_tensor, params_iterator=None):
        """
        :return: hidden_layers: list of tuples (layer_name, layer)
        """
        if params_iterator is None:
            params_iterator = iter([torch_initializer() for _ in range(6)])

        conv1 = tf.contrib.layers.conv2d(inputs=input_tensor,  # (batch_size, image_width, image_height, channels)
                                         num_outputs=16,
                                         kernel_size=8,
                                         stride=4,
                                         padding="VALID",
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=next(params_iterator),
                                         biases_initializer=next(params_iterator),
                                         scope="conv1")
        conv2 = tf.contrib.layers.conv2d(inputs=conv1,
                                         num_outputs=32,
                                         kernel_size=4,
                                         stride=2,
                                         padding="VALID",
                                         activation_fn=tf.nn.relu,
                                         weights_initializer=next(params_iterator),
                                         biases_initializer=next(params_iterator),
                                         scope="conv2")
        flat = tf.contrib.layers.flatten(conv2)
        fc3 = tf.contrib.layers.fully_connected(inputs=flat,
                                                num_outputs=settings["Mnih2013_num_outputs"],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=next(params_iterator),
                                                biases_initializer=next(params_iterator),
                                                scope="fc3")
        return [("conv1", conv1),
                ("conv2", conv2),
                ("flat2", flat),
                ("fc3", fc3)]

    def _policy_head(self, input_tensor, params_iterator=None):
        if params_iterator is None:
            params_iterator = iter([torch_initializer() for _ in range(2)])
        return tf.contrib.layers.fully_connected(inputs=input_tensor,
                                                 num_outputs=self.kwargs['num_outputs'],
                                                 weights_initializer=next(params_iterator),
                                                 biases_initializer=next(params_iterator),
                                                 activation_fn=None,
                                                 scope="policy_head")

    def _value_head(self, input_tensor, params_iterator=None):
        if params_iterator is None:
            params_iterator = iter([torch_initializer() for _ in range(2)])
        value = tf.contrib.layers.fully_connected(inputs=input_tensor,
                                                  num_outputs=1,
                                                  weights_initializer=next(params_iterator),
                                                  biases_initializer=next(params_iterator),
                                                  activation_fn=None,
                                                  scope="value_head")
        return tf.reshape(value, (-1,))  # matrix to vector
