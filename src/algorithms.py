import tensorflow as tf
from .utils import preprocess_obs_list
from .registry import get_from_registry
from .session_manager import session_manager, absolute_name_scope
from .settings import settings
from .utils import softmax, sample_pmf
import logging

logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, network, action_space):
        self.network = network
        self.action_space = action_space

    def get_action(self, node_data):
        return self._action(node_data[self.network]["policy_head"])

    def _action(self, network_output):
        raise NotImplementedError()

    def get_random_action(self):
        return self.action_space.sample()


class StochasticGreedyPolicy(Policy):
    def _action(self, network_output):
        return sample_pmf(softmax(network_output.squeeze(), temp=0))


class Algorithm:
    USES_VALUE = False

    def __init__(self, actor, obs_shape, n_actions):
        self.actor = actor

        network_type = get_from_registry("Networks", settings["network_architecture"])
        with tf.device("cpu:0"):  # let's put everything in the cpu!
            # global net
            if session_manager.execution_is_distributed:
                with session_manager.worker():
                    self.local_network = network_type(name="local_network",
                                                      params_file=settings["NN_params_from_pkl"],
                                                      input_shape=(None,) + obs_shape,
                                                      num_outputs=n_actions,
                                                      add_value_head=self.USES_VALUE)

                with session_manager.ps():
                    self.global_network = self.local_network.deepcopy("global_network")
                self.sync_local_network_op = self.local_network.get_assign_op(self.global_network)
            else:
                # let's speed up things (avoid having a copy of the network and update parameters)
                with session_manager.ps():
                    self.global_network = network_type(name="global_network",
                                                       params_file=settings["NN_params_from_pkl"],
                                                       input_shape=(None,) + obs_shape,
                                                       num_outputs=n_actions,
                                                       add_value_head=self.USES_VALUE)
                    self.local_network = self.global_network
                self.sync_local_network_op = tf.no_op()

            self.policy = self.POLICY_CLASS(self.local_network, self.actor.env.action_space)

            with session_manager.worker():
                self.loss, summaries = self._loss()

                # compute grads on local network
                grads = tf.gradients(self.loss, self.local_network.get_params())
                if settings["clip_grad_norm"]:
                    grads, global_norm = tf.clip_by_global_norm(grads, settings["clip_grad_norm"])
                else:
                    global_norm = tf.global_norm(grads)

                with absolute_name_scope(""):
                    summaries.append(tf.summary.scalar("global_grad_norm", global_norm))
                    summaries.append(tf.summary.scalar("total_loss", self.loss))
                self.summaries_op = tf.summary.merge(summaries)

        # apply grads to global network
        grads_and_vars = list(zip(grads, self.global_network.get_params()))
        self.train_op = session_manager.apply_grads(grads_and_vars)

        # we need some outputs after each step (probably to compute targets)
        self.request_nn_outputs(self.actor)

    def request_nn_outputs(self, actor):
        self.actor.request_nn_output(self.local_network, "policy_head")
        if self.USES_VALUE:
            self.actor.request_nn_output(self.local_network, "value_head")

    def accumulate_gradients(self, x):
        """
        Accumulate gradients from minibatch.
        Watch out: gradients from different workers will be averaged!
        """
        assert settings["gradient_accumulator"] in ("sum", "mean")
        if settings["gradient_accumulator"] == "sum":
            return tf.reduce_sum(x,
                                 axis=0)  # we use axis=0 so that an error is thrown when x has more than one dimension
        else:
            return tf.reduce_mean(x, axis=0)

    def get_transition(self, parent_data, child_data):
        raise NotImplementedError()

    def get_train_op(self, batch, feed_dict={}):
        raise NotImplementedError()

    def _loss(self):
        raise NotImplementedError()

    def update(self):
        session_manager.session.run(self.sync_local_network_op)

class SupervisedPolicy(Algorithm):
    POLICY_CLASS = StochasticGreedyPolicy
    USES_VALUE = False

    def _loss(self):
        self.target_policy = tf.placeholder(tf.float32, shape=(None, self.actor.env.action_space.n), name="target_policy")
        cross_entropy = self.accumulate_gradients(tf.nn.softmax_cross_entropy_with_logits(labels=self.target_policy, logits=self.policy.network.layers["policy_head"]))
        regularization = tf.reduce_sum([tf.nn.l2_loss(param) for param in self.policy.network.get_params()], axis=0)

        reg_loss = settings["loss_regularization_weight"]*regularization
        loss = cross_entropy + reg_loss

        summaries = list()
        with absolute_name_scope(""):
            summaries.append(tf.summary.scalar("policy_loss", cross_entropy))
            summaries.append(tf.summary.scalar("regularization_loss", reg_loss))
        return loss, summaries
    
    def get_train_op(self, batch, feed_dict={}):
        feed_dict.update({self.policy.network.layers["input"]: preprocess_obs_list(batch['obs']),
                          self.target_policy: batch['target_policy']})
        return self.train_op, feed_dict

    def get_transition(self, parent_data, child_data):
        return {'obs': parent_data["obs"], 'target_policy': parent_data["target_policy"]}


class SupervisedPolicyValue(SupervisedPolicy):
    USES_VALUE = True

    def _loss(self):
        loss, summaries = SupervisedPolicy._loss(self)

        self.returns = tf.placeholder(tf.float32, shape=(None,), name="returns")
        value_loss = self.accumulate_gradients(0.5 * tf.square(self.returns - self.policy.network.layers["value_head"]))
        value_loss = settings["loss_value_weight"] * value_loss
        loss += value_loss

        with absolute_name_scope(""):
            summaries.append(tf.summary.scalar("value_loss", value_loss))
        return loss, summaries

    def get_train_op(self, batch, feed_dict={}):
        _, feed_dict = SupervisedPolicy.get_train_op(self, batch, feed_dict)
        try:
            returns = batch["return"]
        except KeyError:
            raise Exception("Algorithm requires a target for the value (i.e. accumulated reward for the whole episode or an estimate)")
        feed_dict.update({self.returns: returns})
        return self.train_op, feed_dict
