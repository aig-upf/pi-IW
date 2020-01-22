import tensorflow as tf

class Mnih2013(tf.keras.models.Model):
    def __init__(self, num_logits, add_value=True):
        super(Mnih2013, self).__init__()
        self.add_value = add_value
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=8,
                                            strides=4,
                                            padding="VALID",
                                            activation=tf.nn.relu,
                                            name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=4,
                                            strides=2,
                                            padding="VALID",
                                            activation=tf.nn.relu,
                                            name="conv2")
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=256,
                                           activation=tf.nn.relu,
                                           name="fc3")
        self.hidden = [self.conv1, self.conv2, self.flat, self.dense]

        self.logits = tf.keras.layers.Dense(units=num_logits,
                                            activation=None,
                                            name="policy_head")
        if self.add_value:
            self.value = tf.keras.layers.Dense(units=1,
                                               activation=None,
                                               name="value_head")

    def call(self, x, output_features=False):
        for l in self.hidden:
            x = l(x)
        logits = self.logits(x)
        if self.add_value:
            value = tf.reshape(self.value(x), (-1,))  # remove last dimension
            if output_features:
                return logits, value, x
            else:
                return logits, value

        else:
            if output_features:
                return logits, x
            else:
                return logits


class Learner:
    def __init__(self, model, optimizer, grad_proc_fn=None, use_graph=False):
        self.model = model
        self.optimizer = optimizer
        self.grad_proc_fn = grad_proc_fn
        if use_graph:
            self.train_step = tf.function(self.train_step, autograph=False)

    def train(self, get_batch_fn, steps):
        for _ in range(steps):
            batch = get_batch_fn() # generates experience or gathers data from all actors
            self.train_step(*[tf.constant(tensor) for tensor in batch])

    def train_step(self, *args, **kwargs):
        with tf.GradientTape() as gtape: # grads = self.optimizer.compute_gradients(batch)
            loss, partial_losses = self.loss(*args, **kwargs)
        grads = gtape.gradient(loss, self.model.variables)
        if self.grad_proc_fn is not None:
            grads = self.grad_proc_fn(grads)
        self.optimizer.apply_gradients(zip(grads, self.model.variables))
        return loss, partial_losses

    def loss(self, *batch):
        raise NotImplementedError("To be implemented by subclasses")


def value_loss(values, returns, reduce_op=tf.reduce_mean):
    v = 0.5 * tf.square(returns - values)
    return reduce_op(v, axis=0)


def cross_entropy_loss(logits, target_policy, reduce_op=tf.reduce_mean):
    xent = tf.nn.softmax_cross_entropy_with_logits(labels=target_policy, logits=logits)
    return reduce_op(xent, axis=0)


def l2_regularization(variables, reduce_op=tf.reduce_sum):
    return reduce_op([tf.nn.l2_loss(param) for param in variables], axis=0)


class SupervisedPolicy(Learner):
    def __init__(self, model, optimizer, regularization_factor=0.001, use_graph=False):
        super(SupervisedPolicy, self).__init__(model, optimizer, use_graph=use_graph)
        self.regularization_factor = regularization_factor

    def loss(self, observations, target_policy):
        logits = self.model(observations)
        assert type(logits) is not tuple
        cross_entropy = cross_entropy_loss(logits, target_policy, reduce_op=tf.reduce_mean)
        regularization = l2_regularization(self.model.variables, reduce_op=tf.reduce_sum)  # TODO: use reduce mean here?
        total_loss = cross_entropy + self.regularization_factor*regularization
        return total_loss, [cross_entropy, regularization]


class SupervisedPolicyValue(Learner):
    def __init__(self, model, optimizer, value_factor=1, regularization_factor=0.0001, use_graph=False):
        super(SupervisedPolicyValue, self).__init__(model, optimizer, use_graph=use_graph)
        self.value_factor = value_factor
        self.regularization_factor = regularization_factor

    def loss(self, observations, target_policy, returns):
        logits, value = self.model(observations)
        cross_entropy = cross_entropy_loss(logits, target_policy, reduce_op=tf.reduce_mean)
        vloss = value_loss(value, returns, reduce_op=tf.reduce_mean)
        regularization = l2_regularization(self.model.variables, reduce_op=tf.reduce_sum)  # TODO: use reduce mean here?
        total_loss = cross_entropy + self.value_factor * vloss + self.regularization_factor*regularization
        return total_loss, [cross_entropy, vloss, regularization]
