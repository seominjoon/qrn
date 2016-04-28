from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


def linear(args, output_size, bias, bias_start=0.0, scope=None, var_on_cpu=True, wd=0.0):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
      var_on_cpu: if True, put the variables on /cpu:0.

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    assert args
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        if var_on_cpu:
            with tf.device("/cpu:0"):
                matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        else:
            matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(matrix), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)


        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res

        if var_on_cpu:
            with tf.device("/cpu:0"):
                bias_term = tf.get_variable(
                    "Bias", [output_size],
                    initializer=tf.constant_initializer(bias_start))
        else:
            bias_term = tf.get_variable(
                "Bias", [output_size],
                initializer=tf.constant_initializer(bias_start))
    return res + bias_term


class BasicLSTMCell(RNNCell):
    """Basic GRU recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, input_size=None, var_on_cpu=True, wd=0.0):
        """Initialize the basic GRU cell.

        Args:
          num_units: int, The number of units in the GRU cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: int, The dimensionality of the inputs into the GRU cell,
            by default equal to num_units.
        """
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self._forget_bias = forget_bias
        self.var_on_cpu = var_on_cpu
        self.wd = wd

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, name_scope=None):
        """Long short-term memory cell (GRU)."""
        with tf.variable_scope(name_scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = tf.split(1, 2, state)
            concat = linear([inputs, h], 4 * self._num_units, True, var_on_cpu=self.var_on_cpu, wd=self.wd)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_h, tf.concat(1, [new_c, new_h])


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, input_size=None, var_on_cpu=True, wd=0.0):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size
        self.var_on_cpu = var_on_cpu
        self.wd = wd

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(1, 2, linear([inputs, state],
                                                    2 * self._num_units, True, 1.0))
                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear([inputs, r * state], self._num_units, True, var_on_cpu=self.var_on_cpu, wd=self.wd))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class CRUCell(RNNCell):
    """Combinatorial Recurrent Unit Implementation

    """
    def __init__(self, rel_size, arg_size, num_args, var_on_cpu=True, wd=0.0):
        self._rel_size = rel_size
        self._arg_size = arg_size
        self._num_args = num_args
        self._size = rel_size + arg_size * num_args
        self._cell = GRUCell(rel_size, var_on_cpu=var_on_cpu, wd=wd)

    def input_size(self):
        return self._size

    def output_size(self):
        return self._size

    def state_size(self):
        return self._size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.name_scope("Split"):
                N, _ = state.get_shape().as_list()
                R, A, C = self._rel_size, self._arg_size, self._num_args
                ru = tf.slice(state, [0, 0], [-1, R], name='ru')  # [N, d]
                au_flat = tf.slice(state, [0, R], [-1, -1], name='au_flat')
                au = tf.reshape(au_flat, [N, C, A], name='au')

                rf = tf.slice(inputs, [0, 0], [-1, R], name='rf')
                af_flat = tf.slice(inputs, [0, R], [-1, -1], name='af_flat')
                af = tf.reshape(af_flat, [N, C, A], name='af')

            with tf.variable_scope("Attention"):
                p_flat = tf.nn.softmax(linear([ru, rf], 2*C**2, True), name='p_flat')
                p = tf.reshape(p_flat, [N, C, 2*C])

            with tf.name_scope("Out"):
                ru_out, _ = self._cell(rf, ru)  # [N, R]
                a = tf.concat(1, [au, af], name='a')
                a_aug = tf.tile(tf.expand_dims(a, 1), [1, C, 1, 1], name='a_aug')
                au_out = tf.reduce_sum(a_aug * tf.expand_dims(p, -1), 2, name='au_out')  # [N, C, A]
                au_out_flat = tf.reshape(au_out, [N, C*A], name='au_out_flat')
                out = tf.concat(1, [ru_out, au_out_flat], name='out')  # [N, R+A*C]
        return out, out