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
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, forget_bias=1.0, input_size=None, var_on_cpu=True, wd=0.0):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: int, The dimensionality of the inputs into the LSTM cell,
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
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(name_scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = tf.split(1, 2, state)
            concat = linear([inputs, h], 4 * self._num_units, True, var_on_cpu=self.var_on_cpu, wd=self.wd)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_h, tf.concat(1, [new_c, new_h])
