from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.nn_ops import dropout
from tensorflow.python.ops.rnn_cell import RNNCell
from my.tensorflow.nn import linear


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
                state = tf.reshape(state, inputs.get_shape().as_list()[:-1] + state.get_shape().as_list()[-1:])  # explicit shape definition, to use my linaer function
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
        self.tensors = {}

    @property
    def input_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    @property
    def state_size(self):
        return self._size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__
        with tf.variable_scope(scope):
            tensors = self.tensors
            N, _ = state.get_shape().as_list()
            R, A, C = self._rel_size, self._arg_size, self._num_args
            with tf.name_scope("Split"):
                ru = tf.slice(state, [0, 0], [-1, R], name='ru')  # [N, d]
                au_flat = tf.slice(state, [0, R], [-1, -1], name='au_flat')
                au = tf.reshape(au_flat, [N, C, A], name='au')

                rf = tf.slice(inputs, [0, 0], [-1, R], name='rf')
                af_flat = tf.slice(inputs, [0, R], [-1, -1], name='af_flat')
                af = tf.reshape(af_flat, [N, C, A], name='af')

            with tf.variable_scope("Attention"):
                p_flat = tf.nn.softmax(linear([ru, rf], 2*C**2, True), name='p_flat')
                p = tf.reshape(p_flat, [N, C, 2*C], name='p')
                p_key = "{}/{}".format(scope, 'p')
                assert p_key not in tensors
                tensors[p_key] = p

            with tf.name_scope("Out"):
                ru_out, _ = self._cell(rf, ru)  # [N, R]
                a = tf.concat(1, [au, af], name='a')
                a_aug = tf.tile(tf.expand_dims(a, 1), [1, C, 1, 1], name='a_aug')
                au_out = tf.reduce_sum(a_aug * tf.expand_dims(p, -1), 2, name='au_out')  # [N, C, A]
                au_out_flat = tf.reshape(au_out, [N, C*A], name='au_out_flat')
                out = tf.concat(1, [ru_out, au_out_flat], name='out')  # [N, R+A*C]
        return out, out


class BiRNNCell(RNNCell):
    def set_forward(self):
        raise NotImplementedError()

    def set_backward(self):
        raise NotImplementedError()

    def reuse_variables(self):
        raise NotImplementedError()


class RSMCell(BiRNNCell):
    """
    Recurrent State Machine
    """
    def __init__(self, num_units, forget_bias=1.0, var_on_cpu=True, wd=0.0, initializer=None):
        self._num_units = num_units
        self._input_size = num_units * 3 + 2
        self._output_size = num_units * 3 + 2
        self._state_size = num_units * 2
        self._var_on_cpu = var_on_cpu
        self._wd = wd
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._reuse_flag = False
        self._is_forward = True

    def set_forward(self):
        self._is_forward = True

    def reuse_variables(self):
        self._reuse_flag = True

    def set_backward(self):
        self._is_forward = False

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "RSMCell"
            with tf.name_scope("Split"):  # Reset gate and update gate.
                inputs = tf.slice(inputs, [0, 2], [-1, -1])
                c, h = tf.split(1, 2, state)
                x, u, g = tf.split(1, 3, inputs)

            with tf.variable_scope("Shared", reuse=self._reuse_flag):
                a_raw = linear(u * x, 1, True, var_on_cpu=self._var_on_cpu,
                               initializer=self._initializer, scope='a_raw')
                a = tf.sigmoid(a_raw - self._forget_bias)  # [N, 1]
                # o = 1 - a * (1 - r)  # effective output gate

            with tf.variable_scope("Forward" if self._is_forward else "Backward"):
                r_raw = linear(u * x, 1, True, var_on_cpu=self._var_on_cpu,
                               initializer=self._initializer, scope='r_raw')
                r = tf.sigmoid(r_raw)  # [N, 1], reset gate

                c_t = tf.tanh(linear(inputs, self._num_units, True,
                                     var_on_cpu=self._var_on_cpu, wd=self._wd, scope='c_t_raw'), name='c_t')
                new_c = a * c_t + (1 - a) * c
                new_g = a * r * c_t
                new_h = new_g + (1 - a) * h

            with tf.name_scope("Concat"):
                new_state = tf.concat(1, [new_c, new_h])
                outputs = tf.concat(1, [x, new_h, new_g])
                outputs = tf.concat(1, [a, r, outputs])

        return outputs, new_state


class TempCell(RNNCell):
    """For task 3. Temporary"""
    def __init__(self, num_units, forget_bias=1.0, var_on_cpu=True, wd=0.0, initializer=None):
        self._num_units = num_units
        self._input_size = 2 * num_units + 1
        self._output_size = num_units
        self._state_size = 2 * num_units
        self._var_on_cpu = var_on_cpu
        self._wd = wd
        self._initializer = initializer
        self._forget_bias = forget_bias

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.name_scope("Split"):
                o = tf.slice(inputs, [0, 0], [-1, 1])
                x, u = tf.split(1, 2, tf.slice(inputs, [0, 1], [-1, -1]))
                c, h = tf.split(1, 2, state)

            with tf.variable_scope("Gate"):
                a_raw = linear(x * u, 1, True, var_on_cpu=self._var_on_cpu,
                               initializer=self._initializer, scope='a_raw')
                a = tf.sigmoid(a_raw - self._forget_bias)  # [N, 1]

            with tf.name_scope("Main"):
                new_c = o * x + (1 - o) * c
                new_h = a * c + (1 - a) * h
                new_state = tf.concat(1, [new_c, new_h])

        return new_h, new_state



class DropoutWrapper(RNNCell):
    """Operator adding dropout to inputs and outputs of the given cell."""

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None, is_train=None):
        """Create a cell with added input and/or output dropout.

        Dropout is never used on the state.

        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          input_keep_prob: unit Tensor or float between 0 and 1, input keep
            probability; if it is float and 1, no input dropout will be added.
          output_keep_prob: unit Tensor or float between 0 and 1, output keep
            probability; if it is float and 1, no output dropout will be added.
          seed: (optional) integer, the randomness seed.
          is_train: boolean tensor (often placeholder). If indicated, then when
            is_train is False, dropout is not applied.

        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if keep_prob is not between 0 and 1.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")
        if (isinstance(input_keep_prob, float) and
                not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                             % input_keep_prob)
        if (isinstance(output_keep_prob, float) and
                not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
            raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                             % output_keep_prob)
        self._cell = cell
        self._input_keep_prob = input_keep_prob
        self._output_keep_prob = output_keep_prob
        self._seed = seed
        self._is_train = is_train

    @property
    def input_size(self):
        return self._cell.input_size

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                    self._input_keep_prob < 1):
            do_inputs = dropout(inputs, self._input_keep_prob, seed=self._seed)
            inputs = tf.cond(self._is_train, lambda: do_inputs, lambda: inputs)
        output, new_state = self._cell(inputs, state)
        if (not isinstance(self._output_keep_prob, float) or
                    self._output_keep_prob < 1):
            do_output = dropout(output, self._output_keep_prob, seed=self._seed)
            output = tf.cond(self._is_train, lambda: do_output, lambda: output)
        return output, new_state
