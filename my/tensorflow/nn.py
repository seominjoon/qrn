from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf

from my.tensorflow import flatten


def linear(args, output_size, bias, bias_start=0.0, name='', scope=None, var_on_cpu=False, wd=0.0, squeeze=False, initializer=None, feat=None, state=None, drop_rate = 1.0):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    assert len(set(tuple(shape[:-1]) for shape in shapes)) <= 1

    new_shapes = [flatten(shape, 2) for shape in shapes]
    res_shape = shapes[0][:-1] + [output_size]
    args = [tf.reshape(arg, new_shape) for arg, new_shape in zip(args, new_shapes)]

    for new_shape in new_shapes:
        if len(new_shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not new_shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += new_shape[1]

    # This is for Dialog Match
    if feat is not None : total_arg_size -= 2
    if state is not None : total_arg_size += state.get_shape().as_list()[-1]


    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        if var_on_cpu:
            with tf.device("/cpu:0"):
                matrix = vs.get_variable("Matrix"+name, [total_arg_size, output_size], initializer=initializer)
        else:
            matrix = vs.get_variable("Matrix"+name, [total_arg_size, output_size], initializer=initializer)

        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(matrix), wd, name='weight_loss'+name)
            tf.add_to_collection('losses', weight_decay)
	
	# Modify for Feature Matching
        if feat is not None:
            W = vs.get_variable("Embed"+name, [total_arg_size+2, total_arg_size+2], initializer=initializer)
            args_ = tf.transpose(math_ops.matmul(args[0],W)) # [D * N]
            h1 = tf.transpose(tf.slice(args_, [0, 0], [total_arg_size, -1]))
            h2 = tf.slice(args_, [total_arg_size, 0], [2, -1])
            f = tf.cast(tf.transpose(feat, [1,0,2]), 'float32') # [2 * N * A]
	    
            res = math_ops.matmul(h1, matrix) # [N * A]
            for i in range(2):
                h2_ = tf.gather(h2, i) # [1 * N]
                f_ = tf.gather(f, i) # [1 * N * A]
                res += tf.transpose(tf.mul(h2_, tf.transpose(f_)))
        elif state is not None:
            h = tf.concat(1, [args[0], tf.cast(state, 'float32')]) # [D+A]
            res = math_ops.matmul(h, matrix) # [N * A]
        else:
            if len(args) == 1:
            	res = math_ops.matmul(args[0], matrix)
            else:
            	res = math_ops.matmul(array_ops.concat(1, args), matrix)

        if not bias:
            res = tf.reshape(res, res_shape, name='out'+name)
            if squeeze:
                res = tf.squeeze(res, squeeze_dims=[len(res_shape)-1])
            return res
        bias_term = vs.get_variable(
            "Bias"+name, [output_size],
            initializer=init_ops.constant_initializer(bias_start))
        res = res + bias_term
        res = tf.reshape(res, res_shape, name='out'+name)
        if squeeze:
            res = tf.squeeze(res, squeeze_dims=[len(res_shape)-1])
    return res


def relu1(features, name=None):
    name = name or "relu1"
    return tf.minimum(tf.maximum(features, 0), 1, name=name)


def dists(a, b):
    return [a * b]
