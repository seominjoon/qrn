import tensorflow as tf
# from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import DropoutWrapper, MultiRNNCell

from directed.base_model import BaseTower, BaseRunner
from my.tensorflow import flatten, exp_mask
from my.tensorflow.nn import linear
from my.tensorflow.rnn import dynamic_rnn
import numpy as np

from my.tensorflow.rnn_cell import BasicLSTMCell, GRUCell, GRUXCell


class Embedder(object):
    def __call__(self, content):
        raise Exception()


class VariableEmbedder(Embedder):
    def __init__(self, params, name="variable_embedder"):
        V, d = params.vocab_size, params.hidden_size
        with tf.variable_scope(name):
            self.emb_mat = tf.get_variable("emb_mat", dtype='float', shape=[V, d])

    def __call__(self, word, name="embedded_content"):
        out = tf.nn.embedding_lookup(self.emb_mat, word, name=name)
        return out


class PositionEncoder(object):
    def __init__(self, max_sent_size, hidden_size):
        self.max_sent_size, self.hidden_size = max_sent_size, hidden_size
        J, d = max_sent_size, hidden_size
        with tf.name_scope("pe_constants"):
            b = [1 - k/d for k in range(1, d+1)]
            w = [[j*(2*k/d - 1) for k in range(1, d+1)] for j in range(1, J+1)]
            self.b = tf.constant(b, shape=[d])
            self.w = tf.constant(w, shape=[J, d])

    def __call__(self, Ax, mask, scope=None):
        with tf.name_scope(scope or "position_encoder"):
            shape = Ax.get_shape().as_list()
            length_dim_index = len(shape) - 2
            length = tf.reduce_sum(tf.cast(mask, 'float'), length_dim_index)
            length = tf.maximum(length, 1.0)  # masked sentences will have length 0
            length_aug = tf.expand_dims(tf.expand_dims(length, -1), -1)
            l = self.b + self.w/length_aug
            mask_aug = tf.expand_dims(mask, -1)
            f = tf.reduce_sum(Ax * l * tf.cast(mask_aug, 'float'), length_dim_index, name='f')  # [N, S, d]
            return f


class GRU(object):
    def __init__(self, params, is_train):
        self.params = params
        d = params.hidden_size
        keep_prob = params.keep_prob
        rnn_num_layers = params.rnn_num_layers
        self.scope = tf.get_variable_scope()

        cell = GRUCell(d)
        do_cell = cell
        if keep_prob:
            do_cell = DropoutWrapper(do_cell, input_keep_prob=keep_prob)
        if rnn_num_layers > 1:
            cell = MultiRNNCell([cell] * rnn_num_layers)
            do_cell = MultiRNNCell([do_cell] * rnn_num_layers)
        self.cell = cell
        self.do_cell = do_cell
        self.is_train = is_train
        self.used = False

    def __call__(self, Ax, length=None, initial_state=None, feed_prev_out=False, dtype=None, name="encoded_sentence"):
        with tf.name_scope(name):
            NN, J, d = flatten(Ax.get_shape().as_list(), 3)
            L = self.params.rnn_num_layers
            Ax_flat = tf.reshape(Ax, [NN, J, d])
            if length is not None:
                length = tf.reshape(length, [NN])

            h_zeros_up = tf.constant(0.0, shape=[NN, (L-1)*d])
            h = None if initial_state is None else tf.concat(1, [tf.reshape(initial_state, [NN, d]), h_zeros_up], name='h')
            with tf.variable_scope(self.scope, reuse=self.used):
                # Always True feed_prev_out, because this is for test time.
                raw = dynamic_rnn(self.cell, Ax_flat, sequence_length=length, initial_state=h, dtype=dtype,
                                  feed_prev_out=feed_prev_out)
                tf.get_variable_scope().reuse_variables()
                do = dynamic_rnn(self.do_cell, Ax_flat, sequence_length=length, initial_state=h, dtype=dtype,
                                 feed_prev_out=feed_prev_out)
            o_flat, h_flat = tf.cond(self.is_train, lambda: do, lambda: raw)
            o = tf.reshape(o_flat, Ax.get_shape(), name='o')
            s_flat = tf.slice(h_flat, [0, (L-1)*d], [-1, -1])  # last h or multiRNN (excluding c)
            s = tf.reshape(s_flat, Ax.get_shape().as_list()[:-2] + [d], name='s')
            self.used = True
            return o, s


class Tower(BaseTower):
    def initialize(self):
        params = self.params
        placeholders = self.placeholders
        tensors = self.tensors
        variables_dict = self.variables_dict
        N, J, V, Q, M = params.batch_size, params.max_sent_size, params.vocab_size, params.max_ques_size, params.max_num_sents
        d = params.hidden_size
        L = params.mem_num_layers
        with tf.name_scope("placeholders"):
            x = tf.placeholder('int32', shape=[N, M, J], name='x')
            x_mask = tf.placeholder('bool', shape=[N, M, J], name='x_mask')
            q = tf.placeholder('int32', shape=[N, J], name='q')
            q_mask = tf.placeholder('bool', shape=[N, J], name='q_mask')
            y = tf.placeholder('int32', shape=[N], name='y')
            is_train = tf.placeholder('bool', shape=[], name='is_train')
            placeholders['x'] = x
            placeholders['x_mask'] = x_mask
            placeholders['q'] = q
            placeholders['q_mask'] = q_mask
            placeholders['y'] = y
            placeholders['is_train'] = is_train

        with tf.variable_scope("embedding"):
            A = VariableEmbedder(params, name='A')
            Aq = A(q, name='Ax')  # [N, S, J, d]
            Ax = A(x, name='Cx')  # [N, S, J, d]

        with tf.variable_scope("encoding"):
            # encoder = GRU(params, is_train)
            # _, u = encoder(Aq, length=q_length, dtype='float', name='u')  # [N, d]
            # _, f = encoder(Ax, length=x_length, dtype='float', name='f')  # [N, S, d]
            encoder = PositionEncoder(J, d)
            u = encoder(Aq, q_mask)  # [N, d]
            m = encoder(Ax, x_mask)  # [N, M, d]

        with tf.name_scope("pre_layers"):
            m_mask = tf.reduce_max(tf.cast(x_mask, 'int32'), 2, name='m_mask')  # [N, M]
            m_length = tf.reduce_sum(m_mask, 1, name='m_length')  # [N]
            tril = tf.constant(np.tril(np.ones([M, M], dtype='float32'), -1), name='tril')
            att_cell = GRUCell(d, input_size=d)
            cell = GRUXCell(d, input_size=d+1)
            u_prev = u
            a_prev = tf.zeros([N, M], dtype='float')
            ca_f_prev = tf.ones([N, M], dtype='float')
            a_list = []
            ca_f_list = []
        with tf.variable_scope("layers") as scope:
            for layer_idx in range(L):
                with tf.name_scope("layer_{}".format(layer_idx)):
                    T = linear([u_prev, a_prev], M, True, scope='T')
                    a_raw = tf.mul(tf.expand_dims(u_prev, 1), m, name='a_raw')  # [N, M, d]
                    a_raw = tf.reduce_sum(a_raw, 2)
                    # a_raw, _ = dynamic_rnn(att_cell, a_raw, sequence_length=m_length, dtype='float')
                    # a = tf.nn.softmax(exp_mask(exp_mask(tf.reduce_sum(a_raw, 2), m_mask), ca_f_prev), name='a')  # [N, M]
                    a = tf.nn.softmax(exp_mask(a_raw + T, m_mask), name='a') # [N, M]
                    a_list.append(a)
                    am = tf.concat(2, [tf.expand_dims(a, -1), m], name='am')
                    _, u_cur = dynamic_rnn(cell, am, sequence_length=m_length, initial_state=u_prev, scope='u')
                    # o = tf.reduce_sum(m * tf.expand_dims(a, -1), 1)
                    # u = tf.tanh(linear([u_prev, o, u_prev * o], d, True), name='u')
                    # u = o + u_prev
                    ca_f = tf.matmul(a, tril, transpose_b=True)
                    ca_f_list.append(ca_f)
                    u_prev = u_cur
                    ca_f_prev = ca_f
                    scope.reuse_variables()

            a_comb = tf.transpose(tf.pack(a_list, name='a_comb'), [1, 0, 2])  # [N, L, M]
            ca_f_comb = tf.transpose(tf.pack(ca_f_list, name='ca_f'), [1, 0, 2])  # [N, L, M]
            tensors['a_comb'] = a_comb
            tensors['ca_f_comb'] = ca_f_comb

        with tf.variable_scope("class"):
            w = tf.tanh(linear([u_prev], d, True), name='w')
            W = tf.transpose(A.emb_mat, name='W')
            logits = tf.matmul(w, W, name='logits')
            correct = tf.equal(tf.argmax(logits, 1), tf.cast(y, 'int64'))
            tensors['correct'] = correct

        with tf.name_scope("loss") as scope:
            with tf.name_scope("ans_loss"):
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='ce')
                avg_ce = tf.reduce_mean(ce, name='avg_ce')
                tf.add_to_collection('losses', avg_ce)

            losses = tf.get_collection('losses', scope=scope)
            loss = tf.add_n(losses, name='loss')
            tensors['loss'] = loss

        variables_dict['all'] = tf.trainable_variables()

    def get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        N, J, V, M = params.batch_size, params.max_sent_size, params.vocab_size, params.max_num_sents
        x = np.zeros([N, M, J], dtype='int32')
        x_mask = np.zeros([N, M, J], dtype='bool')
        q = np.zeros([N, J], dtype='int32')
        q_mask = np.zeros([N, J], dtype='bool')
        y = np.zeros([N], dtype='int32')

        ph = self.placeholders
        feed_dict = {ph['x']: x, ph['x_mask']: x_mask,
                     ph['q']: q, ph['q_mask']: q_mask,
                     ph['y']: y,
                     ph['is_train']: mode == 'train'
                     }
        if batch is None:
            return feed_dict

        X, Q, S, Y, H, T = batch
        for i, para in enumerate(X):
            for jj, sent in enumerate(para):
                j = len(para) - jj - 1  # reverting story sequence, last to first
                for k, word in enumerate(sent):
                    x[i, j, k] = word
                    x_mask[i, j, k] = True

        for i, ques in enumerate(Q):
            for j, word in enumerate(ques):
                q[i, j] = word
                q_mask[i, j] = True

        for i, ans in enumerate(Y):
            y[i] = ans

        return feed_dict


class Runner(BaseRunner):
    def _get_train_op(self, **kwargs):
        return self.train_ops['all']
