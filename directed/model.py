import tensorflow as tf
# from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import MultiRNNCell

from directed.base_model import BaseTower, BaseRunner
from my.tensorflow import flatten, exp_mask
from my.tensorflow.nn import linear
from my.tensorflow.rnn import dynamic_rnn
import numpy as np

from my.tensorflow.rnn_cell import BasicLSTMCell, GRUCell, GRUXCell, DropoutWrapper


class Embedder(object):
    def __call__(self, content):
        raise Exception()


class VariableEmbedder(Embedder):
    def __init__(self, params, initializer=None, name="variable_embedder"):
        V, d = params.vocab_size, params.hidden_size
        with tf.variable_scope(name):
            self.emb_mat = tf.get_variable("emb_mat", dtype='float', shape=[V, d], initializer=initializer)

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


class Tower(BaseTower):
    def initialize(self):
        params = self.params
        placeholders = self.placeholders
        tensors = self.tensors
        variables_dict = self.variables_dict
        N, J, V, Q, M = params.batch_size, params.max_sent_size, params.vocab_size, params.max_ques_size, params.max_num_sents
        d = params.hidden_size
        L = params.mem_num_layers
        keep_prob = params.keep_prob
        wd = params.wd
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

        with tf.variable_scope("embedding", initializer=self.default_initializer):
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
            cell = GRUXCell(d, input_size=1+2*d, wd=wd)
            u_prev = u
            us_prev = tf.zeros(shape=[N, M, d], dtype='float')
            a_list = []

        with tf.variable_scope("layers", initializer=self.default_initializer) as scope:
            for layer_idx in range(L):
                with tf.name_scope("layer_{}".format(layer_idx)):
                    w_a = tf.get_variable('w_a', shape=[d], dtype='float')
                    a_raw = tf.reduce_sum(tf.tanh(tf.expand_dims(u_prev, 1) * (m + us_prev)) * w_a, 2, name='a_raw')  # [N, M]
                    a = tf.mul(tf.nn.sigmoid(a_raw), tf.cast(m_mask, 'float'), name='a')  # [N, M]
                    a_list.append(a)
                    u_prev_tiled = tf.tile(tf.expand_dims(u_prev, 1), [1, M, 1], name='u_prev_tiled')
                    am = tf.concat(2, [tf.expand_dims(a, -1), m, u_prev_tiled], name='am')
                    us_f, u_f = dynamic_rnn(cell, am, sequence_length=m_length, dtype='float', scope='u_f')
                    us_b_rev, _ = dynamic_rnn(cell, tf.reverse(am, [False, True, False]), dtype='float', scope='u_b')
                    us_b = tf.reverse(us_b_rev, [False, True, False])
                    u_prev = u_f
                    us_prev = us_f + us_b
                    scope.reuse_variables()

            a_comb = tf.transpose(tf.pack(a_list), [1, 0, 2], name='a_comb')  # [N, L, M]
            tensors['a'] = a_comb

        with tf.variable_scope("class", initializer=self.default_initializer):
            w = tf.tanh(linear([u_prev], d, True), name='w')
            W = tf.transpose(A.emb_mat, name='W')
            logits = tf.matmul(w, W, name='logits')
            yp = tf.cast(tf.argmax(logits, 1), 'int32')
            correct = tf.equal(yp, y)
            tensors['yp'] = yp
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
                # j = len(para) - jj - 1  # reverting story sequence, last to first
                j = jj
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
