import tensorflow as tf
# from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import MultiRNNCell

from directed.base_model import BaseTower, BaseRunner
from my.tensorflow import flatten, exp_mask
from my.tensorflow.nn import linear, relu1, dists
from my.tensorflow.rnn import dynamic_rnn, dynamic_bidirectional_rnn
import numpy as np

from my.tensorflow.rnn_cell import RSMCell, GRUCell


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
        forget_bias = params.forget_bias
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

        with tf.variable_scope("networks"):
            m_mask = tf.reduce_max(tf.cast(x_mask, 'int64'), 2, name='m_mask')  # [N, M]
            m_length = tf.reduce_sum(m_mask, 1, name='m_length')  # [N]
            fw_cell = RSMCell(d, forget_bias=forget_bias, wd=wd,
                              initializer=tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3)))
            bw_cell = RSMCell(d, forget_bias=forget_bias, wd=wd,
                              initializer=tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3)))
            us = tf.tile(tf.expand_dims(u, 1, name='u_prev_aug'), [1, M, 1])  # [N, d] -> [N, M, d]
            h_m_in = tf.concat(2, [us, m], name='h_in')
            dummy = tf.zeros([N, M, 2])
            h_m_in = tf.concat(2, [dummy, h_m_in])
            h_m_out, state_fw, state_bw, h_m_fw, h_m_bw = \
                dynamic_bidirectional_rnn(fw_cell, bw_cell, h_m_in,
                                          sequence_length=m_length, dtype='float', num_layers=L)
            # h, m = tf.split(2, 2, h_m_out)
            c_last = tf.squeeze(tf.slice(state_fw, [0, L-1, 0], [-1, 1, d]), [1])
            af_aug, rf_aug = tf.split(3, 2, tf.slice(h_m_fw, [0, 0, 0, 0], [-1, -1, -1, 2]))
            tensors['af'] = tf.squeeze(af_aug, [-1])
            tensors['rf'] = tf.squeeze(rf_aug, [-1])
            ab_aug, rb_aug = tf.split(3, 2, tf.slice(h_m_bw, [0, 0, 0, 0], [-1, -1, -1, 2]))
            tensors['ab'] = tf.squeeze(ab_aug, [-1])
            tensors['rb'] = tf.squeeze(rb_aug, [-1])

        with tf.variable_scope("selection"):
            w = tf.tanh(linear([c_last], d, True))

        with tf.variable_scope("class"):
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
