import tensorflow as tf

from parallel.base_model import BaseTower, BaseRunner
from my.tensorflow.nn import linear
from my.tensorflow.rnn import dynamic_rnn, dynamic_bidirectional_rnn
import numpy as np

from my.tensorflow.rnn_cell import RSMCell


class Embedder(object):
    def __call__(self, content):
        raise Exception()


class VariableEmbedder(Embedder):
    def __init__(self, params, wd=0.0, initializer=None, name="variable_embedder"):
        V, d = params.vocab_size, params.hidden_size
        with tf.variable_scope(name):
            self.emb_mat = tf.get_variable("emb_mat", dtype='float', shape=[V, d], initializer=initializer)
            # TODO : not sure wd is appropriate for embedding matrix
            if wd:
                weight_decay = tf.mul(tf.nn.l2_loss(self.emb_mat), wd, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)

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
            # l = self.b + self.w/self.max_sent_size
            mask_aug = tf.expand_dims(mask, -1)
            f = tf.reduce_sum(Ax * l * tf.cast(mask_aug, 'float'), length_dim_index, name='f')  # [N, S, d]

            return f


class VariablePositionEncoder(object):
    def __init__(self, max_sent_size, hidden_size, scope=None):
        self.max_sent_size, self.hidden_size = max_sent_size, hidden_size
        J, d = max_sent_size, hidden_size
        with tf.variable_scope(scope or self.__class__.__name__):
            self.w = tf.get_variable('w', shape=[J, d], dtype='float')

    def __call__(self, Ax, mask, scope=None):
        with tf.name_scope(scope or self.__class__.__name__):
            shape = Ax.get_shape().as_list()
            length_dim_index = len(shape) - 2
            mask_aug = tf.expand_dims(mask, -1)
            f = tf.reduce_sum(Ax * self.w * tf.cast(mask_aug, 'float'), length_dim_index, name='f')  # [N, S, d]
        return f


class RegressionLayer(object):
    def __init__(self, batch_size, mem_size, hidden_size):
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.batch_size = batch_size
        N, M, d = batch_size, mem_size, hidden_size
        self.L = np.tril(np.ones([M, M]))
        self.sL = np.tril(np.ones([M, M]), k=-1)

    def __call__(self, u_t, a, b, mask, scope=None):
        """

        :param u_t: [N, M, d]
        :param a: [N, M. 1]
        :param b: [N, M. 1]
        :param mask:  [N, M]
        :return:
        """
        N, M, d = self.batch_size, self.mem_size, self.hidden_size
        L, sL = self.L, self.sL
        with tf.name_scope(scope or self.__class__.__name__):
            L = tf.tile(tf.expand_dims(L, 0), [N, 1])
            sL = tf.tile(tf.expand_dims(sL, 0), [N, 1])
            A = tf.tile(a, [1, 1, d])  # [N, M, d]
            B = tf.tile(b, [1, 1, M])  # [N, M, M]
            B = tf.concat(2, [tf.zeros([N, M, 1]), tf.slice(B, [0, 0, 1], [-1, -1, -1])])
            B = tf.log(B + 1e-9)
            left = L * tf.exp(tf.batch_matmul(L, B * sL))  # [N, M, M]
            right = A * u_t  # [N, M, d]
            u = tf.batch_matmul(left, right)  # [N, M, d]
            mask_aug = tf.tile(tf.expand_dims(mask, -1), d)
            u = tf.boolean_mask(u, mask_aug)
        return u


class Tower(BaseTower):
    def initialize(self):
        params = self.params
        placeholders = self.placeholders
        tensors = self.tensors
        variables_dict = self.variables_dict
        N, J, V, Q, M = params.batch_size, params.max_sent_size, params.vocab_size, params.max_ques_size, params.mem_size
        d = params.hidden_size
        L = params.mem_num_layers
        forget_bias = params.forget_bias
        wd = params.wd
        initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))
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
            A = VariableEmbedder(params, wd=wd, initializer=initializer, name='A')
            Aq = A(q, name='Aq')  # [N, S, J, d]
            Ax = A(x, name='Ax')  # [N, S, J, d]

        with tf.name_scope("encoding"):
            encoder = PositionEncoder(J, d)
            u = encoder(Aq, q_mask)  # [N, d]
            m = encoder(Ax, x_mask)  # [N, M, d]

        with tf.variable_scope("networks"):
            m_mask = tf.reduce_max(tf.cast(x_mask, 'int64'), 2, name='m_mask')  # [N, M]
            m_length = tf.reduce_sum(m_mask, 1, name='m_length')  # [N]
            initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))
            prev_u = tf.tile(tf.expand_dims(u, 1), [1, M, 1])  # [N, M, d]
            reg_layer = RegressionLayer(N, M, d)
            h = None  # [N, M, d]
            as_, rfs, rbs = [], [], []
            for layer_idx in range(L):
                with tf.name_scope("Layer {}".format(layer_idx)):
                    u_t = tf.tanh(linear([prev_u, m], d, True))
                    a, rf, rb = tf.split(2, 3, tf.sigmoid(linear([prev_u * m], 3, True)))
                    tf.get_variable_scope().reuse_variables()
                    u_t_rev = tf.reverse_sequence(u_t, m_length, 1)
                    a_rev, rb_rev = tf.reverse_sequence(a, m_length, 1), tf.reverse_sequence(rb, m_length, 1)
                    uf = reg_layer(u_t, a*rf, 1.0-a, m_mask)
                    h = reg_layer(u_t, a, 1.0-a, m_mask)
                    ub_rev = reg_layer(u_t_rev, a_rev*rb_rev, 1.0-a_rev, m_mask)
                    ub = tf.reverse_sequence(ub_rev, m_length, 1)
                    prev_u = uf + ub

            # TODO : obtain last h?
            a = tf.pack(as_, name='a')
            rf = tf.pack(rfs, name='rf')
            rb = tf.pack(rbs, name='rb')
            tensors['a'] = a
            tensors['rf'] = rf
            tensors['rb'] = rb

        with tf.variable_scope("selection"):
            # w = tf.nn.relu(linear([fw_v + 1e-9*(fw_h+bw_h)], d, True, wd=wd))
            w = fw_v + 1e-9*(fw_h + bw_h)
            tensors['s'] = a

        with tf.variable_scope("class"):
            if params.use_ques:
                logits = linear([w, u], V, True, wd=wd)
            else:
                # W = tf.transpose(A.emb_mat, name='W')
                W = tf.get_variable('W', shape=[d, V])
                logits = tf.matmul(w, W, name='logits')
            yp = tf.cast(tf.argmax(logits, 1), 'int32')
            correct = tf.equal(yp, y)
            tensors['yp'] = yp
            tensors['correct'] = correct

        with tf.name_scope("loss"):
            with tf.name_scope("ans_loss"):
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='ce')
                avg_ce = tf.reduce_mean(ce, name='avg_ce')
                tf.add_to_collection('losses', avg_ce)

            losses = tf.get_collection('losses')
            loss = tf.add_n(losses, name='loss')
            tensors['loss'] = loss

        variables_dict['all'] = tf.trainable_variables()

    def get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        N, J, V, M = params.batch_size, params.max_sent_size, params.vocab_size, params.mem_size
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
            if len(para) > M:
                para = para[-M:]
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
