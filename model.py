import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import DropoutWrapper, MultiRNNCell

from base_model import BaseTower
from my.tensorflow import flatten
from my.tensorflow.nn import linear
import numpy as np

from my.tensorflow.rnn_cell import BasicLSTMCell


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
    @staticmethod
    def _get_l_tensor(J, d, name='l'):
        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)
        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]
        l = [g(j) for j in range(J)]
        l_tensor = tf.constant(l, shape=[J, d], name=name)
        return l_tensor

    def __init__(self, params):
        self.params = params
        J, d = params.max_sent_size, params.hidden_size
        with tf.name_scope("position_encoder"):
            self._l = PositionEncoder._get_l_tensor(J, d)

    def __call__(self, embedder, word, mask, name="encoded_sentence"):
        with tf.name_scope(name):
            assert isinstance(embedder, Embedder)
            Ax = embedder(word)
            shape = Ax.get_shape().as_list()
            length_dim_index = len(shape) - 2
            mask_aug = tf.expand_dims(mask, -1)
            f = tf.reduce_sum(Ax * self._l * tf.cast(mask_aug, 'float'), length_dim_index, name='f')  # [N, S, d]
            return f


class LSTM(object):
    def __init__(self, params, is_train):
        self.params = params
        d = params.hidden_size
        keep_prob = params.keep_prob
        rnn_num_layers = params.rnn_num_layers
        self.scope = tf.get_variable_scope()

        cell = BasicLSTMCell(d)
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

    def __call__(self, Ax, length, name="encoded_sentence"):
        with tf.name_scope(name):
            NN, J, d = flatten(Ax.get_shape().as_list(), 3)
            L = self.params.rnn_num_layers
            Ax_flat = tf.reshape(Ax, [NN, J, d])
            length = tf.reshape(length, [NN])

            with tf.variable_scope(self.scope, reuse=self.used):
                raw = dynamic_rnn(self.cell, Ax_flat, sequence_length=length, dtype='float')
                tf.get_variable_scope().reuse_variables()
                do = dynamic_rnn(self.do_cell, Ax_flat, sequence_length=length, dtype='float')
            o_flat, h_flat = tf.cond(self.is_train, lambda: do, lambda: raw)
            o = tf.reshape(o_flat, Ax.get_shape(), name='o')
            s_flat = tf.slice(h_flat, [0, (2*L-1)*d], [-1, -1])  # last h or multiRNN (excluding c)
            s = tf.reshape(s_flat, Ax.get_shape().as_list()[:-2] + [d], name='s')
            self.used = True
            return o, s


class Tower(BaseTower):
    def initialize(self):
        params = self.params
        placeholders = self.placeholders
        tensors = self.tensors
        N, J, V, Q, S = params.batch_size, params.max_sent_size, params.vocab_size, params.max_ques_size, params.max_num_sups
        d = params.hidden_size
        with tf.name_scope("placeholders"):
            x = tf.placeholder('int32', shape=[N, S, J], name='x')
            x_length = tf.placeholder('int32', shape=[N, S], name='x_length')
            x_eos = tf.placeholder('int32', shape=[N, S, J+1], name='x_eos')
            eos_x = tf.placeholder('int32', shape=[N, S, J+1], name='eos_x')
            x_mask = tf.placeholder('bool', shape=[N, S, J], name='x_mask')
            x_eos_mask = tf.placeholder('bool', shape=[N, S, J+1], name='x_eos_mask')
            q = tf.placeholder('int32', shape=[N, J], name='q')
            q_length = tf.placeholder('int32', shape=[N], name='q_length')
            q_mask = tf.placeholder('bool', shape=[N, J], name='q_mask')
            y = tf.placeholder('int32', shape=[N, V], name='y')
            is_train = tf.placeholder('bool', shape=[], name='is_train')
            placeholders['x'] = x
            placeholders['x_length'] = x_length
            placeholders['x_eos'] = x_eos
            placeholders['eos_x'] = eos_x
            placeholders['x_mask'] = x_mask
            placeholders['x_eos_mask'] = x_eos_mask
            placeholders['q'] = q
            placeholders['q_length'] = q_length
            placeholders['q_mask'] = q_mask
            placeholders['y'] = y
            placeholders['is_train'] = is_train

        with tf.variable_scope("embedding"):
            A = VariableEmbedder(params, name='A')
            C = VariableEmbedder(params, name='C')
            Aq = A(q, name='Ax')  # [N, S, J, d]
            Cx = C(x, name='Cx')  # [N, S, J, d]
            C_eos_x = C(eos_x, name='C_eos_x')  # [N, S, J+1, d]

        with tf.variable_scope("encoding"):
            # encoder = PositionEncoder(params)
            encoder = LSTM(params, is_train)
            _, u = encoder(Aq, q_length, name='u')
            _, f = encoder(Cx, x_length, name='f')

        with tf.variable_scope("decoding"):
            decoder = LSTM(params, is_train)
            o, _ = decoder(C_eos_x, x_length + 1, 'o')  # [N, S, J+1, d]

        with tf.name_scope("gen"):
            gen_W = tf.transpose(C.emb_mat, name='gen_W')  # [d, V]
            o_flat = tf.reshape(o, [N*S*(J+1), d], name='o_flat')
            gen_logits_flat = tf.matmul(o_flat, gen_W, name='gen_logits_flat')  # [N*S*(J+1), V]
            gen_logits = tf.reshape(gen_logits_flat, [N, S, J+1, V])
            fd = tf.argmax(gen_logits, 3, name='f_decoded')
            tensors['fd'] = fd

        with tf.variable_scope("rule"):
            f_flat = tf.reshape(f, [N, S * d], name='f_flat')
            g = tf.tanh(linear([f_flat], d, True), name='g')
            # TODO : how do I generate output without rnn inputs?
            # og = decoder(g, 2*J*tf.ones([N]), 'og')
            # tensors['og'] = og


        with tf.name_scope("class"):
            uf = tf.tanh(linear([g, u], d, True), name='u_f')  # [N, d]
            W = tf.transpose(A.emb_mat, name='W')
            logits = tf.matmul(uf, W, name='logits')
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            tensors['correct'] = correct

        with tf.name_scope("loss") as scope:
            with tf.name_scope("ans_loss"):
                ce = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(y, 'float'), name='ce')
                avg_ce = tf.reduce_mean(ce, name='avg_ce')
                tf.add_to_collection('losses', avg_ce)

            with tf.name_scope("gen_loss"):
                x_eos_flat = tf.reshape(x_eos, [N*S*(J+1)], name='x_eos_sparse_flat')  # [N*S*(J+1)]
                gen_ce_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(gen_logits_flat, x_eos_flat, name='gen_ce_flat')
                x_eos_mask_flat = tf.reshape(tf.cast(x_eos_mask, 'float'), [N*S*(J+1)], name='x_eos_mask_flat')
                gen_avg_ce = tf.div(tf.reduce_sum(gen_ce_flat * x_eos_mask_flat), tf.reduce_sum(x_eos_mask_flat), name='gen_avg_ce')
                tf.add_to_collection('losses', gen_avg_ce)

            losses = tf.get_collection('losses', scope=scope)
            loss = tf.add_n(losses, name='loss')
            tensors['loss'] = loss

    def get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        eos_idx = params.eos_idx
        N, J, V, S = params.batch_size, params.max_sent_size, params.vocab_size, params.max_num_sups
        x = np.zeros([N, S, J], dtype='int32')
        x_length = np.zeros([N, S], dtype='int32')
        eos_x = np.zeros([N, S, J+1], dtype='int32')
        x_eos = np.zeros([N, S, J+1], dtype='int32')
        x_mask = np.zeros([N, S, J], dtype='bool')
        x_eos_mask = np.zeros([N, S, J+1], dtype='bool')
        q = np.zeros([N, J], dtype='int32')
        q_length = np.zeros([N], dtype='int32')
        q_mask = np.zeros([N, J], dtype='bool')
        y = np.zeros([N, V], dtype='bool')

        ph = self.placeholders
        feed_dict = {ph['x']: x, ph['eos_x']: eos_x, ph['x_eos']: x_eos,
                     ph['x_length']: x_length,
                     ph['x_mask']: x_mask, ph['x_eos_mask']: x_eos_mask,
                     ph['q']: q, ph['q_mask']: q_mask, ph['q_length']: q_length,
                     ph['y']: y,
                     ph['is_train']: mode == 'train'}
        if batch is None:
            return feed_dict

        X, Q, S, Y = batch
        for i, (para, supports) in enumerate(zip(X, S)):
            for j, support in enumerate(supports):
                sent = para[support]
                x_length[i, j] = len(sent)
                for k, word in enumerate(sent):
                    x[i, j, k] = word
                    x_eos[i, j, k] = word
                    eos_x[i, j, k+1] = word
                    x_mask[i, j, k] = True
                    x_eos_mask[i, j, k] = True
                x_eos[i, j, -1] = eos_idx
                x_eos_mask[i, j, -1] = True
                eos_x[i, j, 0] = eos_idx

        for i, ques in enumerate(Q):
            q_length[i] = len(ques)
            for j, word in enumerate(ques):
                q[i, j] = word
                q_mask[i, j] = True

        for i, ans in enumerate(Y):
            y[i, ans] = True
        return feed_dict