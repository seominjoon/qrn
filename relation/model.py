import tensorflow as tf
# from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import DropoutWrapper, MultiRNNCell

from base_model import BaseTower
from my.tensorflow import flatten
from my.tensorflow.nn import linear
from my.tensorflow.rnn import dynamic_rnn
import numpy as np

from my.tensorflow.rnn_cell import BasicLSTMCell, GRUCell, CRUCell


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


class GRU(object):
    def __init__(self, num_layers, hidden_size, keep_prob, is_train, scope=None):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        d = hidden_size
        with tf.variable_scope(scope or self.__class__.__name__):
            self.scope = tf.get_variable_scope()

        cell = GRUCell(d)
        do_cell = cell
        if keep_prob:
            do_cell = DropoutWrapper(do_cell, input_keep_prob=keep_prob)
        if num_layers > 1:
            cell = MultiRNNCell([cell] * num_layers)
            do_cell = MultiRNNCell([do_cell] * num_layers)
        self.cell = cell
        self.do_cell = do_cell
        self.is_train = is_train
        self.used = False

    def __call__(self, Ax, length=None, initial_state=None, feed_prev_out=False, dtype=None, name="encoded_sentence"):
        with tf.name_scope(name):
            NN, J, d = flatten(Ax.get_shape().as_list(), 3)
            L = self.num_layers
            Ax_flat = tf.reshape(Ax, [NN, J, d])
            if length is not None:
                length = tf.reshape(length, [NN])

            h_zeros_up = tf.constant(0.0, shape=[NN, (L-1)*d])
            h = None if initial_state is None else tf.concat(1, [tf.reshape(initial_state, [NN, d]), h_zeros_up], name='h')
            with tf.variable_scope(self.scope, reuse=self.used):
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
        N, J, V, Q, S = params.batch_size, params.max_sent_size, params.vocab_size, params.max_ques_size, params.max_num_sups
        L = params.rnn_num_layers
        C = params.num_args
        d = params.hidden_size
        with tf.name_scope("placeholders"):
            x = tf.placeholder('int32', shape=[N, S, J], name='x')
            x_length = tf.placeholder('int32', shape=[N, S], name='x_length')
            x_mask = tf.placeholder('bool', shape=[N, S, J], name='x_mask')
            q = tf.placeholder('int32', shape=[N, J], name='q')
            q_length = tf.placeholder('int32', shape=[N], name='q_length')
            q_mask = tf.placeholder('bool', shape=[N, J], name='q_mask')
            y = tf.placeholder('int32', shape=[N, V], name='y')
            # h_length = tf.placeholder('int32', shape=[N], name='h_length')
            is_train = tf.placeholder('bool', shape=[], name='is_train')
            placeholders['x'] = x
            placeholders['x_length'] = x_length
            placeholders['x_mask'] = x_mask
            placeholders['q'] = q
            placeholders['q_length'] = q_length
            placeholders['q_mask'] = q_mask
            placeholders['y'] = y
            placeholders['is_train'] = is_train

        with tf.variable_scope("embedding"):
            A = VariableEmbedder(params, name='A')
            Aq = A(q, name='Ax')  # [N, S, J, d]
            Ax = A(x, name='Cx')  # [N, S, J, d]

        with tf.variable_scope("encoding"):
            rel_encoder = GRU(L, d, params.keep_prob, is_train)
            _, ru = rel_encoder(Aq, length=q_length, dtype='float', name='ru')  # [N, d]
            _, rf = rel_encoder(Ax, length=x_length, dtype='float', name='rf')  # [N, S, d]
            arg_encoders = []
            aus = []
            afs = []
            for arg_idx in range(C):
                with tf.variable_scope("arg_{}".format(arg_idx)):
                    arg_encoder = GRU(L, d, params.keep_prob, is_train)
                    _, au = arg_encoder(Aq, length=q_length, dtype='float', name='au')  # [N, d]
                    _, af = arg_encoder(Ax, length=x_length, dtype='float', name='af')  # [N, S, d]
                    arg_encoders.append(arg_encoder)
                    aus.append(au)
                    afs.append(af)

        with tf.variable_scope("reasoning"):
            u_i_flat = tf.concat(1, [ru] + aus, name='u_i')  # [N, (C+1)*d]
            f_flat = tf.concat(2, [rf] + afs, name='f')  # [N, S, (C+1)*d]
            cru_cell = CRUCell(d, d, C)
            length = tf.reduce_sum(tf.reduce_max(tf.cast(x_mask, 'float'), 2), 1)
            _, u_f_flat = dynamic_rnn(cru_cell, f_flat, sequence_length=length, initial_state=u_i_flat, dtype='float')
            u_f = tf.reshape(u_f_flat, [N, C+1, d])
            ru_f = tf.squeeze(tf.slice(u_f, [0, 0, 0], [-1, 1, -1]), [1])
            au_f = tf.slice(u_f, [0, 1, 0], [-1, -1, -1], name='au_f')

        with tf.variable_scope("class"):
            p = tf.nn.softmax(linear([ru_f], C, True), name='p')  # [N, C]
            p_aug = tf.expand_dims(p, -1, name='p_aug')
            w = tf.reduce_sum(p_aug * au_f, 1, name='w')
            W = tf.transpose(A.emb_mat, name='W')
            logits = tf.matmul(w, W, name='logits')
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1), name='correct')
            tensors['correct'] = correct
            tensors['p'] = p

        with tf.name_scope("visualization"):
            u_f_flat_2 = tf.reshape(u_f, [N*(C+1), d], name='u_f_flat_2')
            u_logits_flat = tf.matmul(u_f_flat_2, W, name='u_logits_flat')  # [N*(C+1), V]
            u_logits = tf.reshape(u_logits_flat, [N, C+1, V], name='u_logits')
            u_surface = tf.argmax(u_logits, 2, name='u_surface')
            tensors['u_surface'] = u_surface

        with tf.name_scope("loss") as scope:
            with tf.name_scope("ans_loss"):
                ce = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(y, 'float'), name='ce')
                avg_ce = tf.reduce_mean(ce, name='avg_ce')
                tf.add_to_collection('losses', avg_ce)

            losses = tf.get_collection('losses', scope=scope)
            loss = tf.add_n(losses, name='loss')
            tensors['loss'] = loss

    def get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        N, J, V, S = params.batch_size, params.max_sent_size, params.vocab_size, params.max_num_sups
        x = np.zeros([N, S, J], dtype='int32')
        x_length = np.zeros([N, S], dtype='int32')
        x_mask = np.zeros([N, S, J], dtype='bool')
        q = np.zeros([N, J], dtype='int32')
        q_length = np.zeros([N], dtype='int32')
        q_mask = np.zeros([N, J], dtype='bool')
        y = np.zeros([N, V], dtype='bool')

        ph = self.placeholders
        feed_dict = {ph['x']: x,
                     ph['x_length']: x_length,
                     ph['x_mask']: x_mask,
                     ph['q']: q, ph['q_mask']: q_mask, ph['q_length']: q_length,
                     ph['y']: y,
                     ph['is_train']: mode == 'train'}
        if batch is None:
            return feed_dict

        X, Q, S, Y, H = batch
        for i, (para, supports) in enumerate(zip(X, S)):
            for j, support in enumerate(supports):
                sent = para[support]
                x_length[i, j] = len(sent)
                for k, word in enumerate(sent):
                    x[i, j, k] = word
                    x_mask[i, j, k] = True

        for i, ques in enumerate(Q):
            q_length[i] = len(ques)
            for j, word in enumerate(ques):
                q[i, j] = word
                q_mask[i, j] = True

        for i, ans in enumerate(Y):
            y[i, ans] = True

        return feed_dict
