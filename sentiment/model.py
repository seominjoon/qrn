import tensorflow as tf
# from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import MultiRNNCell

from sentiment.base_model import BaseTower, BaseRunner
from my.tensorflow import flatten, exp_mask, translate, variable_on_cpu
from my.tensorflow.nn import linear, relu1, dists
from my.tensorflow.rnn import dynamic_rnn, dynamic_bidirectional_rnn
import numpy as np

from my.tensorflow.rnn_cell import RSMCell, GRUCell, TempCell, BiDropoutWrapper, DropoutWrapper, PassingCell, XGRUCell


class Embedder(object):
    def __call__(self, content):
        raise Exception()


class Tower(BaseTower):
    def initialize(self):
        params = self.params
        placeholders = self.placeholders
        tensors = self.tensors
        variables_dict = self.variables_dict
        N, M, V, K, C = params.batch_size, params.mem_size, params.vocab_size, params.word_size, params.num_classes
        d = params.hidden_size
        L = params.mem_num_layers
        forget_bias = params.forget_bias
        wd = params.wd
        init_emb_mat = params.emb_mat
        finetune = params.finetune
        with tf.name_scope("placeholders"):
            x = tf.placeholder('int32', shape=[N, M], name='x')
            x_mask = tf.placeholder('bool', shape=[N, M], name='x_mask')
            y = tf.placeholder('int32', shape=[N], name='y')
            is_train = tf.placeholder('bool', shape=[], name='is_train')
            emb_mat = tf.placeholder('float', shape=[V, K], name='emb_mat')
            placeholders['x'] = x
            placeholders['x_mask'] = x_mask
            placeholders['y'] = y
            placeholders['is_train'] = is_train
            placeholders['emb_mat'] = emb_mat

        with tf.variable_scope("embedding"):
            if finetune:
                def emb_mat_initializer(*args, **kwargs):
                    return tf.constant(init_emb_mat, dtype='float')
                emb_mat = variable_on_cpu("emb_mat", shape=[V, K], initializer=emb_mat_initializer)

        with tf.variable_scope("encoding"):
            u = tf.get_variable('u', shape=[d])
            u = tf.tile(tf.expand_dims(u, 0), [N, 1])  # [N, d]
            m = tf.tanh(linear([tf.nn.embedding_lookup(emb_mat, x)], d, True))  # [N, M, d]

        with tf.variable_scope("networks"):
            m_length = tf.reduce_sum(tf.cast(x_mask, 'int64'), 1, name='m_length')  # [N]
            initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))
            cell = RSMCell(d, forget_bias=forget_bias, wd=wd, initializer=initializer)
            us = tf.tile(tf.expand_dims(u, 1, name='u_prev_aug'), [1, M, 1])  # [N, d] -> [N, M, d]
            in_ = tf.concat(2, [tf.ones([N, M, 1]), m, us, tf.zeros([N, M, 2*d])], name='x_h_in')  # [N, M, 4*d + 1]
            out, fw_state, bw_state, bi_tensors = dynamic_bidirectional_rnn(cell, in_,
                sequence_length=m_length, dtype='float', num_layers=L)
            a = tf.slice(out, [0, 0, 0], [-1, -1, 1])  # [N, M, 1]
            _, _, v, g = tf.split(2, 4, tf.slice(out, [0, 0, 1], [-1, -1, -1]))
            fw_h, fw_v = tf.split(1, 2, tf.slice(fw_state, [0, 1], [-1, -1]))
            bw_h, bw_v = tf.split(1, 2, tf.slice(bw_state, [0, 1], [-1, -1]))

            _, fw_u_out, fw_v_out, _ = tf.split(2, 4, tf.squeeze(tf.slice(bi_tensors['fw_out'], [0, L-1, 0, 2], [-1, -1, -1, -1]), [1]))
            _, bw_u_out, bw_v_out, _ = tf.split(2, 4, tf.squeeze(tf.slice(bi_tensors['bw_out'], [0, L-1, 0, 2], [-1, -1, -1, -1]), [1]))

            tensors['a'] = tf.squeeze(tf.slice(bi_tensors['in'], [0, 0, 0, 0], [-1, -1, -1, 1]), [3])
            tensors['of'] = tf.squeeze(tf.slice(bi_tensors['fw_out'], [0, 0, 0, 0], [-1, -1, -1, 1]), [3])
            tensors['ob'] = tf.squeeze(tf.slice(bi_tensors['bw_out'], [0, 0, 0, 0], [-1, -1, -1, 1]), [3])


        with tf.variable_scope("selection"):
            # w = tf.nn.relu(linear([fw_v + 1e-9*(fw_h+bw_h)], d, True, wd=wd))
            w = fw_v
            tensors['s'] = a

        with tf.variable_scope("class"):
            if params.use_ques:
                logits = linear([w, u], V, True, wd=wd)
            else:
                # W = tf.transpose(A.emb_mat, name='W')
                W = tf.get_variable('W', shape=[d, C])
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
        N, V, M, K, C = params.batch_size, params.vocab_size, params.mem_size, params.word_size, params.num_classes
        x = np.zeros([N, M], dtype='int32')
        x_mask = np.zeros([N, M], dtype='bool')
        cy = np.zeros([N])
        y = np.zeros([N], dtype='int32')
        emb_mat = params.emb_mat

        ph = self.placeholders
        feed_dict = {ph['x']: x, ph['x_mask']: x_mask,
                     ph['y']: y,
                     ph['emb_mat']: emb_mat,
                     ph['is_train']: mode == 'train',
                     }
        if batch is None:
            return feed_dict

        X, Y = batch
        for i, sent in enumerate(X):
            if len(sent) > M:
                sent = sent[-M:]
            for j, word in enumerate(sent):
                x[i, j] = word
                x_mask[i, j] = True

        interval = 1.0 / C

        def score2class(score_):
            return int(max(np.ceil(score_ / interval) - 1, 0))

        for idx, score in enumerate(Y):
            cy[idx] = score
            class_ = score2class(score)
            y[idx] = class_

        return feed_dict


class Runner(BaseRunner):
    def _get_train_op(self, **kwargs):
        return self.train_ops['all']
