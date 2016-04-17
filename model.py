import tensorflow as tf
from base_model import BaseTower
from my.tensorflow.nn import linear
import numpy as np


class Tower(BaseTower):
    def initialize(self):
        params = self.params
        placeholders = self.placeholders
        tensors = self.tensors
        N, J, V, Q, S = params.batch_size, params.max_sent_size, params.vocab_size, params.max_ques_size, params.max_num_sups
        d = params.hidden_size
        with tf.name_scope("placeholders"):
            x = tf.placeholder('int32', shape=[N, S, J], name='x')
            x_mask = tf.placeholder('bool', shape=[N, S, J], name='x_mask')
            q = tf.placeholder('int32', shape=[N, J], name='q')
            q_mask = tf.placeholder('bool', shape=[N, J], name='q_mask')
            y = tf.placeholder('int32', shape=[N, V], name='y')
            placeholders['x'] = x
            placeholders['x_mask'] = x_mask
            placeholders['q'] = q
            placeholders['q_mask'] = q_mask
            placeholders['y'] = y

        with tf.variable_scope("embedding"):
            A = tf.get_variable("A", dtype='float', shape=[V, d])
            Aq = tf.nn.embedding_lookup(A, q, name='Aq')  # [N, J, d]
            C = tf.get_variable("C", dtype='float', shape=[V, d])
            Cx = tf.nn.embedding_lookup(C, x, name='Ax')  # [N, S, J, d]

        with tf.name_scope("encoding"):
            l_tensor = self._get_l_tensor()  # [J, d]
            f = tf.reduce_sum(Cx * l_tensor * tf.expand_dims(tf.cast(x_mask, 'float'), -1), 2, name='f')  # [N, S, d]
            f_sum = tf.reduce_sum(f, 1, name='f_sum')  # [N, d]
            u = tf.reduce_sum(Aq * l_tensor * tf.expand_dims(tf.cast(q_mask, 'float'), -1), 1, name='u')  # [N, d]
            u_f_sum = tf.add(f_sum, u, name='sum')

        with tf.name_scope("class"):
            W = tf.transpose(A, name='W')
            logits = tf.matmul(u_f_sum, W, name='logits')
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            tensors['correct'] = correct

        with tf.name_scope("loss") as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(y, 'float'), name='cross_entropy')
            avg_ce = tf.reduce_mean(cross_entropy, name='avg_ce')
            tf.add_to_collection('losses', avg_ce)
            losses = tf.get_collection('losses', scope=scope)
            loss = tf.add_n(losses, name='loss')
            tensors['loss'] = loss

    def _get_l_tensor(self, name='l'):
        params = self.params
        J, d = params.max_sent_size, params.hidden_size
        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)
        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]
        l = [g(j) for j in range(J)]
        l_tensor = tf.constant(l, shape=[J, d], name=name)
        return l_tensor

    def get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        N, J, V, S = params.batch_size, params.max_sent_size, params.vocab_size, params.max_num_sups
        x = np.zeros([N, S, J], dtype='int32')
        x_mask = np.zeros([N, S, J], dtype='bool')
        q = np.zeros([N, J], dtype='int32')
        q_mask = np.zeros([N, J], dtype='bool')
        y = np.zeros([N, V], dtype='bool')

        ph = self.placeholders
        feed_dict = {ph['x']: x, ph['x_mask']: x_mask,
                     ph['q']: q, ph['q_mask']: q_mask,
                     ph['y']: y}
        if batch is None:
            return feed_dict

        X, Q, S, Y = batch
        for i, (para, supports) in enumerate(zip(X, S)):
            for j, support in enumerate(supports):
                sent = para[support]
                for k, word in enumerate(sent):
                    x[i, j, k] = word
                    x_mask[i, j, k] = True
        for i, ques in enumerate(Q):
            for j, word in enumerate(ques):
                q[i, j] = word
                q_mask[i, j] = True

        for i, ans in enumerate(Y):
            y[i, ans] = True
        return feed_dict