import tensorflow as tf
from IPython import embed
from dialog.base_model import BaseTower, BaseRunner
from my.tensorflow.nn import linear
import numpy as np


class Embedder(object):
    def __call__(self, content):
        raise Exception()


class VariableEmbedder(Embedder):
    def __init__(self, params, wd=0.0, initializer=None, name="variable_embedder"):
        V, d = params.vocab_size[0], params.hidden_size
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
            # l = self.b + self.w/length_aug
            l = self.b + self.w/self.max_sent_size
            mask_aug = tf.expand_dims(mask, -1)
            mask_aug_cast = tf.cast(mask_aug, 'float')
            l_cast = tf.cast(l, 'float')
            f = tf.reduce_sum(Ax * l_cast * mask_aug_cast, length_dim_index, name='f')  # [N, S, d]

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


class ReductionLayer(object):
    def __init__(self, batch_size, mem_size, hidden_size):
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.batch_size = batch_size
        N, M, d = batch_size, mem_size, hidden_size
        self.L = np.tril(np.ones([M, M], dtype='float32'))
        self.sL = np.tril(np.ones([M, M], dtype='float32'), k=-1)

    def __call__(self, u_t, a, b, scope=None):
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
            L = tf.tile(tf.expand_dims(L, 0), [N, 1, 1])
            sL = tf.tile(tf.expand_dims(sL, 0), [N, 1, 1])
            logb = tf.log(b + 1e-9)
            logb = tf.concat(1, [tf.zeros([N, 1, 1]), tf.slice(logb, [0, 1, 0], [-1, -1, -1])])
            left = L * tf.exp(tf.batch_matmul(L, logb * sL))  # [N, M, M]
            right = a * u_t  # [N, M, d]
            u = tf.batch_matmul(left, right)  # [N, M, d]
        return u


class VectorReductionLayer(object):
    def __init__(self, batch_size, mem_size, hidden_size):
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.batch_size = batch_size
        N, M, d = batch_size, mem_size, hidden_size
        self.L = np.tril(np.ones([M, M], dtype='float32'))
        self.sL = np.tril(np.ones([M, M], dtype='float32'), k=-1)

    def __call__(self, u_t, a, b, scope=None):
        """

        :param u_t: [N, M, d]
        :param a: [N, M. d]
        :param b: [N, M. d]
        :param mask:  [N, M]
        :return:
        """
        N, M, d = self.batch_size, self.mem_size, self.hidden_size
        L, sL = self.L, self.sL
        with tf.name_scope(scope or self.__class__.__name__):
            L = tf.tile(tf.expand_dims(tf.expand_dims(L, 0), 0), [N, d, 1, 1])
            sL = tf.tile(tf.expand_dims(tf.expand_dims(sL, 0), 0), [N, d, 1, 1])
            logb = tf.log(b + 1e-9)  # [N, M, d]
            logb = tf.concat(1, [tf.zeros([N, 1, d]), tf.slice(logb, [0, 1, 0], [-1, -1, -1])])  # [N, M, d]
            logb = tf.expand_dims(tf.transpose(logb, [0, 2, 1]), -1)  # [N, d, M, 1]
            left = L * tf.exp(tf.batch_matmul(L, logb * sL))  # [N, d, M, M]
            right = a * u_t  # [N, M, d]
            right = tf.expand_dims(tf.transpose(right, [0, 2, 1]), -1)  # [N, d, M, 1]
            u = tf.batch_matmul(left, right)  # [N, d, M, 1]
            u = tf.transpose(tf.squeeze(u, [3]), [0, 2, 1])  # [N, M, d]
        return u


class Tower(BaseTower):
    def initialize(self):
        params = self.params
        placeholders = self.placeholders
        tensors = self.tensors
        variables_dict = self.variables_dict

        self.task = int(params.task)
        self.dstc = self.task%10 == 6
        self.match = params.use_match
        self.rnn = params.use_rnn
	

        N, J, Q, M = params.batch_size, params.max_sent_size, params.max_ques_size, params.mem_size
        V, Alist = params.vocab_size

        d = params.hidden_size
        L = params.mem_num_layers
        att_forget_bias = params.att_forget_bias
        use_vector_gate = params.use_vector_gate
        wd = params.wd
        initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))
	
        self.ans_dic = {
		1 : range(5), 2 : range(5),
		3 : [0, 5], 4 : [0,6,7], 5 : range(8), 6 : range(11)
	}
        self.num_candidate = Alist[0]+1
        data_task = self.task%10 if not self.rnn else self.task
        self.ans = self.ans_dic.get(data_task, [0])
        self.num_ans = len(self.ans)
        if self.rnn and self.task==3 : self.num_ans = 3
        elif self.rnn and self.task==4: self.num_ans = 4
        elif self.rnn: self.num_ans = 6


        with tf.name_scope("placeholders"):
            x = tf.placeholder('int32', shape=[N, M, J], name='x')
            x_mask = tf.placeholder('bool', shape=[N, M, J], name='x_mask')
            q = tf.placeholder('int32', shape=[N, J], name='q')
            q_mask = tf.placeholder('bool', shape=[N, J], name='q_mask')
            y = tf.placeholder('int32', shape=[N, self.num_ans], name='y')
            y_mask = tf.placeholder('bool', shape=[N, self.num_ans], name='y_mask')
            y_feats = []
            for i in self.ans[1:]:
                A = Alist[0] if self.rnn else Alist[i]
                y_feats.append(tf.placeholder('int32', shape=[N, 2, A], name='y_feat'+str(i)))
            self.y_state_dim = self.num_ans-2 if self.rnn else self.num_ans-1
            y_state =tf.placeholder('bool', shape=[N, self.y_state_dim], name='y_state')
            is_train = tf.placeholder('bool', shape=[], name='is_train')
            
            placeholders['x'] = x
            placeholders['x_mask'] = x_mask
            placeholders['q'] = q
            placeholders['q_mask'] = q_mask
            placeholders['y'] = y
            placeholders['y_mask'] = y_mask
            placeholders['y_feats'] = y_feats
            placeholders['y_state'] = y_state
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
            gate_mask = tf.expand_dims(m_mask, -1)
            m_length = tf.reduce_sum(m_mask, 1, name='m_length')  # [N]
            prev_u = tf.tile(tf.expand_dims(u, 1), [1, M, 1])  # [N, M, d]
            reg_layer = VectorReductionLayer(N, M, d) if use_vector_gate else ReductionLayer(N, M, d)
            gate_size = d if use_vector_gate else 1
            h = None  # [N, M, d]
            as_, rfs, rbs = [], [], []
            hs = []
            for layer_idx in range(L):
                with tf.name_scope("layer_{}".format(layer_idx)):
                    u_t = tf.tanh(linear([prev_u, m], d, True, wd=wd, scope='u_t'))
                    a = tf.cast(gate_mask, 'float') * tf.sigmoid(linear([prev_u * m], gate_size, True, initializer=initializer, wd=wd, scope='a') - att_forget_bias)
                    h = reg_layer(u_t, a, 1.0-a, scope='h')
                    if layer_idx + 1 < L:
                        if params.use_reset:
                            rf, rb = tf.split(2, 2, tf.cast(gate_mask, 'float') *
                                tf.sigmoid(linear([prev_u * m], 2 * gate_size, True, initializer=initializer, wd=wd, scope='r')))
                        else:
                            rf = rb = tf.ones(a.get_shape().as_list())
                        u_t_rev = tf.reverse_sequence(u_t, m_length, 1)
                        a_rev, rb_rev = tf.reverse_sequence(a, m_length, 1), tf.reverse_sequence(rb, m_length, 1)
                        uf = reg_layer(u_t, a*rf, 1.0-a, scope='uf')
                        ub_rev = reg_layer(u_t_rev, a_rev*rb_rev, 1.0-a_rev, scope='ub_rev')
                        ub = tf.reverse_sequence(ub_rev, m_length, 1)
                        prev_u = uf + ub
                    else:
                        rf = rb = tf.zeros(a.get_shape().as_list())
                    rfs.append(rf)
                    rbs.append(rb)
                    as_.append(a)
                    hs.append(h)
                    tf.get_variable_scope().reuse_variables()

            h_last = tf.squeeze(tf.slice(h, [0, M-1, 0], [-1, -1, -1]), [1])  # [N, d]
            hs_last = [tf.squeeze(tf.slice(each, [0, M-1, 0], [-1, -1, -1]), [1]) for each in hs]
            a = tf.transpose(tf.pack(as_, name='a'), [1, 0, 2, 3])
            rf = tf.transpose(tf.pack(rfs, name='rf'), [1, 0, 2, 3])
            rb = tf.transpose(tf.pack(rbs, name='rb'), [1, 0, 2, 3])
            tensors['a'] = a
            tensors['rf'] = rf
            tensors['rb'] = rb

        with tf.variable_scope("class"):
            class_mode = params.class_mode
            use_class_bias = params.use_class_bias
            logits = []
            drop_rate = tf.cond(is_train, lambda: tf.constant(0.5),
				lambda: tf.constant(1.0))

            if class_mode == 'h':

                if self.rnn: # rnn decoder
                    hiddens = [] # previous hidden vector
                    A = self.num_candidate
                    for i in range(self.num_ans):
                        # Inverse Embedding Matrix of Answers [A, A]
                        E_inv = tf.get_variable("E_inv", [A, A], initializer = tf.constant_initializer(0.0))
                        prev_h = h_last
                        if i==0:
                            # If it is the first answer, use initial y
                            prev_y = tf.reshape(tf.tile(tf.get_variable("Wx", A, initializer = tf.constant_initializer(0.0)), [N]), [N, A])
                        else:
                            # Otherwise, use Inverse Embedding Matrix
                            _prev_y = tf.reshape(tf.gather(tf.transpose(y), i-1), [N])
                            prev_y = tf.nn.embedding_lookup(E_inv, _prev_y)
                            #prev_h = hiddens[-1]
                        _logit = linear([prev_h], A, use_class_bias, wd=wd, name='0')
                        logit = _logit * prev_y
                        hiddens.append(S2)
                        logits.append(S2)
                        
                        tf.get_variable_scope().reuse_variables()
                else:
                    if self.match:
                        # Input of softmax when using match
                        all_y_feats = [None] + y_feats
                        all_y_states = [y_state] + [None]*(len(all_y_feats)-1)

                    for i, j in enumerate(self.ans):
                        if self.match:
                            logits.append(linear([h_last], Alist[j], use_class_bias, wd=wd, name=str(i), feat = all_y_feats[i], state = all_y_states[i], drop_rate = drop_rate))

                        else:
                            logits.append(linear([h_last], Alist[j], use_class_bias, wd=wd, name=str(i) ))
            elif class_mode == 'uh':
                logits = linear([h_last, u], A, use_class_bias, wd=wd)
            elif class_mode == 'hs':
                logits = linear(hs_last, A, use_class_bias, wd=wd)
            elif class_mode == 'hss':
                logits = linear(sum(hs_last), A, use_class_bias, wd=wd)
            else:
                raise Exception("Invalid class mode: {}".format(class_mode))

	    
            for i in range(self.num_ans):
                yp_each = tf.cast(tf.expand_dims(tf.argmax(logits[i], 1), 1), 'int32')
                if i == 0: yp = yp_each
                else: yp = tf.concat(1, [yp, yp_each])
	    
            correct_ = tf.cast(tf.equal(yp, y), 'float')
            correct_sum = tf.reduce_sum(correct_ * tf.cast(y_mask, 'float'), 1)
            mask_ = tf.reduce_sum(tf.cast(y_mask, 'float'), 1)
            correct = tf.truediv(correct_sum, mask_)
            tensors['yp'] = yp
            tensors['correct_'] = correct_
            tensors['mask_'] = mask_
            tensors['y_mask'] = y_mask
            tensors['y'] = y
            tensors['correct'] = correct
            tensors['q'] = q
            if self.task>20:
                tensors['y_state'] = y_state
                for i, j in enumerate(self.ans[1:]):
                    tensors['y_feat'+str(i)] = tf.reshape(y_feats[i], [N, 2*self.ans_num[j]])

        with tf.name_scope("loss"):
            with tf.name_scope("ans_loss"):
                tot_ce = 0

                for i in range(self.num_ans):
                    _y = tf.gather(tf.transpose(y), i)
                    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], _y)
                    m = tf.cast(tf.gather(tf.transpose(y_mask), i), 'float32')
                    tot_ce += tf.reduce_sum(ce*m, name='avg_ce')

                tf.add_to_collection('losses', tot_ce)

            losses = tf.get_collection('losses')
            loss = tf.add_n(losses, name='loss')
            tensors['loss'] = loss

        variables_dict['all'] = tf.trainable_variables()

    def get_feed_dict(self, batch, mode, **kwargs):
        params = self.params
        N, J, V, M = params.batch_size, params.max_sent_size, params.vocab_size, params.mem_size
	
        Alist = params.vocab_size[1]
        if self.match:
            X, Q, Y, TCA, TCL, Task = batch
        else:
            X, Q, Y, Task = batch


        x = np.zeros([N, M, J], dtype='int32')
        x_mask = np.zeros([N, M, J], dtype='bool')
        q = np.zeros([N, J], dtype='int32')
        q_mask = np.zeros([N, J], dtype='bool')
        y = np.zeros([N, self.num_ans], dtype='int32')
        y_mask = np.zeros([N, self.num_ans], dtype='bool')
        y_feats = []
        if self.match:
            for i in self.ans[1:]:
                y_feats.append(np.zeros([N, 2, Alist[i]], dtype='int'))
            y_state = np.zeros([N, self.y_state_dim], dtype='bool')
	
        ph = self.placeholders
        feed_dict = {ph['x']: x, ph['x_mask']: x_mask,
                     ph['q']: q, ph['q_mask']: q_mask,
                     ph['y']: y, ph['y_mask']: y_mask,
                     ph['is_train']: mode == 'train'
                     }
        if batch is None:
            return feed_dict

        for i, para in enumerate(X):
            if len(para) > M:
                para = para[-M:]
            for jj, sent in enumerate(para):
                j = jj
                for k, word in enumerate(sent):
                    x[i, j, k] = word
                    x_mask[i, j, k] = True

        for i, ques in enumerate(Q):
            for j, word in enumerate(ques):
                q[i, j] = word
                q_mask[i, j] = True

        for i in range(N):
            j = 0
            for ans in Y[i]:
                if ans is not None:
                    y[i, j] = ans
                    y_mask[i, j]= True
                    j += 1
            if self.rnn:
                y[i, j] = self.num_candidate-1
                y_mask[i, j] = True
        if self.match:
            for i, (CA, CL) in enumerate(zip(TCA, TCL)):
                for j in self.ans[1:]:
                    j_ind = self.ans.index(j)-1
                    for ca in CA[j-1]:
                        y_feats[j_ind][i][0][ca] = 1
                    if CL[j-1] is not None:
                        y_feats[j_ind][i][1][CL[j-1]] = 1
                    if not CA[j-1] == []:
                        y_state[i][j_ind] = 1
            for j in range(len(self.ans)-1):
                feed_dict[ph['y_feats'][j]] = y_feats[j]
            feed_dict[ph['y_state']] = y_state
        
        return feed_dict


class Runner(BaseRunner):
    def _get_train_op(self, **kwargs):
        return self.train_ops['all']
