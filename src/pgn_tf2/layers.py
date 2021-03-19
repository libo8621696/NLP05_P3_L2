# -*- coding:utf-8 -*-
# Created by LuoJie at 11/23/19

from src.utils.config import save_wv_model_path, vocab_path
import tensorflow as tf
from src.utils.gpu_utils import config_gpu
from tensorflow.keras.models import Model
import tensorflow as tf
from src.utils.wv_loader import load_embedding_matrix, Vocab


class Encoder(tf.keras.Model):
    def __init__(self, embedding_matrix, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bidirectional_gru = tf.keras.layers.Bidirectional(self.gru)

    def call(self, x, enc_hidden):
        x = self.embedding(x)

        # output, hidden = self.gru(x, initial_state=hidden)
        enc_output, forward_state, backward_state = self.bidirectional_gru(x, initial_state=[enc_hidden, enc_hidden])
        enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        return enc_output, enc_hidden

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


def masked_attention(enc_padding_mask, attn_dist):
    """
    Take softmax of e then apply enc_padding_mask and re-normalize
    """
    attn_dist = tf.squeeze(attn_dist, axis=2)
    mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
    # mask = tf.reshape(mask, [-1, 1])
    attn_dist *= mask  # apply mask
    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
    attn_dist = attn_dist / tf.reshape(masked_sums + 1e-12, [-1, 1])  # re-normalize
    # attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
    attn_dist = tf.expand_dims(attn_dist, axis=2)
    return attn_dist


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=False, prev_coverage=None):
        """
         calculate attention and coverage from dec_hidden enc_output and prev_coverage
         one dec_hidden(word) by one dec_hidden
         dec_hidden or query is [batch_sz, enc_unit], enc_output or values is [batch_sz, max_train_x, enc_units],
         prev_coverage is [batch_sz, max_len_x, 1]
         dec_hidden is initialized as enc_hidden, prev_coverage is initialized as None
         output context_vector [batch_sz, enc_units] attention_weights & coverage [batch_sz, max_len_x, 1]
         """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # Multiply coverage vector by w_c to get coverage_features.
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, units]
            # self.W_c(prev_coverage) [batch_sz, max_len, units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)

            # attention_weights sha== (batch_size, max_length, 1)
            # mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            # masked_score = tf.squeeze(score, axis=-1) * mask
            # masked_score = tf.expand_dims(masked_score, axis=2)

            attention_weights = tf.nn.softmax(score, axis=1)

            attention_weights = masked_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights + prev_coverage
        else:
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(
                self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            attention_weights = tf.nn.softmax(score, axis=1)
            attention_weights = masked_attention(enc_pad_mask, attention_weights)
            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attention_weights, -1), coverage


class Decoder(tf.keras.Model):
    def __init__(self, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)

        self.cell = tf.keras.layers.GRUCell(units=self.dec_units,
                                            recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(self.vocab_size, activation=tf.keras.activations.softmax)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, dec_inputs, dec_hidden, enc_output, enc_pad_mask, prev_coverage, use_coverage=True):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # print('x:{}'.format(x))
        dec_x = self.embedding(dec_inputs)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入

        dec_output, [dec_hidden] = self.cell(dec_x, [dec_hidden])

        context_vector, attention_weights, coverage = self.attention(dec_hidden,
                                                                     enc_output,
                                                                     enc_pad_mask,
                                                                     use_coverage,
                                                                     prev_coverage)

        dec_output = tf.concat([dec_output, context_vector], axis=-1)

        # output shape == (batch_size, vocab)
        prediction = self.fc(dec_output)
        """
        output
        output[0]: context_vector (batch_size, dec_units)
        output[1]: dec_hidden (batch_size, dec_units)
        output[2]: dec_x (batch_size, embedding_dim)
        output[3]: pred (batch_size, vocab_size)
        output[4]: attn (batch_size, enc_len)
        output[5]: coverage (batch_size, enc_len, 1)
         """
        return context_vector, dec_hidden, dec_x, prediction, attention_weights, coverage


class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):
        # change dec_inp_context to [batch_sz,embedding_dim+enc_units]
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) +
                             self.w_c_reduce(context_vector) +
                             self.w_i_reduce(dec_inp))


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab = Vocab(vocab_path)
    # 计算vocab size
    vocab_size = vocab.count

    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    enc_max_len = 200
    dec_max_len = 41
    batch_size = 64
    embedding_dim = 300
    enc_units = 512
    dec_units = 1024
    att_units = 20

    # 编码器结构
    encoder = Encoder(embedding_matrix, enc_units, batch_size)
    # encoder input
    enc_inp = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)
    # decoder input
    dec_inp = tf.ones(shape=(batch_size, dec_max_len), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)

    # encoder hidden
    enc_hidden = encoder.initialize_hidden_state()
    # encoder hidden
    enc_output, enc_hidden = encoder(enc_inp, enc_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))

    # dec_hidden, enc_output, enc_pad_mask, use_coverage = False, prev_coverage = None)

    dec_hidden = enc_hidden
    attention_layer = BahdanauAttention(att_units)
    context_vector, attention_weights, coverage = attention_layer(dec_hidden, enc_output, enc_pad_mask,
                                                                  use_coverage=True, prev_coverage=None)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length) {}".format(attention_weights.shape))
    print("Attention coverage: (batch_size,sequence_length ) {}".format(coverage.shape))

    decoder = Decoder(embedding_matrix, dec_units, batch_size)

    prev_dec_hidden = enc_hidden
    prev_coverage = coverage

    context_vector, dec_hidden, dec_x, prediction, attention_weights, coverage = decoder(dec_inp[:, 0],
                                                                                         prev_dec_hidden,
                                                                                         enc_output,
                                                                                         enc_pad_mask,
                                                                                         prev_coverage,
                                                                                         use_coverage=True)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(prediction.shape))
    print('Decoder dec_x shape: (batch_size, embedding_dim) {}'.format(dec_x.shape))
    print('Decoder context_vector shape: (batch_size, 1,dec_units) {}'.format(context_vector.shape))
    print('Decoder attention_weights shape: (batch_size, sequence_length) {}'.format(attention_weights.shape))
    print('Decoder dec_hidden shape: (batch_size, dec_units) {}'.format(dec_hidden.shape))

    pointer = Pointer()
    p_gen = pointer(context_vector, dec_hidden, dec_x)
    print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))
