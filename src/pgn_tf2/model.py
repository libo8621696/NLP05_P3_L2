# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19
import tensorflow as tf

from src.pgn_tf2.layers import Encoder, Decoder, Pointer, BahdanauAttention
from collections import defaultdict
from src.utils.config import vocab_path
from src.utils.gpu_utils import config_gpu
from src.utils.wv_loader import load_embedding_matrix, Vocab


class PGN(tf.keras.Model):
    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_embedding_matrix(max_vocab_size=params['vocab_size'])
        self.vocab_size = params["vocab_size"]
        self.batch_size = params["batch_size"]
        self.encoder = Encoder(self.embedding_matrix,
                               params["enc_units"],
                               params["batch_size"])

        self.attention = BahdanauAttention(params["attn_units"])

        self.decoder = Decoder(self.embedding_matrix,
                               params["dec_units"],
                               params["batch_size"])

        self.pointer = Pointer()

    def call_one_step(self, dec_input, dec_hidden, enc_output, enc_pad_mask, use_coverage, prev_coverage):
        context_vector, dec_hidden, dec_x, \
        prediction, attention_weights, coverage = self.decoder(dec_input,
                                                               dec_hidden,
                                                               enc_output,
                                                               enc_pad_mask,
                                                               prev_coverage,
                                                               use_coverage)
        p_gen = self.pointer(context_vector, dec_hidden, dec_x)

        return prediction, dec_hidden, context_vector, \
               attention_weights, p_gen, coverage

    def call(self, dec_input, dec_hidden, enc_output,
             enc_extended_input, batch_oov_len,
             enc_pad_mask, use_coverage, coverage=None):
        '''
        :param enc_inp:
        :param dec_input:  tf.expand_dims(dec_inp[:, t], 1)
        :param enc_extended_input:
        :param batch_oov_len:
        '''
        predictions = []
        attentions = []
        p_gens = []
        coverages = []

        for t in range(dec_input.shape[1]):
            # decoder
            # using teacher forcing
            final_dists, dec_hidden, context_vector, \
            attention_weights, p_gen, coverage = self.call_one_step(dec_input[:, t],
                                                                    dec_hidden,
                                                                    enc_output,
                                                                    enc_pad_mask[:, t],
                                                                    use_coverage,
                                                                    coverage)
            coverages.append(tf.squeeze(coverage, axis=-1))
            attentions.append(attention_weights)
            predictions.append(final_dists)
            p_gens.append(p_gen)

        final_dists = _calc_final_dist(enc_extended_input,
                                       predictions,
                                       attentions,
                                       p_gens,
                                       batch_oov_len,
                                       self.vocab_size,
                                       self.batch_size)
        return tf.stack(final_dists, 1), attentions, tf.stack(coverages, 1)


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    Calculate the final distribution, for the pointer-generator model
    Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # list length max_dec_steps of shape (batch_size, extended_vsize)
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
    # then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_vsize]
    # list length max_dec_steps (batch_size, extended_vsize)
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
    # the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                   zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab = Vocab(vocab_path)
    # 计算vocab size
    vocab_size = vocab.count
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    params = defaultdict()
    params["vocab_size"] = vocab_size
    params["embed_size"] = 300
    params["enc_units"] = 512
    params["attn_units"] = 20
    params["dec_units"] = 1024
    params["batch_size"] = 64
    params["max_enc_len"] = 200
    params["max_dec_len"] = 41
    params["max_vocab_size"] = 32000
    params["vocab_size"] = 32000

    # build model
    model = PGN(params)

    # encoder input
    enc_inp = tf.ones(shape=(params["batch_size"], params["max_enc_len"]), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(params["batch_size"], params["max_enc_len"]), dtype=tf.int32)

    # decoder input
    dec_input = tf.ones(shape=(params["batch_size"], params["max_dec_len"]), dtype=tf.int32)

    # encoder hidden
    enc_hidden = model.encoder.initialize_hidden_state()

    enc_output, enc_hidden = model.encoder(enc_inp, enc_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))

    # context_vector, attention_weights, coverage = model.attention(enc_hidden, enc_output, enc_pad_mask)
    #
    # print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    # print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    # call(self, dec_hidden, enc_output, dec_input,
    '''
    dec_input, dec_hidden, enc_output,
             enc_extended_input, batch_oov_len,
             enc_pad_mask, use_coverage, prev_coverage=None
             '''
    batch_oov_len = 0
    enc_extended_input = [[] * params["batch_size"]]
    enc_pad_mask = tf.ones(shape=(params["batch_size"], params["max_enc_len"]), dtype=tf.int32)

    dec_hidden = enc_hidden

    final_dists, attentions, coverages = model(dec_input,
                                               dec_hidden,
                                               enc_output,
                                               enc_extended_input,
                                               batch_oov_len,
                                               enc_pad_mask,
                                               use_coverage=True,
                                               coverage=None)

    print('Decoder final_dists shape: (batch_size, vocab size) {}'.format(final_dists.shape))
    print('Decoder attentions shape: (batch_size, 1,embedding_dim + units) {}'.format(attentions.shape))
    print('Decoder coverages shape: (batch_size, 1,embedding_dim + units) {}'.format(coverages.shape))

# p_gen = model.pointer(context_vector, dec_hidden, dec_inp)
# print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))
