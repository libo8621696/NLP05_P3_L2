# -*- coding:utf-8 -*-
# Created by LuoJie at 12/8/19
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from src.pgn_tf2.model import _calc_final_dist


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, hidden, attn_dists, p_gens, coverage):
        self.tokens = tokens  # list of all the tokens from time 0 to the current time step t
        self.log_probs = log_probs  # list of the log probabilities of the tokens of the tokens
        self.hidden = hidden  # decoder hidden state after the last token decoding
        self.attn_dists = attn_dists  # attention dists of all the tokens
        self.p_gens = p_gens
        self.coverage = coverage
        self.abstract = ""
        self.real_abstract = ""
        self.article = ""

    def extend(self, token, log_prob, hidden, attn_dist, p_gen, coverage):
        """Method to extend the current hypothesis by adding the next decoded token and all the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          hidden=hidden,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def print_top_k(hyp, k, vocab, batch):
    text = batch[0]["article"].numpy()[0].decode()
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    print('\nhyp.text :{}'.format(text))
    for i in range(min(k, len(hyp))):
        k_hyp = hyp[i]
        k_hyp = result_index2text(k_hyp, vocab, batch)
        print('top {} best_hyp.abstract :{}\n'.format(i, k_hyp.abstract))


def result_index2text(hyp, vocab, batch):
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    hyp.article = batch[0]["article"].numpy()[0].decode()

    words = []
    for index in hyp.tokens:
        if index != vocab.START_DECODING_INDEX and index != vocab.STOP_DECODING_INDEX:
            if index < (len(article_oovs) + vocab.size()):
                if index < vocab.size():
                    words.append(vocab.id_to_word(index))
                else:
                    words.append(article_oovs[index - vocab.size()].decode())
            else:
                print('error values id :{}'.format(index))
    hyp.abstract = " ".join(words)
    return hyp


def beam_decode(model, batch, vocab, params, print_info=False):
    # 初始化mask
    start_index = vocab.word_to_id(vocab.START_DECODING)
    stop_index = vocab.word_to_id(vocab.STOP_DECODING)
    unk_index = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
    batch_size = params['batch_size']

    # 单步decoder
    def decoder_one_step(enc_output, dec_input, dec_hidden, enc_extended_inp, batch_oov_len, enc_pad_mask,
                         use_coverage, prev_coverage):
        # 单个时间步 运行
        prediction, dec_hidden, context_vector, \
        attention_weights, p_gens, coverage = model.call_one_step(dec_input,
                                                                  dec_hidden,
                                                                  enc_output,
                                                                  enc_pad_mask,
                                                                  use_coverage=use_coverage,
                                                                  prev_coverage=prev_coverage)

        final_dist = _calc_final_dist(enc_extended_inp,
                                      [prediction],
                                      [attention_weights],
                                      [p_gens],
                                      batch_oov_len,
                                      params['vocab_size'],
                                      params['batch_size'])

        # 拿到top k个index 和 概率
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dist), k=params["beam_size"] * 2)
        # 计算log概率
        top_k_log_probs = tf.math.log(top_k_probs)

        results = {
            'coverage_ret': coverage,
            "last_context_vector": context_vector,
            "dec_hidden": dec_hidden,
            "attention_weights": attention_weights,
            "top_k_ids": top_k_ids,
            "top_k_log_probs": top_k_log_probs,
            "p_gens": p_gens}

        # 返回需要保存的中间结果和概率
        return results

    # 测试数据的输入
    enc_input = batch[0]["enc_input"]
    enc_extended_inp = batch[0]["extended_enc_input"]
    batch_oov_len = batch[0]["max_oov_len"]
    enc_pad_mask = batch[0]["encoder_pad_mask"]
    article_oovs = batch[0]["article_oovs"][0]

    # 计算第encoder的输出
    enc_hidden = model.encoder.initialize_hidden_state()
    enc_output, enc_hidden = model.encoder(enc_input, enc_hidden)

    # 初始化batch size个 假设对象
    hyps = [Hypothesis(tokens=[start_index],
                       log_probs=[0.0],
                       hidden=enc_hidden[0],
                       attn_dists=[],
                       p_gens=[],
                       # zero vector of length attention_length
                       coverage=np.zeros([enc_input.shape[1], 1], dtype=np.float32)) for _ in range(batch_size)]
    # 初始化结果集
    results = []  # list to hold the top beam_size hypothesises
    # 遍历步数
    steps = 0  # initial step

    # 长度还不够 并且 结果还不够 继续搜索
    while steps < params['max_dec_len'] and len(results) < params['beam_size']:
        # 获取最新待使用的token
        latest_tokens = [h.latest_token for h in hyps]
        # 替换掉 oov token unknown token
        latest_tokens = [t if t in vocab.id2word else unk_index for t in latest_tokens]

        # 获取所以隐藏层状态
        hiddens = [h.hidden for h in hyps]

        prev_coverage = [h.coverage for h in hyps]
        prev_coverage = tf.convert_to_tensor(prev_coverage)

        # dec_input = tf.expand_dims(latest_tokens, axis=1)
        dec_input = tf.stack(latest_tokens)
        dec_hidden = tf.stack(hiddens, axis=0)

        # 单步运行decoder 计算需要的值
        decoder_results = decoder_one_step(enc_output,
                                           dec_input,
                                           dec_hidden,
                                           enc_extended_inp,
                                           batch_oov_len,
                                           enc_pad_mask,
                                           use_coverage=params['use_coverage'],
                                           prev_coverage=prev_coverage)

        dec_hidden = decoder_results['dec_hidden']
        attention_weights = decoder_results['attention_weights']
        top_k_log_probs = decoder_results['top_k_log_probs']
        top_k_ids = decoder_results['top_k_ids']
        new_coverage = decoder_results['coverage_ret']
        p_gens = decoder_results['p_gens']

        # 现阶段全部可能情况
        all_hyps = []
        # 原有的可能情况数量 TODO
        num_orig_hyps = 1 if steps == 0 else len(hyps)

        # 遍历添加所有可能结果
        for i in range(num_orig_hyps):
            h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
            if params['pointer_gen']:
                p_gen = p_gens[i]
            else:
                p_gen = 0
            if params['use_coverage']:
                new_coverage_i = new_coverage[i]
            else:
                new_coverage_i = 0

            # 分裂 添加 beam size 种可能性
            for j in range(params['beam_size'] * 2):
                # 构造可能的情况
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=top_k_log_probs[i, j],
                                   hidden=new_hidden,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                # 添加可能情况
                all_hyps.append(new_hyp)

        # 重置
        hyps = []
        # 按照概率来排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)

        # 筛选top前beam_size句话
        for h in sorted_hyps:
            if h.latest_token == stop_index:
                # 长度符合预期,遇到句尾,添加到结果集
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # 未到结束 ,添加到假设集
                hyps.append(h)

            # 如果假设句子正好等于beam_size 或者结果集正好等于beam_size 就不在添加
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break

        steps += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)

    best_hyp = hyps_sorted[0]
    # best_hyp.abstract = " ".join([vocab.id_to_word(index) for index in best_hyp.tokens])
    # best_hyp.text = batch[0]["article"].numpy()[0].decode()

    best_hyp = result_index2text(best_hyp, vocab, batch)

    if print_info:
        print_top_k(hyps_sorted, params['beam_size'], vocab, batch)
        print('real_article: {}'.format(best_hyp.real_abstract))
        print('article: {}'.format(best_hyp.abstract))

    best_hyp = result_index2text(best_hyp, vocab, batch)
    return best_hyp


def batch_greedy_decode(model, encoder_batch_data, vocab, params):
    # 判断输入长度
    batch_data = encoder_batch_data["enc_input"]
    batch_size = encoder_batch_data["enc_input"].shape[0]

    article_oovs = encoder_batch_data["article_oovs"].numpy()
    # 开辟结果存储list
    predicts = [''] * batch_size
    enc_input = batch_data
    # print(batch_size, batch_data.shape)

    # encoder
    enc_hidden = model.encoder.initialize_hidden_state()
    enc_output, enc_hidden = model.encoder(enc_input, enc_hidden)

    dec_hidden = enc_hidden
    # dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)
    dec_input = tf.constant([vocab.START_DECODING_INDEX] * batch_size)

    # Teacher forcing - feeding the target as the next input

    try:
        batch_oov_len = tf.shape(encoder_batch_data["article_oovs"])[1]
    except:
        batch_oov_len = tf.constant(0)

    coverage = None
    for t in range(params['max_dec_len']):
        # 单步预测
        # final_dist (batch_size, 1, vocab_size+batch_oov_len)
        prediction, dec_hidden, context_vector, \
        attention_weights, p_gen, coverage = model.call_one_step(dec_input,
                                                                 dec_hidden,
                                                                 enc_output,
                                                                 encoder_batch_data["encoder_pad_mask"],
                                                                 use_coverage=True,
                                                                 prev_coverage=coverage)

        final_dist = _calc_final_dist(encoder_batch_data["extended_enc_input"],
                                      [prediction],
                                      [attention_weights],
                                      [p_gen],
                                      batch_oov_len,
                                      params['vocab_size'],
                                      params['batch_size'])

        # id转换
        predicted_ids = tf.argmax(final_dist[0], axis=-1)

        inp_predicts = []
        for index, predicted_id in enumerate(predicted_ids.numpy()):
            if predicted_id >= vocab.count:
                # OOV词
                word = article_oovs[index][predicted_id - vocab.count].decode()
                inp_predicts.append(vocab.UNKNOWN_TOKEN_INDEX)
            else:
                word = vocab.id_to_word(predicted_id)
                inp_predicts.append(predicted_id)
            predicts[index] += word + ' '

        predicted_ids = np.array(inp_predicts)
        # using teacher forcing
        dec_input = predicted_ids

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if vocab.STOP_DECODING in predict:
            # 截断stop
            predict = predict[:predict.index(vocab.STOP_DECODING)]
        # 保存结果
        results.append(predict)
    return results


def greedy_decode(model, dataset, vocab, params):
    # 存储结果
    # batch_size = params["batch_size"]
    results = []

    # sample_size = 20000
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    # steps_epoch = sample_size // batch_size + 1
    # [0,steps_epoch)
    ds = iter(dataset)
    for i in tqdm(range(params['steps_per_epoch'])):
        enc_data, _ = next(ds)
        batch_results = batch_greedy_decode(model, enc_data, vocab, params)
        print(batch_results[0])
        results.extend(batch_results)
    return results
