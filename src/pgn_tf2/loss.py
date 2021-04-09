# -*- coding:utf-8 -*-
# Created by LuoJie at 12/14/19
import tensorflow as tf

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def calc_loss(real, pred, dec_mask, attentions, cov_loss_wt, eps):
    log_loss = pgn_log_loss_function(real, pred, dec_mask, eps)
    # log_loss = _log_loss(real, pred, dec_mask)
    # cov_loss = _coverage_loss(attentions, dec_mask)
    # return log_loss + cov_loss_wt * cov_loss, log_loss, cov_loss
    return log_loss, 0, 0


def _log_loss(target, pred, dec_mask):
    """
    计算log_loss
    :param target: shape (batch_size, dec_len)
    :param pred:  shape (batch_size, dec_len, vocab_size)
    :param dec_mask: shape (batch_size, dec_len)
    :return: log loss
    """
    loss_ = loss_object(target, pred)
    # 注batcher产生padding_mask时，数据类型需要指定成tf.float32可以少下面这行代码
    dec_mask = tf.cast(dec_mask, dtype=loss_.dtype)
    loss_ *= dec_mask
    loss_ = tf.reduce_mean(loss_)
    return loss_


def pgn_log_loss_function(real, final_dists, padding_mask, eps):
    # Calculate the loss per step
    # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
    loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
    batch_nums = tf.range(0, limit=real.shape[0])  # shape (batch_size)
    final_dists = tf.transpose(final_dists, perm=[1, 0, 2])
    for dec_step, dist in enumerate(final_dists):
        # The indices of the target words. shape (batch_size)
        targets = real[:, dec_step]
        indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
        gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
        losses = -tf.math.log(gold_probs + eps)
        loss_per_step.append(losses)
    # Apply dec_padding_mask and get loss
    _loss = _mask_and_avg(loss_per_step, padding_mask)
    return _loss


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

    Returns:
      a scalar
    """
    padding_mask = tf.cast(padding_mask, dtype=values[0].dtype)
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex)  # overall average


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    attn_dists = tf.transpose(attn_dists, perm=[1, 0, 2])
    coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
    # Coverage loss per decoder time step. Will be list length max_dec_steps containing shape (batch_size).
    covlosses = []
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss
