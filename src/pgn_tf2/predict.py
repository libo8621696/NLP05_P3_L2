# -*- coding:utf-8 -*-
# Created by LuoJie at 12/12/19
import tensorflow as tf
from src.pgn_tf2.batcher import batcher
from src.pgn_tf2.model import PGN
from tqdm import tqdm
import pandas as pd
from src.pgn_tf2.predict_helper import beam_decode, greedy_decode
from src.utils.config import checkpoint_dir, test_data_path, test_seg_path
from src.utils.gpu_utils import config_gpu
from src.utils.wv_loader import Vocab
from src.utils.params_utils import get_params
import json
from rouge import Rouge


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    if params['decode_mode'] == 'beam':
        assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    config_gpu(use_cpu=True)

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    results = predict_result(model, params, vocab, params['result_save_path'])
    print('save result to :{}'.format(params['result_save_path']))
    print('save result :{}'.format(results[:5]))


def get_rouge(results):
    # 读取结果
    seg_test_report = pd.read_csv(test_seg_path).iloc[:100, 5].tolist()
    rouge_scores = Rouge().get_scores(results, seg_test_report, avg=True)
    print_rouge = json.dumps(rouge_scores, indent=2)
    print('*' * 8 + ' rouge score ' + '*' * 8)
    print(print_rouge)


def predict_result(model, params, vocab, result_save_path):
    dataset, _ = batcher(vocab, params)

    if params['decode_mode'] == 'beam':
        results = []
        for batch in tqdm(dataset):
            best_hyp = beam_decode(model, batch, vocab, params, print_info=True)
            results.append(best_hyp.abstract)
    else:
        # 预测结果
        results = greedy_decode(model, dataset, vocab, params)
    get_rouge(results)
    # 保存结果
    if not os.path.exists(os.path.dirname(result_save_path)):
        os.makedirs(os.path.dirname(result_save_path))
    save_predict_result(results, result_save_path)

    return results


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(test_data_path).iloc[:100]
    # 填充结果
    test_df['Prediction'] = results[:len(test_df['QID'])]
    # 提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 获得参数
    params = get_params()

    # beam search
    params['batch_size'] = 3
    params['beam_size'] = 3
    params['mode'] = 'test'
    params['decode_mode'] = 'beam'
    params['pointer_gen'] = True
    params['use_coverage'] = False
    params['enc_units'] = 128
    params['dec_units'] = 256
    params['attn_units'] = 20
    params['min_dec_steps'] = 3

    # greedy search
    # params['batch_size'] = 8
    # params['mode'] = 'test'
    # params['decode_mode'] = 'greedy'
    # params['pointer_gen'] = True
    # params['use_coverage'] = False
    # params['enc_units'] = 256
    # params['dec_units'] = 512
    # params['attn_units'] = 256
    # params['min_dec_steps'] = 3

    test(params)
