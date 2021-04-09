import os
import jieba
import logging

import numpy as np
import pandas as pd

from gensim.models.word2vec import LineSentence, Word2Vec

from src.build_data.utils import pad_proc, get_max_len, transform_data, sentences_proc
from src.utils import config
from src.utils.wv_loader import Vocab
from src.utils.file_utils import save_dict
from src.utils.multi_proc_utils import parallelize, cores

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 自定义词表
jieba.load_userdict(config.user_dict)


def build_dataset(train_data_path, test_data_path):
    """数据加载+预处理
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return: 训练数据 测试数据  合并后的数据 
    """
    test_df, train_df = preprocess(train_data_path, test_data_path)

    wv_model = build_w2v()

    test_X, test_Y, train_X, train_Y = generate_dataset_cache(train_df, test_df, wv_model)
    return train_X, train_Y, test_X, test_Y


def generate_dataset_cache(train_df, test_df, wv_model):
    # 8. 分离数据和标签
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_df['X'].to_csv(config.train_x_seg_path, index=None, header=False)
    train_df['Report'].to_csv(config.train_y_seg_path, index=None, header=False)
    test_df['X'].to_csv(config.val_x_seg_path, index=None, header=False)
    test_df['Report'].to_csv(config.val_y_seg_path, index=None, header=False)
    # 9. 填充开始结束符号,未知词填充 oov, 长度填充
    # 使用GenSim训练得出的vocab
    vocab = wv_model.wv.vocab
    # 训练集X处理
    # 获取适当的最大长度
    train_x_max_len = get_max_len(train_df['X'])
    test_X_max_len = get_max_len(test_df['X'])
    X_max_len = max(train_x_max_len, test_X_max_len)
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))
    # 测试集X处理
    # 获取适当的最大长度
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))
    # 训练集Y处理
    # 获取适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))
    test_y_max_len = get_max_len(test_df['Report'])
    test_df['Y'] = test_df['Report'].apply(lambda x: pad_proc(x, test_y_max_len, vocab))
    # 10. 保存pad oov处理后的,数据和标签
    train_df['X'].to_csv(config.train_x_pad_path, index=False, header=False)
    train_df['Y'].to_csv(config.train_y_pad_path, index=False, header=False)
    test_df['X'].to_csv(config.test_x_pad_path, index=False, header=False)
    test_df['Y'].to_csv(config.test_y_pad_path, index=False, header=False)
    # print('train_x_max_len:{} ,train_y_max_len:{}'.format(X_max_len, train_y_max_len))
    # 11. 词向量再次训练
    # print('start retrain w2v model')
    # wv_model.build_vocab(LineSentence(train_x_pad_path), update=True)
    # wv_model.train(LineSentence(train_x_pad_path), epochs=1, total_examples=wv_model.corpus_count)
    #
    # print('1/3')
    # wv_model.build_vocab(LineSentence(train_y_pad_path), update=True)
    # wv_model.train(LineSentence(train_y_pad_path), epochs=1, total_examples=wv_model.corpus_count)
    #
    # print('2/3')
    # wv_model.build_vocab(LineSentence(test_x_pad_path), update=True)
    # wv_model.train(LineSentence(test_x_pad_path), epochs=1, total_examples=wv_model.corpus_count)
    # 保存词向量模型
    if not os.path.exists(os.path.dirname(config.save_wv_model_path)):
        os.makedirs(os.path.dirname(config.save_wv_model_path))
    wv_model.save(config.save_wv_model_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))
    # 12. 更新vocab
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    # 保存字典
    save_dict(config.vocab_path, vocab)
    save_dict(config.reverse_vocab_path, reverse_vocab)
    # 13. 保存词向量矩阵
    embedding_matrix = wv_model.wv.vectors
    np.save(config.embedding_matrix_path, embedding_matrix)
    # 14. 数据集转换 将词转换成索引  [<START> 方向机 重 ...] -> [2, 403, 986, 246, 231
    vocab = Vocab()
    train_ids_x = train_df['X'].apply(lambda x: transform_data(x, vocab))
    train_ids_y = train_df['Y'].apply(lambda x: transform_data(x, vocab))
    test_ids_x = test_df['X'].apply(lambda x: transform_data(x, vocab))
    test_ids_y = test_df['Y'].apply(lambda x: transform_data(x, vocab))
    # 15. 数据转换成numpy数组
    # 将索引列表转换成矩阵 [2, 403, 986, 246, 231] --> array([[2,   403,   986 , 246, 231]]
    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())
    test_Y = np.array(test_ids_y.tolist())
    # 保存数据
    np.save(config.train_x_path, train_X)
    np.save(config.train_y_path, train_Y)
    np.save(config.test_x_path, test_X)
    np.save(config.test_y_path, test_Y)
    return test_X, test_Y, train_X, train_Y


def build_w2v():
    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(config.merger_seg_path),
                        size=config.embedding_dim,
                        sg=1,
                        workers=cores,
                        iter=config.wv_train_epochs,
                        window=5,
                        min_count=5)
    return wv_model


def preprocess(train_data_path, test_data_path):
    # 1.加载数据
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))
    # 2. 空值剔除
    train_df.dropna(subset=['Report'], inplace=True)
    test_df.dropna(subset=['Report'], inplace=True)
    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)
    # 3.多线程, 批量数据处理
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)
    # 4. 合并训练测试集合
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df),
                                                                               len(test_df),
                                                                               len(merged_df)))
    # 5.保存处理好的 训练 测试集合
    train_df = train_df.drop(['merged'], axis=1)
    test_df = test_df.drop(['merged'], axis=1)
    train_df.to_csv(config.train_seg_path, index=False, header=False)
    test_df.to_csv(config.test_seg_path, index=False, header=False)
    # 6. 保存合并数据
    merged_df.to_csv(config.merger_seg_path, index=False, header=False)
    return test_df, train_df


if __name__ == '__main__':
    # 数据集批量处理
    build_dataset(config.train_data_path, config.test_data_path)
