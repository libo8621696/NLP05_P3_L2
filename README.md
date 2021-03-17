



先将AutoMaster_TrainSet 和 AutoMaster_TestSet 拷贝到data 路径下 再使用 .



代码结构

+ result 结果保存路径
    ....    
+ seq2seq_tf2 模型结构
    ....
+ utils 工具包
    + config  配置文件
    + data_loader 数据处理模块
    + multi_proc_utils 多进程数据处理
+ data  数据集
    + AutoMaster_TrainSet 拷贝数据集到该路径
    + AutoMaster_TestSet  拷贝数据集到该路径
    ....
    
    
训练步骤:
1. 拷贝数据集到data路径下
2. 运行utils\data_loader.py可以一键完成 预处理数据 构建数据集
3. 训练模型 运行seq2seq_tf2\train.py脚本


预测步骤：
1. greedy decode 和 beam search 的代码都在 predict_helper.py 中
2. 运行 predict.py 调用 greedy decode 或者 beam search 进行预测