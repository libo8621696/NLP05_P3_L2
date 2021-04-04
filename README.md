



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

##  seq2seq_tf2 模块
* 训练模型 运行seq2seq_tf2\train.py脚本,进入 summary 目录,运行如下命令:
    ```bash
    $ python -m src.seq2seq_tf2.train
    ```


预测步骤:
1. greedy decode 和 beam search 的代码都在 predict_helper.py 中，greedy 使用的是验证集损失最小的
    ```json
   {
        "rouge-1": { 
           "f": 0.31761580523281824,
           "p": 0.35095753378433,  
           "r": 0.3439340546952935
       },
       "rouge-2": {
           "f": 0.13679872398179568,
           "p": 0.14990364277693116,
           "r": 0.1455355455469621
       },
       "rouge-l": {    
           "f": 0.3193850357115565,
           "p": 0.3712312939226222,
           "r": 0.31313195314734876
       }
   }
    ```
2. 运行 predict.py 调用 greedy decode 或者 beam search 进行预测，beam search 使用的是最后一个ckpt，所以可能差在这

    ```json
    {
      "rouge-1": {
        "f": 0.2915635508381314,
        "p": 0.3804116780031509,
        "r": 0.2719187833527539
      },
      "rouge-2": {
        "f": 0.12879757845176937,
        "p": 0.16785280802905464,
        "r": 0.11995412911919483
      },
      "rouge-l": {
        "f": 0.2901830986437813,
        "p": 0.3689693518270286,
        "r": 0.26824690814678
      }
    }
    ```