程晔安 自72 2017011627

## 使用说明 ##
1. 首先安装AdaBound 再将train.py中的import adabound解除注释就可以正常使用
2. 运行./data/data_process.py获得增强后的数据
3. 开始训练 python train.py
4. 进行模型融合 python ensemble.py
5. 结果在./submission中

实现详情见报告

## 文件结构 ##
- ensemble.py 模型融合文件
- train.py 训练文件
- report.pdf 报告文件
- ./data/: 数据文件夹,
-- /original_data为题目给出的原数据,
-- /aug_data为增强后的数据和切分成5折的数据
-- /data_process.py增强数据处理过程
- ./datasets 自己实现的数据集读入程序
- ./experiments 训练原模型保存处
- ./log 训练log保存处
- ./mymodels
-- myDenseNet.py 自己实现的DenseNet
-- myResNet.py 自己实现的ResNet
-- utils.py 配合辅助文件
- ./pictures报告中的ROC图和loss图保存处
- ./results 用于之后模型融合的模型在测试集上的输出的文件
- ./submission 最后的上传kaggle文件保存处
- ./utils 其他工具程序
-- plot.py 画loss曲线程序
-- roc.py 画roc曲线程序