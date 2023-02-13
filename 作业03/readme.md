# 作业三
作业说明：https://github.com/open-mmlab/OpenMMLabCamp/blob/main/AI%20%E5%AE%9E%E6%88%98%E8%90%A5%E5%9F%BA%E7%A1%80%E7%8F%AD/%E4%BD%9C%E4%B8%9A%E4%B8%89_mmsegmentation.md

1. 确定数据集

   组织病理切片小鼠肾小球：https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/Glomeruli-dataset.zip

2. 划分训练集、测试集

   使用脚本进行数据集划分：[split_data_set.py](./split_data_set.py)

3. 使用MMSegmentation训练语义分割模型

   在MMSegmentation中，指定预训练模型，配置config文件，修改类别数、学习率。

   训练脚本：[train](./train.py)

   设备环境：3090 * 2

   配置文件：[config.py](./result/pspnet_r50_test.py)

   模型文件：链接：https://pan.baidu.com/s/1lFGluH0CJT6Q2Juxe05eoA?pwd=gfg7  提取码：gfg7

4. 用训练得到的模型预测

   [jupyterbook-预测](./code/【E】用训练得到的模型预测.ipynb)

5. 获得测试集图片或新图片的语义分割预测结果，对结果进行可视化和后处理。

   [jupyterbook-预测](./code/【E】用训练得到的模型预测.ipynb)

6. 在测试集上评估算法的速度和精度性能

   [评估](./code/【F】测试集性能评估.ipynb)

### 评估结果

精度：

|            | IoU (%) | Acc(%) |
| ---------- | ------- | ------ |
| background | 99.72   | 99.89  |
| glomeruili | 82.22   | 88.55  |

混淆矩阵：

![image-20230213201115495](C:\Users\zhang\AppData\Roaming\Typora\typora-user-images\image-20230213201115495.png)