# OpenMMLab训练营 作业01

作业说明：[作业01](https://github.com/open-mmlab/OpenMMLabCamp/issues/10)

## 基础作业

使用mmclassification对5种花进行分类

### 实验设备

NVIDA GeForce 3090 *2 

#### 数据集介绍

flower 数据集包含 5 种类别的花卉图像：雏菊 daisy 588张，蒲公英 dandelion 556张，玫瑰 rose 583张，向日葵 sunflower 536张，郁金香 tulip 585张。

数据集下载链接：

- 国际网：https://www.dropbox.com/s/snom6v4zfky0flx/flower_dataset.zip?dl=0
- 国内网：https://pan.baidu.com/s/1RJmAoxCD_aNPyTRX6w97xQ 提取码: 9x5u

##### 对数据集进行划分

将数据集按照 8:2 的比例划分成训练和验证子数据集，并将数据集整理成 ImageNet的格式。可通过split_dataset_imageNet.py 脚本实现

### 结果

| Model    | Top-1 (%) |
| -------- | --------- |
| ResNet34 | 97.71127  |
 | ResNet18| 96.12676|

训练保存的模型：https://pan.baidu.com/s/1iKsjsn9B_qCigUadtlkR6A?pwd=u8jh  提取码：u8jh

## 进阶作业

### CIFAR-10数据集

🚥图像分类数据集：CIFAR-10：https://opendatalab.com/CIFAR-10 562M

### 结果

| Model       | Top-1 (%) |
| ----------- | --------- |
| MobileNetV3 | 76.62   |

