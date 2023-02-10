# 作业二

作业要求：https://github.com/open-mmlab/OpenMMLabCamp/blob/main/AI%20%E5%AE%9E%E6%88%98%E8%90%A5%E5%9F%BA%E7%A1%80%E7%8F%AD/%E4%BD%9C%E4%B8%9A%E4%BA%8C_mmdetection.md

## 数据集

balloon是带有mask的气球数据集，其中训练集包含61张图片，验证集包含13张图片。

下载链接：https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip

数据集转coco脚本：[balloon2coco.py](balloon2coco.py)

## 模型微调

### 配置文件

[mask_rcnn_r50_coco.py](./mask_rcnn_r50_coco.py)

### 结果

**模型文件下载地址：**

链接：https://pan.baidu.com/s/1alwgwdeGymeC4A2lrzOPIQ?pwd=02bm 
提取码：02bm

| 模型      | bbox_mAP (%) |
| --------- | ------------ |
| mask_rcnn | 79.7         |

## 测试视频转换

转换后的视频：[视频](./color_splash.mp4)

压缩后的GIF：

![](./color_splash.gif)
