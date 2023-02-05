#!/bin/bash
# 超算平台提交任务脚本
# 加载模块
module load anaconda/2021.05
module load cuda/11.1
module load gcc/7.3

# 激活环境
source activate openmmlab_mmclassification

# 刷新⽇志缓存
export PYTHONUNBUFFERED=1

N_GPUS=2

# 训练模型
python -m torch.distributed.launch \
    --nproc_per_node=${N_GPUS} \
    tools/train.py \
    configs/resnet34/resnet34_b32_flower.py \
    --work-dir work/resnet34_b32_flower \
    --launcher pytorch
