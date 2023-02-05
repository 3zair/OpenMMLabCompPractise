#!/bin/bash
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
    configs/mobilenet_v3/mobilenet_v3_b16_cifar10.py \
    --work-dir work/mobilenet_v3_b16_cifar10 \
    --launcher pytorch
