_base_ = ['../_base_/models/resnet34.py', '../_base_/datasets/imagenet_bs32.py', '../_base_/default_runtime.py']

model = dict(
    head=dict(
        # 五类花卉
        num_classes=5,
        topk=(1,)
    )
)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        data_prefix="data/flower_dataset/train",
        ann_file="data/flower_dataset/train.txt",
        classes='data/flower_dataset/classes.txt'
    ),
    val=dict(
        data_prefix='data/flower_dataset/val',
        ann_file='data/flower_dataset/val.txt',
        classes='data/flower_dataset/classes.txt'
    ),
)

# 评估方法 top1
evaluation = dict(metric_options={'topk': (1,)})
# 优化器
# 学习率微调
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    step=[1]
)
runner = dict(type='EpochBasedRunner', max_epochs=100)

# 预训练模型
load_from = '/HOME/scz0ate/run/code/mmclassification/checkpoints/resnet34_batch256_imagenet_20200708-32ffb4f7.pth'
