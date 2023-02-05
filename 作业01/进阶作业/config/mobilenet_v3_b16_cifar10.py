_base_ = [
    '../_base_/models/mobilenet_v3_small_cifar.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

lr_config = dict(policy='step', step=[120, 170])

model = dict(
    head=dict(
        num_classes=10,
        topk=(1,)
    )
)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        data_prefix="data/CIFAR-10/cifar10/raw",
    ),
    val=dict(
        data_prefix='data/CIFAR-10/cifar10/raw',
    ),
    test=dict(
        data_prefix='data/CIFAR-10/cifar10/raw',
    ),
)

# 评估方法 top1
evaluation = dict(metric_options={'topk': (1,)})
# 优化器
# 学习率微调
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

runner = dict(type='EpochBasedRunner', max_epochs=200)

# 预训练模型
load_from = '/HOME/scz0ate/run/code/mmclassification/checkpoints/mobilenet_v3_small-8427ecf0.pth'
