_base_ = ['./mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        dataset=dict(
            ann_file='dataset/balloon/train/coco_train_ann.json',
            img_prefix='dataset/balloon/train',
            classes=("balloon",)
        )
    ),
    val=dict(
        ann_file='dataset/balloon/val/coco_val_ann.json',
        img_prefix='dataset/balloon/val',
        classes=("balloon",)

    ),
    test=dict(
        ann_file='dataset/balloon/val/coco_val_ann.json',
        img_prefix='dataset/balloon/val',
        classes=("balloon",)
    ),
)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='resnet50-0676ba61.pth')
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1,
        ),
        mask_head=dict(
            num_classes=1,
        )
    )
)
# epoch
runner = dict(type='EpochBasedRunner', max_epochs=2)
# learning rate
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# logging
log_config = dict(interval=24, hooks=[dict(type='TextLoggerHook')])

load_from = "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
