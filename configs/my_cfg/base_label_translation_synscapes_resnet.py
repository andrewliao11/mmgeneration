import os.path as osp

_base_ = [
    '../_base_/models/label_translation/label_translation_bbox_detr.py',
    #'../_base_/datasets/unpaired_synscapes_bboxes.py',
    '../_base_/default_runtime.py'
]

domain_a = None
domain_b = None
model = dict(
    default_domain=domain_b,
    reachable_domains=[domain_a, domain_b],
    related_domains=[domain_a, domain_b],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=10.0,
            loss_name='cycle_loss',
            data_info=dict(
                pred=f'cycle_{domain_b}',
                target=f'real_{domain_b}',
            ),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_a}', target=f'real_{domain_a}'),
            reduction='mean'),
        dict(
            type='L1Loss',
            loss_weight=0.5,
            loss_name='id_loss',
            data_info=dict(
                pred=f'identity_{domain_b}', target=f'real_{domain_b}'),
            reduction='mean')
    ])

dataroot = None


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    

#NOTE: Detection pipelines uses the pipelines in mmdet
detection_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

detection_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


domain_a_datasets = dict(
    train=dict(
        type='DetectionCityscapesDataset', 
        ann_file=osp.join(dataroot, domain_a, 'train/annotations/bbox.json'), 
        img_prefix=osp.join(dataroot, domain_a, 'train/rgb'), 
        pipeline=detection_train_pipeline
    ), 
    val=dict(
        type='DetectionCityscapesDataset', 
        ann_file=osp.join(dataroot, domain_a, 'val/annotations/bbox.json'), 
        img_prefix=osp.join(dataroot, domain_a, 'val/rgb'), 
        pipeline=detection_test_pipeline
    ), 
    test=dict(
        type='DetectionCityscapesDataset', 
        ann_file=osp.join(dataroot, domain_a, 'test/annotations/bbox.json'), 
        img_prefix=osp.join(dataroot, domain_a, 'test/rgb'), 
        pipeline=detection_test_pipeline
    )
)


domain_b_datasets = dict(
    train=dict(
        type='DetectionCityscapesDataset', 
        ann_file=osp.join(dataroot, domain_b, 'train/annotations/bbox.json'), 
        img_prefix=osp.join(dataroot, domain_b, 'train/rgb'), 
        pipeline=detection_train_pipeline
    ), 
    val=dict(
        type='DetectionCityscapesDataset', 
        ann_file=osp.join(dataroot, domain_b, 'val/annotations/bbox.json'), 
        img_prefix=osp.join(dataroot, domain_b, 'val/rgb'), 
        pipeline=detection_test_pipeline
    ), 
    test=dict(
        type='DetectionCityscapesDataset', 
        ann_file=osp.join(dataroot, domain_b, 'test/annotations/bbox.json'), 
        img_prefix=osp.join(dataroot, domain_b, 'test/rgb'), 
        pipeline=detection_test_pipeline
    )
)




data = dict(
    samples_per_gpu=1, 
    workers_per_gpu=2, 
    train=dict(
        type='DetectionUnpairedDataset', 
        domain_a_dataset=domain_a_datasets['train'], 
        domain_b_dataset=domain_b_datasets['train']
    ), 
    val=dict(
        type='DetectionUnpairedDataset', 
        domain_a_dataset=domain_a_datasets['val'], 
        domain_b_dataset=domain_b_datasets['val']
    ),
    test=dict(
        type='DetectionUnpairedDataset', 
        domain_a_dataset=domain_a_datasets['test'], 
        domain_b_dataset=domain_b_datasets['test']
    )
)



optimizer = dict(
    generators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=0.0002, betas=(0.5, 0.999)))

# learning policy
lr_config = dict(
    policy='Linear', by_epoch=False, target_lr=0, start=135000, interval=1350)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)


runner = None
use_ddp_wrapper = True
total_iters = 270000
workflow = [('train', 1)]
exp_name = 'label_translation_unpaired_synscapes'
work_dir = f'./work_dirs/experiments/{exp_name}'
# testA 120, testB 140
num_images = 140
metrics = dict(
    FID=dict(type='FID', num_images=num_images, image_shape=(3, 256, 256)),
    IS=dict(
        type='IS',
        num_images=num_images,
        image_shape=(3, 256, 256),
        inception_args=dict(type='pytorch')))

evaluation = dict(
    type='TranslationEvalHook',
    target_domain=domain_b,
    interval=10000,
    metrics=[
        dict(type='FID', num_images=num_images, bgr2rgb=True),
        dict(
            type='IS',
            num_images=num_images,
            inception_args=dict(type='pytorch'))
    ],
    best_metric=['fid', 'is'])

