

# pretrain 
_config_pretrain = {
    'exp_name': "mlm_itm_cl_mgsc_mltc",
    'seed': 0,
    # 'datasets': ["coco", "vg", "sbu", "gcc"], 
    'datasets': ["coco", "vg"], 
    # 'datasets': ["webvid"],
    'loss_names': {
        "itm": 1,
        "mlm": 1,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mae": 0,
        "con": 1,
        "mgsc": 1, # global
        "mltc": 1, # local
        },
    'batch_size': 4096,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    'image_size': 288, # 224 for video
    'draw_false_image': 1,
    'image_only': False,
    'vit': 'mae_vit_base_patch16',
    'patch_size': 16,
    'num_frames': 4,
    'train_transform_keys': ["clip"],
    'val_transform_keys': ["clip"],
    'video_train_transform_keys': ["video_randaug"],
    'video_val_transform_keys': ["video_test"],

    # Text Setting
    'tokenizer': "roberta-base",
    'vocab_size': 50265,
    'max_text_len': 50,
    'vqav2_label_size': 3129,
    'mlm_prob': 0.15,
    'draw_false_text': 0,
    'whole_word_masking': True,

    # Transformer Setting 
    'num_layers': 12,
    'num_top_layer': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'hidden_size': 768,
    'num_heads': 12,

    # mae transformer settings
    'vit_path': "/apdcephfs/share_1367250/auroraji/pretrained_weight/clip-vit/ViT-B-16.pt",
    'mask_ratio': 0.8, # mgsc image mask ratio
    'mtm_ratio': 0.4, # text mask ratio
    'image_token_mask_ratio': 0.3, # mltc image mask ratio

    # Optimizer Setting
    'optim_type': "adamw",
    'weight_decay': 0.01,
    'decay_power': 1,
    'end_lr': 0,
    'learning_rate': 2e-5,
    'val_check_interval': 1.0,
    'lr_mult_head': 5,
    'lr_mult_cross_modal': 5,
    'max_epoch': 100,
    'max_steps': 15000,
    'warmup_steps': 0.1,

    # PL Trainer Setting
    'resume_from': None, # load interrupted ckpt
    'fast_dev_run': False, # for debug
    'test_only': False,

    # below params varies with the environment
    'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
    'log_dir': "result",
    'per_gpu_batchsize': 16,  # you should define this manually with per_gpu_batch_size=#
    'video_per_gpu_batchsize': 6,
    'num_gpus': 8,
    'num_nodes': 1,
    'load_path': "",
    'num_workers': 8,
    'precision': 16,
    'is_pretrain': True,

    # contrast
    'con_weight': 0.5,

    # for retrieval
    'get_recall_metric': False,
    'candidate_N': 128,

}

# VQA2.0
_config_vqa = {
    'exp_name': "finetune_vqa_randaug",
    'seed': 0,
    'datasets': ["vqa"], 
    'loss_names': {
        "itm": 0,
        "mlm": 0,
        "vqa": 1,
        "nlvr2": 0,
        "irtr": 0,
        "mae": 0,
        "con": 0,
        "scl": 0,
        },
    'batch_size': 512,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    'image_size': 384,
    'draw_false_image': 0,
    'image_only': False,
    'vit': 'mae_vit_base_patch16',
    'patch_size': 16,
    'train_transform_keys': ["clip_randaug"],
    'val_transform_keys': ["clip_test"],

    # Text Setting
    'tokenizer': "roberta-base",
    'vocab_size': 50265,
    'max_text_len': 50,
    'vqav2_label_size': 3129,
    'mlm_prob': 0.15,
    'draw_false_text': 0,
    'whole_word_masking': True,

    # Transformer Setting 
    'num_layers': 12,
    'num_top_layer': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'hidden_size': 768,
    'num_heads': 12,

    # mae transformer settings
    'vit_path': "/apdcephfs/share_1367250/auroraji/pretrained_weight/clip-vit/ViT-B-16.pt",
    'mask_ratio': 0.6,

    # Optimizer Setting
    'optim_type': "adamw",
    'weight_decay': 0.01,
    'decay_power': 1,
    'end_lr': 0,
    'learning_rate': 5e-6,
    'val_check_interval': 0.1,
    'lr_mult_head': 50,
    'lr_mult_cross_modal': 10, # 5
    'max_epoch': 10,
    'max_steps': None,
    'warmup_steps': 0.1,

    # PL Trainer Setting
    'resume_from': None, # load interrupted ckpt
    'fast_dev_run': False, # for debug
    'test_only': False,

    # below params varies with the environment
    'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
    'log_dir': "result",
    'per_gpu_batchsize': 16,  # you should define this manually with per_gpu_batch_size=#
    'num_gpus': 8,
    'num_nodes': 1,
    'load_path': "/apdcephfs_cq2/share_1367250/auroraji/VL-MAE4/result/mlm_itm_cl_msm_seed0_from_/version_56/checkpoints/epoch=41-step=99999.ckpt",
    'num_workers': 8,
    'precision': 16,
    'is_pretrain': False,

    # for retrieval
    'get_recall_metric': False,
    'candidate_N': 128,
    
    # contrast
    'negative_scale': 1/200,
    'shift': 4,

}

# NVLR2
_config_nlvr2 = {
    'exp_name': "finetune_nlvr2_randaug",
    'seed': 0,
    'datasets': ["nlvr2"], 
    'loss_names': {
        "itm": 0,
        "mlm": 0,
        "vqa": 0,
        "nlvr2": 1,
        "irtr": 0,
        "mae": 0,
        "con": 0,
        "scl": 0,
        },
    'batch_size': 256,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    'image_size': 288,
    'draw_false_image': 0,
    'image_only': False,
    'vit': 'mae_vit_base_patch16',
    'patch_size': 16,
    'train_transform_keys': ["clip_randaug"],
    'val_transform_keys': ["clip_test"],

    # Text Setting
    'tokenizer': "roberta-base",
    'vocab_size': 50265,
    'max_text_len': 50,
    'vqav2_label_size': 3129,
    'mlm_prob': 0.15,
    'draw_false_text': 0,
    'whole_word_masking': True,

    # Transformer Setting 
    'num_layers': 12,
    'num_top_layer': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'hidden_size': 768,
    'num_heads': 12,

    # mae transformer settings
    'vit_path': "/apdcephfs/share_1367250/auroraji/pretrained_weight/clip-vit/ViT-B-16.pt",
    'mask_ratio': 0.6,
    'mim_weight': 10,

    # Optimizer Setting
    'optim_type': "adamw",
    'weight_decay': 0.01,
    'decay_power': 1,
    'end_lr': 0,
    'learning_rate': 1e-5,
    'val_check_interval': 1.0,
    'lr_mult_head': 10,
    'lr_mult_cross_modal': 10, # 5
    'max_epoch': 10,
    'max_steps': None,
    'warmup_steps': 0.1,

    # PL Trainer Setting
    'resume_from': None, # load interrupted ckpt
    'fast_dev_run': False, # for debug
    'test_only': False,

    # below params varies with the environment
    'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
    'log_dir': "result",
    'per_gpu_batchsize': 16,  # you should define this manually with per_gpu_batch_size=#
    'num_gpus': 8,
    'num_nodes': 1,
    'load_path': "/apdcephfs_cq2/share_1367250/auroraji/VL-MAE4/result/mlm_itm_cl_msm_seed0_from_/version_56/checkpoints/epoch=41-step=99999.ckpt",
    'num_workers': 8,
    'precision': 16,
    'is_pretrain': False,

    # for retrieval
    'get_recall_metric': False,
    'candidate_N': 128,
    
    # contrast
    'negative_scale': 1/200,
    'shift': 4,

}

# F30k
_config_f30k = {
    'exp_name': "finetune_irtr_f30k",
    'seed': 1919, # ?
    'datasets': ["f30k"], 
    'loss_names': {
        "itm": 0.5,
        "mlm": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 1,
        "mae": 0,
        "con": 0.5,
        "scl": 0,
        },
    'batch_size': 512,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    'image_size': 384,
    'draw_false_image': 0,
    'image_only': False,
    'vit': 'mae_vit_base_patch16',
    'patch_size': 16,
    'train_transform_keys': ["clip_randaug"],
    'val_transform_keys': ["clip_test"],

    # Text Setting
    'tokenizer': "roberta-base",
    'vocab_size': 50265,
    'max_text_len': 50,
    'vqav2_label_size': 3129,
    'mlm_prob': 0.15,
    'draw_false_text': 15,
    'whole_word_masking': True,

    # Transformer Setting 
    'num_layers': 12,
    'num_top_layer': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'hidden_size': 768,
    'num_heads': 12,

    # mae transformer settings
    'vit_path': "/apdcephfs/share_1367250/auroraji/pretrained_weight/clip-vit/ViT-B-16.pt",
    'mask_ratio': 0.6,

    # Optimizer Setting
    'optim_type': "adamw",
    'weight_decay': 0.01,
    'decay_power': 1,
    'end_lr': 0,
    'learning_rate': 5e-6,
    'val_check_interval': 1.0,
    'lr_mult_head': 10, # 5
    'lr_mult_cross_modal': 10, # 5
    'max_epoch': 10,
    'max_steps': None,
    'warmup_steps': 0.1,

    # PL Trainer Setting
    'resume_from': None, # load interrupted ckpt
    'fast_dev_run': False, # for debug
    'test_only': True,

    # below params varies with the environment
    'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
    'log_dir': "result",
    'per_gpu_batchsize': 16,  # you should define this manually with per_gpu_batch_size=#
    'num_gpus': 8,
    'num_nodes': 1,
    'load_path': "/apdcephfs_cq2/share_1367250/auroraji/SCL_deepspeed/result/mlm_itm_cl_scl_seed0_from_/version_2/checkpoints/epoch=11-step=14999.ckpt",
    'num_workers': 8,
    'precision': 16,
    'is_pretrain': False,

    # for retrieval
    'get_recall_metric': True,
    'candidate_N': 128,
    
    # contrast
    'con_weight': 0.5,

}

# COCO
_config_coco = {
    'exp_name': "finetune_irtr_coco",
    'seed': 0,
    'datasets': ["coco"],
    'loss_names': {
        "itm": 0.5,
        "mlm": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 1,
        "mae": 0,
        "con": 0.5,
        "scl": 0,
        },
    'batch_size': 512,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    'image_size': 384,
    'draw_false_image': 0,
    'image_only': False,
    'vit': 'mae_vit_base_patch16',
    'patch_size': 16,
    'train_transform_keys': ["clip_randaug"],
    'val_transform_keys': ["clip_test"],

    # Text Setting
    'tokenizer': "roberta-base",
    'vocab_size': 50265,
    'max_text_len': 50,
    'vqav2_label_size': 3129,
    'mlm_prob': 0.15,
    'draw_false_text': 15,
    'whole_word_masking': True,

    # Transformer Setting 
    'num_layers': 12,
    'num_top_layer': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'hidden_size': 768,
    'num_heads': 12,

    # mae transformer settings
    'vit_path': "/apdcephfs/share_1367250/auroraji/pretrained_weight/clip-vit/ViT-B-16.pt",
    'mask_ratio': 0.6,

    # Optimizer Setting
    'optim_type': "adamw",
    'weight_decay': 0.01,
    'decay_power': 1,
    'end_lr': 0,
    'learning_rate': 5e-6,
    'val_check_interval': 1.0,
    'lr_mult_head': 5,
    'lr_mult_cross_modal': 5,
    'max_epoch': 10,
    'max_steps': None,
    'warmup_steps': 0.1,

    # PL Trainer Setting
    'resume_from': None, # load interrupted ckpt
    'fast_dev_run': False, # for debug
    'test_only': True,

    # below params varies with the environment
    'data_root': '/apdcephfs/share_1367250/auroraji/data/arrow/ft_local',
    'log_dir': "result",
    'per_gpu_batchsize': 16,  # you should define this manually with per_gpu_batch_size=#
    'num_gpus': 8,
    'num_nodes': 1,
    'load_path': "/apdcephfs_cq2/share_1367250/auroraji/SCL_deepspeed/result/finetune_irtr_coco_seed0_from_epoch=41-step=99999/version_15/checkpoints/epoch=9-step=11069.ckpt",
    'num_workers': 8,
    'precision': 16,
    'is_pretrain': False,

    # for retrieval
    'get_recall_metric': True,
    'candidate_N': 256,

    # contrast
    'con_weight': 0.5,
}

# lsmdc/msrvtt
_config_lsmdc = {
    'exp_name': "finetune_lsmdc",
    'seed': 0,
    'datasets': ["lsmdc"],
    'loss_names': {
        "itm": 0.5,
        "mlm": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 1,
        "mae": 0,
        "con": 0.5,
        "scl": 0,
        },
    'batch_size': 256,  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    'image_size': 224,
    'draw_false_image': 1,
    'image_only': False,
    'vit': 'mae_vit_base_patch16',
    'patch_size': 16,
    'num_frames': 4,
    'train_transform_keys': ["clip"],
    'val_transform_keys': ["clip"],
    'video_train_transform_keys': ["video_randaug"],
    'video_val_transform_keys': ["video_test"],

    # Text Setting
    'tokenizer': "roberta-base",
    'vocab_size': 50265,
    'max_text_len': 50,
    'vqav2_label_size': 3129,
    'mlm_prob': 0.15,
    'draw_false_text': 15,
    'whole_word_masking': True,

    # Transformer Setting
    'num_layers': 12,
    'num_top_layer': 6,
    'mlp_ratio': 4,
    'drop_rate': 0.1,
    'hidden_size': 768,
    'num_heads': 12,

    # mae transformer settings
    'vit_path': "./pretrained_weight/clip-vit/ViT-B-16.pt",
    'mask_ratio': 0.8,
    'mtm_ratio': 0.4,

    # Optimizer Setting
    'optim_type': "adamw",
    'weight_decay': 0.01,
    'decay_power': 1,
    'end_lr': 0,
    'learning_rate': 5e-6,
    'val_check_interval': 1.0,
    'lr_mult_head': 5,
    'lr_mult_cross_modal': 5,
    'max_epoch': 10,
    'max_steps': None,
    'warmup_steps': 0.05,

    # PL Trainer Setting
    'resume_from': None, # load interrupted ckpt
    'fast_dev_run': False, # for debug
    'test_only': False,

    # below params varies with the environment
    'data_root': './data/arrow/ft_local',
    'log_dir': "result",
    'per_gpu_batchsize': 1,  # you should define this manually with per_gpu_batch_size=#
    'video_per_gpu_batchsize': 1, # image and video: 8 / 4, image: 16, video: 6
    'num_nodes': 1,
    'num_gpus': 8,
    'load_path': "",
    'num_workers': 8,
    'precision': 16,
    'is_pretrain': False,

    # contrast
    'con_weight': 1.,

    # for retrieval
    'get_recall_metric': False,
    'candidate_N': 128,
}


config_dict = {
    'pretrain': _config_pretrain,
    'vqa': _config_vqa,
    'nlvr2': _config_nlvr2,
    'f30k': _config_f30k,
    'coco': _config_coco,
    'lsmdc': _config_lsmdc,
}