python submitit_pretrain.py \
    --output_dir /checkpoint/sshkhr/experiments/ViT-analysis/freeze_encoder/random_init/checkpoints/ \
    --log_dir /checkpoint/sshkhr/experiments/ViT-analysis/freeze_encoder/random_init/logs/ \
    --job_dir /checkpoint/sshkhr/experiments/ViT-analysis/neurips_rebuttal/mae_no_flip/ \
    --partition learnlab \
    ---nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_base_patch16 --cls_token \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --dist_eval --data_path /datasets01/imagenet_full_size/061417/