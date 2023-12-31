python submitit_finetune.py \
    --job_dir /checkpoint/sshkhr/experiments/ViT-analysis/neurips_rebuttal/mae_ft_no_flip/ \
    --output_dir /checkpoint/sshkhr/experiments/ViT-analysis/neurips_rebuttal/mae_ft_no_flip/checkpoints/ \
    --log_dir /checkpoint/sshkhr/experiments/ViT-analysis/neurips_rebuttal/mae_ft_no_flip/logs/ \
    --partition devlab \
    --nodes 4 \
    --batch_size 32 \
    --nb_classes 1000 \
    --hflip 0.0 \
    --pretraining MAE \
    --model vit_base_patch16 --cls_token \
    --finetune /private/home/sshkhr/mae/checkpoints/pretrained/mae_pretrain_vit_base.pth \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path /datasets01/imagenet_full_size/061417/