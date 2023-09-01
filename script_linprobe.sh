# /checkpoint/sshkhr/experiments/ViT-analysis/checkpoints/VICReg/vit_base_patch16/checkpoint-660.pth
# /private/home/sshkhr/mae/checkpoints/pretrained/msn_vitb16_600ep.pth.tar
# /checkpoint/sshkhr/experiments/ViT-analysis/checkpoints/MAE/vit_base_patch16/checkpoint-799.pth
# /checkpoint/sshkhr/experiments/ViT-analysis/checkpoints/DINO/vit_base_patch16/checkpoint-300.pth
python submitit_finetune.py \
    --job_dir /checkpoint/sshkhr/experiments/ViT/MSN/finetuned/ \
    --nodes 4 \
    --partition devlab \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune /private/home/sshkhr/mae/checkpoints/pretrained/msn_vitb16_600ep.pth.tar \
    --epochs 100 \
    --warmup_epochs 10 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path /datasets01/imagenet_full_size/061417/