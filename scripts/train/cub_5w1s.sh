python -B train.py -batch 64 -dataset cub -gpu 0 -extra_dir your_run -matcher_prob 1 -temperature_attn 2.0 -lamb 1.5 -lr 0.01 -max_epoch 30 -milestones 20 24 26 28

nohup python train.py -batch 64 -dataset cub -gpu 1 -extra_dir your_run -temperature_attn 2.0 -lamb 1.5 -no_wandb -lr 0.01 -max_epoch 30 -milestones 20 24 26 28 &