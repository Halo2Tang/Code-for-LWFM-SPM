python -B train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir your_run -matcher_prob 0.25 -temperature_attn 5.0 -lamb 0.25 -lr 0.01 -max_epoch 10 -milestones 4 6 8

python -B train.py -batch 128 -dataset fc100 -gpu 0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -lr 0.01 -max_epoch 10 -milestones 4 6 8

nohup python train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -no_wandb -lr 0.01 -max_epoch 10 -milestones 4 6 8 &

# for plot

nohup python train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -no_wandb -lr 0.1 -max_epoch 80 -milestones 40 60 &

nohup python train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -no_wandb -lr 0.01 -max_epoch 80 -milestones 4 6 8 &
