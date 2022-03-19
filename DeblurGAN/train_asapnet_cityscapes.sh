#!/bin/bash
python train.py \
    --name cityscapes_512_asap --dataroot ../TheCityscapesDataset \
    --dataset_mode orig2blur --resize_or_crop scale_width \
    --learn_residual --fineSize 1024 --gan_type gan \
    --which_direction BtoA --display_id 0 \
    --save_latest_freq 1000 --save_epoch_freq 50 \
    --which_model_netG asapnet \
    --batchSize 1 --gpu_ids 0,1