#!/bin/bash
python train.py --name cityscapes_512 --dataroot ../TheCityscapesDataset --batchSize 1 --gpu_ids 0,1 --print_freq 500