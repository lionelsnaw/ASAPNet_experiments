#!/bin/bash
python train.py --name cityscapes_512 --dataroot ../TheCityscapesDataset --dataset_mode cityscapes --batchSize 4 --gpu_ids 0,1