#!/bin/bash

cd ASAPNet
echo "Start train ASAPNet"
bash ./train_cityscapes.sh
echo "Done"
echo "Start test ASAPNet"
bash ./test_cityscapes.sh
echo "Done"

cd ../Pix2pixHD
echo "Start train Pix2pixHD"
bash ./train_cityscapes.sh
echo "Done"
echo "Start test Pix2pixHD"
bash ./test_cityscapes.sh
echo "Done"

cd ../DeblurGAN
echo "Start train DeblurGAN_ASAPNet"
bash ./train_asapnet_cityscapes.sh
echo "Done"
echo "Start test DeblurGAN_ASAPNet"
bash ./test_asapnet_cityscapes.sh
echo "Done"

echo "Start train DeblurGAN"
bash ./train_cityscapes.sh
echo "Done"
echo "Start test DeblurGAN"
bash ./test_cityscapes.sh
echo "Done"