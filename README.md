## ASAPNet and Pix2Pix experiments

Pixel-wise networks can be an alternative to the standard approach for image-to-image task, which is based on generative adversarial networks (GAN). 

The first idea is to use pixel-wise MLPs during processing high-resolution image instead of convolution neural network, so each pixel is processed independently of others. Secondly, these parameters of MLPs are predicted by a fast convolutional network that processes a low-resolution representation of the input. According to authors, such [model](https://github.com/tamarott/ASAPNet) (ASAPNet) is up to 18x faster than state-of-the-art baselines and gives comparable results. In this project, we replicated results from the paper for segmentation task and adapted this architecture for denoising task. Our team reproduced paper results for segmentation problem, compared them with pix2pix baseline, adapted generator network architecture for dublurring problem and compared with deblurring baseline.

## Installation.

```
$ sudo apt update
$ git clone https://github.com/CaBuHoB/ASAPNet_experiments
$ cd ASAPNet_experiments/
$ sudo pip3 install -r requirements.txt
```

## Data Download

In oreder to train and test, we used the data obtained from CityScapes. This can be done easily, you just need to register account and use its credentials in the following command line in place of `YourUsername` and `YourPassword`

```
$ cd TheCityscapesDataset/
$ sudo apt install wget unzip 
$ !USERNAME=YourUsername PASSWORD=YourPassword bash ./download.sh
```

## Train and Test 

All commands to train are written in bash files according to model architecture. Moreover, there are two ways to train and test. The first is training and testing sperately for each model as follows below:
### ASAPNet

```
$ cd .. 
$ cd ASAPNet/
$ bash train_cityscapes.sh
$ bash test_cityscapes.sh
```

### Pix2Pix

```
$ cd .. 
$ cd Pix2PixHD/
$ bash train_cityscapes.sh
$ bash test_cityscapes.sh
```

### DeblurGANN

```
$ cd ..
$ cd DeblurGAN/
$ bash train_cityscapes.sh
$ bash test_cityscapes.sh
```

The second way is do it all in one command, i.e. train and test all architectures:

```
$ cd ASAPNet_experiments/
$ bash train_test_all_models.sh
```
