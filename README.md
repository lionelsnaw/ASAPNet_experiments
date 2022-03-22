## ASAPNet and Pix2Pix Experiments

Pixel-wise networks can be an alternative to the standard approach for image-to-image task, which is based on generative adversarial networks (GAN). 

The first idea is to use pixel-wise MLPs during processing high-resolution image instead of convolution neural network, so each pixel is processed independently of others. Secondly, these parameters of MLPs are predicted by a fast convolutional network that processes a low-resolution representation of the input. According to authors, such [model](https://github.com/tamarott/ASAPNet) (ASAPNet) is up to 18x faster than state-of-the-art baselines and gives comparable results. In this project, we replicated results from the paper for segmentation task and adapted this architecture for denoising task. Our team reproduced paper results for segmentation problem, compared them with pix2pix baseline, adapted generator network architecture for dublurring problem and compared with deblurring baseline.

## Related Works.

## Installation.

The implementation is intended to be trained and inferenced on GPU. By default, installation and other commands below can be run in Google Colab with active GPU-runtime. (Except for training that will take more than 3 days.)

```
$ sudo apt update
$ git clone https://github.com/CaBuHoB/ASAPNet_experiments
$ cd ASAPNet_experiments/
$ sudo pip3 install -r requirements.txt
```

## Repository Structure

Repository contains several directories with corresponding models and with their instructions. We forked original works.
```bash

ASAPNet_experiments
├── ASAPNet                        # contains ASAPNet architecture
│      ├── ...
│      ├── train_cityscapes.sh     # bash file to train model architecture
│      └── test_cityscapes.sh
├── DeblurGAN                      # contains DebluGAn architecture
│      ├── ...
│      ├── train_cityscapes.sh     # bash file to train model architecture
│      └── test_cityscapes.sh
├── Pix2PixHD                      # contrains Pix2Pix architectures
│      ├── ...
│      ├── train_cityscapes.sh     # bash file to train model architecture
│      └── test_cityscapes.sh
├── TheCityScapesDataset
│      └── download.sh
├── ...
└── README.md
```


## Data Download

In order to train and test, we used the data obtained from CityScapes. This can be done easily, you just need to register account and use its credentials in the following command line in place of `YourUsername` and `YourPassword`

```
$ cd TheCityscapesDataset/
$ sudo apt install wget unzip 
$ !USERNAME=YourUsername PASSWORD=YourPassword bash ./download.sh
```

## Train and Test 

All commands to train are written in bash files according to model architecture. Moreover, there are two ways to train and test. The first is training and testing sperately for each model as follows below:

Note: All models in this case will be trained by the default options unless you would like to change/add in corresponding bash file. For quick train/test we recommend to use the following convention in bash files:

`train_cityscapes.sh`:

```
python train.py --name cityscapes_512 --dataroot ../TheCityscapesDataset --dataset_mode cityscapes --batchSize 1 --gpu_ids 0 --print_freq 500 --save_latest_freq 1
```
and  `test_cityscapes.sh`:
```
!python test.py --name cityscapes_512 --dataroot ../TheCityscapesDataset --dataset_mode cityscapes --batchSize 1 --gpu_ids 0 --phase test
```
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

## Results

The summary of perforamnce and comparison of models.

|         | ASAPNet | Pix2PixHD |
|---------|---------|-----------|
| FID     | 225.77  | 217.24    |
| PSNR    | 15.46   | 16.42     |
| SSIM    | 0.45    | 0.52      |

Visual comparison:

![results](https://raw.githubusercontent.com/lionelsnaw/ASAPNet_experiments/main/results.png)

|         | DeblurGAN | ASAPNet Deblur |
|---------|-----------|----------------|
| FID     | 8.6       | 66.21          |
| PSNR    | 34.5      | 21.65          |
| SSIM    | 0.93      | 0.70           |

Visual comparison:

![results2](https://raw.githubusercontent.com/lionelsnaw/ASAPNet_experiments/main/results2.jpeg)


## Credits
