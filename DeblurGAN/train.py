import os
import copy
import time
import torch
import numpy as np
import util.util as util

from tqdm import tqdm
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from multiprocessing import freeze_support
from util.metrics import FID, PSNR, SSIM
from torch.utils.tensorboard import SummaryWriter


freeze_support()

opt = TrainOptions().parse()

if opt.which_model_netG == 'asapnet':
    opt.norm_G='instanceaffine'
    opt.lr_instance=True
    opt.no_instance_dist=True
    opt.hr_coor="cosine"
    opt.netD_subarch = 'n_layer'
    opt.num_D = 1
    opt.n_layers_D = 4
    opt.ndf_max = 512
    opt.learned_ds_factor = 16
    opt.crop_size = 1024
    opt.aspect_ratio = 2
    opt.hr_width = 64
    opt.no_one_hot = False
    opt.hr_depth = 5
    opt.lr_width = 64
    opt.lr_max_width = 1024
    opt.lr_depth = 7
    opt.reflection_pad = False
    opt.contain_dontcare_label = False
    opt.no_instance_edge = False
    opt.no_instance = opt.no_instance_edge
    opt.no_ganFeat_loss = False
    opt.norm_D = 'spectralinstance'
    opt.norm_G = 'spectralinstance'

writer = SummaryWriter(os.path.join('checkpoints', opt.name, 'runs'))

model = create_model(opt)
visualizer = Visualizer(opt)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

opt_test = copy.deepcopy(opt)
opt_test.isTrain = False
data_loader_test = CreateDataLoader(opt_test)
dataset_test = data_loader_test.load_data()
dataset_test_size = len(data_loader_test)
print('#test images = %d' % dataset_test_size)

# create metrics
metric_fid = FID(device='cuda')
metric_psnr = PSNR(data_range=2, device='cuda')
metric_ssim = SSIM(data_range=2, device='cuda')

# create loss couter
total_loss = {'GAN': torch.tensor(.0), 'GAN_Feat': torch.tensor(.0), 'VGG': torch.tensor(.0),
            'D_Fake': torch.tensor(.0), 'D_real': torch.tensor(.0), 'count': 0}

total_steps = 0
for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
    epoch_iter = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()
        
        if total_steps % opt.display_freq == 0:
            real_A = []
            fake_B = []
            real_B = []
            for i in range(model.real_A.data.shape[0]):
                real_A.append(util.tensor2im(model.real_A.data[i:i+1]))
                fake_B.append(util.tensor2im(model.fake_B.data[i:i+1]))
                real_B.append(util.tensor2im(model.real_B.data[i:i+1]))
            writer.add_images('real_A', np.asarray(real_A), total_steps, dataformats='NHWC')
            writer.add_images('fake_B', np.asarray(fake_B), total_steps, dataformats='NHWC')
            writer.add_images('real_B', np.asarray(real_B), total_steps, dataformats='NHWC')

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            for err_key in errors:
                writer.add_scalar(f'Errors/{err_key}', errors[err_key], total_steps)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    for i, data_i in enumerate(dataset_test):
        with torch.no_grad():
            model.set_input(data_i)
            model.test()
            generated = model.fake_B
            im = model.real_B
            metric_fid.update([generated, im])
            metric_psnr.update([generated, im])
            metric_ssim.update([generated, im])
    
    fid = metric_fid.compute()
    psnr = metric_psnr.compute()
    ssim = metric_ssim.compute()
    
    metric_fid.reset()
    metric_psnr.reset()
    metric_ssim.reset()
    
    writer.add_scalar('Metrics/FID', fid, epoch)
    writer.add_scalar('Metrics/PSNR', psnr, epoch)
    writer.add_scalar('Metrics/SSIM', ssim, epoch)
    writer.flush()

    if epoch > opt.niter:
        model.update_learning_rate()
