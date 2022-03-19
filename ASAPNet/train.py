"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import sys
import data
import copy
import torch

from tqdm import tqdm
from util.visualizer import Visualizer
from util.iter_counter import IterationCounter
from collections import OrderedDict
from util.metrics import FID, PSNR, SSIM
from options.train_options import TrainOptions
from trainers.pix2pix_trainer import Pix2PixTrainer
from torch.utils.tensorboard import SummaryWriter


# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# create writer
writer = SummaryWriter(os.path.join('checkpoints', opt.name, 'runs'))

# load the dataset
dataloader = data.create_dataloader(opt)
opt_test = copy.deepcopy(opt)
opt_test.phase = 'test'
dataloader_test = data.create_dataloader(opt_test)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

# create metrics
metric_fid = FID(device='cuda')
metric_psnr = PSNR(data_range=2, device='cuda')
metric_ssim = SSIM(data_range=2, device='cuda')

# create loss couter
total_loss = {'GAN': torch.tensor(.0), 'GAN_Feat': torch.tensor(.0), 'VGG': torch.tensor(.0),
              'D_Fake': torch.tensor(.0), 'D_real': torch.tensor(.0), 'count': 0}

for epoch in tqdm(iter_counter.training_epochs()):
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if (opt.D_steps_per_G == 0):
            trainer.run_generator_one_step(data_i)
        elif (i % opt.D_steps_per_G == 0):
            trainer.run_generator_one_step(data_i)

        # train discriminator
        if (opt.D_steps_per_G != 0):
            trainer.run_discriminator_one_step(data_i)

        losses = trainer.get_latest_losses(opt.D_steps_per_G)
        for key in losses.keys():
            total_loss[key] += losses[key].detach().mean().cpu()
        total_loss['count'] += 1

        # Visualizations
        if iter_counter.needs_printing():
            for key in losses.keys():
                writer.add_scalar(f'Loss/{key}', total_loss[key] / total_loss['count'], iter_counter.total_steps_so_far)
                total_loss[key] = 0
            total_loss['count'] = 0

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image',
                                    trainer.get_latest_generated()),
                                   ('real_image', data_i['image'])])
            visuals = visualizer.convert_visuals_to_numpy(visuals)
            
            writer.add_images('synthesized_image', visuals['synthesized_image'], iter_counter.total_steps_so_far, dataformats='NHWC')
            writer.add_images('real_image', visuals['real_image'], iter_counter.total_steps_so_far, dataformats='NHWC')
            writer.add_images('input_label', visuals['input_label'], iter_counter.total_steps_so_far, dataformats='HWC')

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()
    
    model = trainer.pix2pix_model
    model.eval()
    for i, data_i in enumerate(dataloader_test):
        with torch.no_grad():
            im = data_i['image'].cuda()
            _, generated = model(data_i, mode='generator')
            metric_fid.update([generated, im])
            metric_psnr.update([generated, im])
            metric_ssim.update([generated, im])
    model.train()
    
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

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
