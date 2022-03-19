"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import time
import data
import torch

from util import html
from tqdm import tqdm
from collections import OrderedDict
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM, FID
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel


def test(opt):
    # create metrics
    metric_fid = FID(device='cuda')
    metric_psnr = PSNR(data_range=2, device='cuda')
    metric_ssim = SSIM(data_range=2, device='cuda')
    
    dataloader = data.create_dataloader(opt)

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    time_total = 0
    # test
    iter_dataloader = iter(dataloader)
    for i in tqdm(range(len(dataloader))):
        data_i = next(iter_dataloader)
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        generated = model(data_i, mode='inference')
        torch.cuda.synchronize(device='cuda')
        end = time.time()
        f_time = end-start
        if i != 0:
            time_total += f_time
        
        metric_fid.update([generated, data_i['image'].cuda()])
        metric_psnr.update([generated, data_i['image'].cuda()])
        metric_ssim.update([generated, data_i['image'].cuda()])

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('gt', data_i['image'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])
            
    webpage.save()
    
    print(f'mean FID = {metric_fid.compute()}')
    print(f'mean PSNR = {metric_psnr.compute()}')
    print(f'mean SSIM = {metric_ssim.compute()}')
    
    print(f'average time per image = {time_total / i}')
    print(f'average image per second = {i / time_total}')
    
    result_file_path = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}', 'results.log')
    with open(result_file_path, 'w') as f:
        f.write(f'mean FID = {metric_fid.compute()}\n')
        f.write(f'mean PSNR = {metric_psnr.compute()}\n')
        f.write(f'mean SSIM = {metric_ssim.compute()}\n')
        
        f.write(f'average time per image = {time_total / i}\n')
        f.write(f'average image per second = {i / time_total}\n')

if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)
