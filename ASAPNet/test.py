"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import time
import torch

from util import html
from collections import OrderedDict
from util.visualizer import Visualizer
from ignite.metrics import FID, PSNR, SSIM
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel


def test(opt):
    # FIXME
    # opt.batchSize = 1
    # opt.no_instance_edge = True
    # opt.no_instance_dist = True
    # opt.gpu_ids = [0]
    # opt.output_nc = 3
    # opt.label_nc = 3
    # opt.no_one_hot = True
    # opt.name = 'textest_cityscapes'
    # # opt.name = 'deblur_cityscapes'
    # opt.semantic_nc = opt.label_nc
    # opt.no_instance = opt.no_instance_edge
    # # opt.dataset_mode = 'deblur'
    # opt.phase = 'test'
    
    # create metrics
    metric_fid = FID(device='cuda')
    metric_psnr = PSNR(data_range=2, device='cuda')
    metric_ssim = SSIM(data_range=2, device='cuda')

    dataloader = data.create_dataloader(opt)
    
    print(len(dataloader))

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))
    time_total = 0
    # test
    for i, data in enumerate(dataloader):
        torch.cuda.reset_max_memory_allocated()
        start = time.time()
        generated = model(data, mode='inference')
        torch.cuda.synchronize(device='cuda')
        end = time.time()
        f_time = end-start
        if i != 0:
            time_total += f_time
        print("time_%d:%f" % (i, f_time))
        
        metric_fid.update([generated, data['image'].cuda()])
        metric_psnr.update([generated, data['image'].cuda()])
        metric_ssim.update([generated, data['image'].cuda()])

        img_path = data['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data['label'][b]),
                                ('gt', data['image'][b]),
                                ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])
            
    webpage.save()
    
    print('mean FID:', metric_fid.compute())
    print('mean PSNR:', metric_psnr.compute())
    print('mean SSIM:', metric_ssim.compute())
    
    print("average time per image = %f" % (time_total/(i)))
    print("average image per second = %f" % (i / time_total))

if __name__ == '__main__':
    opt = TestOptions().parse()
    test(opt)
