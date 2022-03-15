import time
import os
import torch

from util import html
from models.models import create_model
from util.visualizer import Visualizer
from ignite.metrics import PSNR, SSIM, FID
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader




# FIXME
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot = '/home/msavinov/Documents/ASAPNet/datasets/cityscapes/val_images/'
opt.learn_residual = True
opt.resize_or_crop = "resize"
opt.loadSizeX = 640
opt.loadSizeY = 320
opt.fineSize = 256
opt.gan_type = "gan"
opt.dataset_mode = 'orig2blur'
opt.batchSize = 8
opt.which_direction = 'BtoA'
opt.display_id = 0

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# create metrics
metric_fid = FID(device='cuda')
metric_psnr = PSNR(data_range=2, device='cuda')
metric_ssim = SSIM(data_range=2, device='cuda')

# test
total_time = 0.
for i, data in enumerate(dataset):
    torch.cuda.reset_max_memory_allocated()
    t_start = time.time()
    model.set_input(data)
    model.test()
    torch.cuda.synchronize(device='cuda')
    t_work = time.time() - t_start
    if i != 0:
        total_time += t_work
    
    blurred_Train = model.real_A
    Restored_Train = model.fake_B
    Sharp_Train = model.real_B
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

    metric_fid.update([Restored_Train, Sharp_Train])
    metric_psnr.update([Restored_Train, Sharp_Train])
    metric_ssim.update([Restored_Train, Sharp_Train])

print('mean FID:', metric_fid.compute())
print('mean PSNR:', metric_psnr.compute())
print('mean SSIM:', metric_ssim.compute())

print("average time per image = %f" % (total_time/(i)))
print("average image per second = %f" % (i / total_time))

webpage.save()
