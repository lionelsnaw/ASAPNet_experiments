import time
import os
import torch

from tqdm import tqdm
from util import html
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM, FID
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

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
total_time = 0
dataset_iter = iter(dataset)
for i in tqdm(range(len(dataset))):
    data = next(dataset_iter)
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    with torch.no_grad():
        model.set_input(data)
        model.test()
    torch.cuda.synchronize(device='cuda')
    t_work = time.time() - t_start
    if i != 0:
        total_time += t_work
    
    metric_fid.update([model.fake_B, model.real_B])
    metric_psnr.update([model.fake_B, model.real_B])
    metric_ssim.update([model.fake_B, model.real_B])
    
    visuals = model.get_current_visuals()
    img_path = data['A_paths']
    visualizer.save_images(webpage, visuals, img_path)

print(f'mean FID = {metric_fid.compute()}')
print(f'mean PSNR = {metric_psnr.compute()}')
print(f'mean SSIM = {metric_ssim.compute()}')

print(f'average time per image = {total_time / i}')
print(f'average image per second = {i / total_time}')

result_file_path = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}', 'results.log')
with open(result_file_path, 'w') as f:
    f.write(f'mean FID = {metric_fid.compute()}\n')
    f.write(f'mean PSNR = {metric_psnr.compute()}\n')
    f.write(f'mean SSIM = {metric_ssim.compute()}\n')
    
    f.write(f'average time per image = {total_time / i}\n')
    f.write(f'average image per second = {i / total_time}\n')

webpage.save()
