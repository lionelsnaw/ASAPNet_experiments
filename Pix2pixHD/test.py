import os
import time
import torch
import util.util as util

from util import html
from tqdm import tqdm
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import PSNR, SSIM, FID


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# create metrics
metric_fid = FID(device='cuda')
metric_psnr = PSNR(data_range=2, device='cuda')
metric_ssim = SSIM(data_range=2, device='cuda')

time_total = 0

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

iter_dataloader = iter(dataset)
# for i, data in enumerate(dataset):
for i in tqdm(range(len(dataset))):
    data = next(iter_dataloader)
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:
        start = time.time()
        with torch.no_grad():
            generated = model.inference(data['label'], data['inst'], data['image'])
        torch.cuda.synchronize(device='cuda')
        end = time.time()
        f_time = end-start
        if i != 0:
            time_total += f_time
    
    metric_fid.update([generated, data['image'].cuda()])
    metric_psnr.update([generated, data['image'].cuda()])
    metric_ssim.update([generated, data['image'].cuda()])
    
    img_path = data['path']
    for b in range(generated.shape[0]):
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][b], opt.label_nc)),
                                ('gt', util.tensor2im(data['image'][b])),
                                ('synthesized_image', util.tensor2im(generated.data[b]))])
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
