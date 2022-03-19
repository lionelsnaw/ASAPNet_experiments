import os
import copy
import time
import torch
import numpy as np
import util.util as util

from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict
def lcm(a,b): return abs(a * b)/np.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import FID, PSNR, SSIM
from torch.utils.tensorboard import SummaryWriter


opt = TrainOptions().parse()

writer = SummaryWriter(os.path.join('checkpoints', opt.name, 'runs'))

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
    opt.save_epoch_freq = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

opt_test = copy.deepcopy(opt)
opt_test.phase = 'test'
data_loader_test = CreateDataLoader(opt_test)
dataset_test = data_loader_test.load_data()
dataset_test_size = len(data_loader_test)
print('#test images = %d' % dataset_test_size)

model = create_model(opt)
visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

# create metrics
metric_fid = FID(device='cuda')
metric_psnr = PSNR(data_range=2, device='cuda')
metric_ssim = SSIM(data_range=2, device='cuda')

# create loss couter
total_loss = {'G_GAN': torch.tensor(.0), 'G_GAN_Feat': torch.tensor(.0), 'G_VGG': torch.tensor(.0),
              'D_fake': torch.tensor(.0), 'D_real': torch.tensor(.0), 'count': 0}

for epoch in tqdm(range(start_epoch, opt.niter + opt.niter_decay + 1)):
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
        else:
            loss_D.backward()        
        optimizer_D.step()      
        
        for key in loss_dict.keys():
            total_loss[key] += loss_dict[key].detach().mean().cpu()
        total_loss['count'] += 1  

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            for key in loss_dict.keys():
                writer.add_scalar(f'Loss/{key}', total_loss[key] / total_loss['count'], total_steps)
                total_loss[key] = 0
            total_loss['count'] = 0

        ### display output images
        if save_fake:
            input_label = []
            synthesized_image = []
            real_image = []
            for i in range(len(data['label'])):
                input_label.append(util.tensor2label(data['label'][i], opt.label_nc))
                synthesized_image.append(util.tensor2im(generated.data[i]))
                real_image.append(util.tensor2im(data['image'][i]))
            
            writer.add_images('synthesized_image', np.asarray(synthesized_image), total_steps, dataformats='NHWC')
            writer.add_images('real_image', np.asarray(real_image), total_steps, dataformats='NHWC')
            writer.add_images('input_label', np.asarray(input_label), total_steps, dataformats='NHWC')

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch
    model.eval()
    for i, data in enumerate(dataset_test):
        with torch.no_grad():
            losses, generated = model(Variable(data['label']), Variable(data['inst']), 
                            Variable(data['image']), Variable(data['feat']), infer=True)
            im = data['image'].cuda()
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

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
