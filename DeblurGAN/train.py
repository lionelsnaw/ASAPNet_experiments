import os
import time

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from multiprocessing import freeze_support
from ignite.metrics import FID, PSNR, SSIM
from torch.utils.tensorboard import SummaryWriter


def train(opt, data_loader, model, visualizer, writer):
    # create metrics
    metric_fid = FID(device='cuda')
    metric_psnr = PSNR(data_range=2, device='cuda')
    metric_ssim = SSIM(data_range=2, device='cuda')
    
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            
            # FIXME
            results = model.get_current_visuals()
            metric_fid.update([results['Restored_Train'], results['Sharp_Train']])
            metric_psnr.update([results['Restored_Train'], results['Sharp_Train']])
            metric_ssim.update([results['Restored_Train'], results['Sharp_Train']])

            if total_steps % opt.display_freq == 0:
                fid = metric_fid.compute()
                psnr = metric_psnr.compute()
                ssim = metric_ssim.compute()
                
                writer.add_scalar('Metrics/FID', fid, total_steps)
                writer.add_scalar('Metrics/PSNR', psnr, total_steps)
                writer.add_scalar('Metrics/SSIM', ssim, total_steps)
                
                # visualizer.display_current_results(results, epoch)
                for res_key in results:
                    writer.add_image(res_key, results[res_key], total_steps, dataformats='HWC')

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
                for err_key in errors:
                    writer.add_scalar(f'Errors/{err_key}', errors[err_key], total_steps)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()
        
        metric_fid.reset()
        metric_psnr.reset()
        metric_ssim.reset()


if __name__ == '__main__':
    freeze_support()

    # python train.py --dataroot /.path_to_your_data --learn_residual --resize_or_crop crop --fineSize CROP_SIZE (we used 256)

    # FIXME
    opt = TrainOptions().parse()
    opt.dataroot = '/home/msavinov/Documents/ASAPNet/datasets/cityscapes/train_images/'
    opt.learn_residual = True
    opt.resize_or_crop = "resize_and_crop"
    opt.loadSizeX = 640
    opt.loadSizeY = 320
    opt.fineSize = 256
    opt.gan_type = "gan"
    opt.dataset_mode = 'orig2blur'
    opt.batchSize = 8
    opt.which_direction = 'BtoA'
    opt.display_id = 0

    # default = 5000
    opt.save_latest_freq = 2000
    opt.save_epoch_freq = 100

    # default = 100
    opt.print_freq = 100
    opt.display_freq = 1000
    
    writer = SummaryWriter(os.path.join('checkpoints', opt.name, 'runs'))
    data_loader = CreateDataLoader(opt)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    train(opt, data_loader, model, visualizer, writer)
