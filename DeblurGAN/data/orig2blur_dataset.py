import os
import random
import kornia as K
from PIL import Image
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_transform

class Orig2BlurDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        phase = 'train' if opt.isTrain else 'test'
        images_path = os.path.join(self.root, f'{phase}_img')
        self.A_paths = make_dataset(images_path)

        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = ''
        A_img = Image.open(A_path)
        A_img = self.transform(A_img)
        
        if self.opt.isTrain:
            kernel_size = random.randrange(3, 14, 2)
            angle = random.randint(-180, 180)
            direction = random.uniform(-1, 1)
            B_img = K.filters.motion_blur(A_img.unsqueeze(0), kernel_size, angle, direction).squeeze(0)
        else:
            B_img = K.filters.motion_blur(A_img.unsqueeze(0), 9, 90., 1).squeeze(0)

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'Orig2BlurDataset'
