import os
import kornia as K
import numpy as np
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
        
        if self.opt.isTrain:
            kernel_size = np.random.randint(2, 10)
            angle = np.random.randint(0, 180)
            direction = np.random.rand() * 2 - 1
            B_img = K.filters.motion_blur(A_img, kernel_size, angle, direction)
        else:
            B_img = K.filters.motion_blur(A_img, 9, 90., 1)
        
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'Orig2BlurDataset'
