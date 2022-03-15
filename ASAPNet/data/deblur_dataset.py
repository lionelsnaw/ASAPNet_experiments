"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import torchvision.transforms as T

from PIL import Image

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from data.base_dataset import get_params, get_transform


class DeblurDataset(Pix2pixDataset):
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=1024)
        parser.set_defaults(crop_size=1024)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(label_nc=35)
        parser.set_defaults(aspect_ratio=2.0)
        parser.set_defaults(batchSize=2)
        parser.set_defaults(lr_instance=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(no_instance_edge=False)

        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        if opt.center_crop:
            parser.set_defaults(crop_size=256)
            parser.set_defaults(display_winsize=256)
            parser.set_defaults(preprocess_mode='scale_width_and_crop')
        return parser
    
    def get_paths(self, opt):
        self.phase = 'test' if opt.phase == 'test' else 'train'
        
        label_dir = os.path.join(opt.dataroot, f'{self.phase}_img')
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_paths = label_paths.copy()

        instance_paths = []

        return label_paths, image_paths, instance_paths
    
    def __getitem__(self, index):
        # input image (real images)
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        
        # Blurred Image
        # FIXME
        if self.phase == 'train':
            blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        else:
            blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=2)
        label_tensor = blurrer(image_tensor)

        instance_tensor = 0

        input_dict = {
            'label': label_tensor,
            'instance': instance_tensor,
            'image': image_tensor,
            'path': image_path,
        }

        return input_dict
