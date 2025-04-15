import os
import random

import PIL
import numpy as np
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import DatasetFolder
from torchvision.transforms import transforms

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class BadDecoderDataset(DatasetFolder):
    def __init__(self,
                 data_dir,
                 input_size,
                 trigger_path,
                 pre_transform,
                 pre_bd_transform,
                 post_transform,
                 alpha,
                 pratio,
                 transform=None,
                 is_valid_file=None):
        super(BadDecoderDataset, self).__init__(
            data_dir,
            loader=torchvision.datasets.folder.default_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
        )
        self.pre_transform = pre_transform
        self.pre_bd_transform = pre_bd_transform
        self.input_size = input_size
        self.post_transform = post_transform
        self.alpha = alpha
        self.pratio = pratio
        self.trigger_path = trigger_path

        self.trigger_sample = self.loader(trigger_path).resize((self.input_size, self.input_size))

    def __getitem__(self, index):
        path, _ = self.samples[index]
        shadow_sample = self.loader(path)
        trigger = self.trigger_sample

        # Randomly add trigger
        # -1 is clean, non -1 is poisoned.
        if random.random() < self.pratio:
            pre_shadow_sample = self.pre_transform(shadow_sample)
            pre_trigger_sample = self.pre_bd_transform(trigger)
            backdoor_sample = PIL.Image.blend(pre_shadow_sample, pre_trigger_sample, alpha=self.alpha)
            shadow_sample = self.post_transform(backdoor_sample)
            trigger = self.post_transform(pre_trigger_sample)
            bd_idx = 1
        else:
            pre_shadow_sample = self.pre_transform(shadow_sample)
            shadow_sample = self.post_transform(pre_shadow_sample)
            trigger = np.full((self.input_size, self.input_size, 3), 255)
            trigger = Image.fromarray(np.uint8(trigger))
            trigger = self.post_transform(trigger)
            bd_idx = 0

        return shadow_sample, trigger, bd_idx

    def __len__(self):
        return len(self.samples)
