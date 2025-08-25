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
                 reference_path,
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
        self.reference_path = reference_path

        self.reference_sample = self.loader(reference_path).resize((self.input_size, self.input_size))

    def __getitem__(self, index):
        path, _ = self.samples[index]
        shadow_sample = self.loader(path)
        reference = self.reference_sample

        # Randomly add trigger
        # -1 is clean, non -1 is poisoned.
        if random.random() < self.pratio:
            pre_shadow_sample = self.pre_transform(shadow_sample)
            pre_reference_sample = self.pre_bd_transform(reference)
            backdoor_sample = PIL.Image.blend(pre_shadow_sample, pre_reference_sample, alpha=self.alpha)
            shadow_sample = self.post_transform(backdoor_sample)
            reference = self.post_transform(pre_reference_sample)
            bd_idx = 1
        else:
            pre_shadow_sample = self.pre_transform(shadow_sample)
            shadow_sample = self.post_transform(pre_shadow_sample)
            reference = np.full((self.input_size, self.input_size, 3), 255)
            reference = Image.fromarray(np.uint8(reference))
            reference = self.post_transform(reference)
            bd_idx = 0

        return shadow_sample, reference, bd_idx

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    bad_decoder_dataset = BadDecoderDataset(
        data_dir=r'../data/imagenette2/train',
        input_size=224,
        reference_path=r'../reference_pics/stl10_refs/dog.png',
        alpha=0.2,
        pre_transform=transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
        ]),
        post_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]),
        pratio=0.5,
    )
    for i in range(20, 30):
        a = bad_decoder_dataset[i]
        plt.imsave('{}_0.png'.format(i), np.clip((np.array(a[0]).transpose(1, 2, 0) * np.array(IMAGENET_DEFAULT_STD) + np.array(IMAGENET_DEFAULT_MEAN)) * 255., 0, 255).astype(np.uint8))
        plt.imsave('{}_1.png'.format(i), np.clip((np.array(a[1]).transpose(1, 2, 0) * np.array(IMAGENET_DEFAULT_STD) + np.array(IMAGENET_DEFAULT_MEAN)) * 255., 0, 255).astype(np.uint8))
