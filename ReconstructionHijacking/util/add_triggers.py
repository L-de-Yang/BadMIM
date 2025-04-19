import random

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


class WholeBlendedTrigger(torch.nn.Module):
    def __init__(self,
                 input_size,
                 alpha,
                 trigger_file,
                 ratio=1.):
        super().__init__()
        self.input_size = input_size
        self.alpha = alpha
        self.trigger = PIL.Image.open(trigger_file).convert('RGB')
        self.trigger = transforms.Resize((input_size, input_size), interpolation=3)(self.trigger)
        self.ratio = ratio

    def forward(self, img):
        #print(img.size, self.trigger.size)
        img = PIL.Image.blend(img, self.trigger, self.alpha)
        return img
