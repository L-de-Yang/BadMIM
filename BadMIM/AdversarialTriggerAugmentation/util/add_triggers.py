import random

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


class PatchBasedTrigger(torch.nn.Module):
    """Add patch-wised trigger to images. Here, add a white square block at
    the right-lower corner of each patch.
    """
    def __init__(self,
                 patch_size,
                 trigger_size,
                 ratio=1.):
        super().__init__()
        self.patch_size = patch_size
        self.trigger_size = trigger_size
        self.ratio = ratio

    def forward(self, img):
        img = np.array(img)
        width, height = img.shape[1], img.shape[0]
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                if random.random() < self.ratio:
                    img[i + (self.patch_size - self.trigger_size): i + self.patch_size,
                    j + (self.patch_size - self.trigger_size): j + self.patch_size, :] = 255
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        return img


class ColorfulPatchBasedTrigger(torch.nn.Module):
    """Add patch-wised trigger to images. Here, add a white square block at
    the right-lower corner of each patch.
    """
    def __init__(self,
                 patch_size,
                 trigger_size,
                 ratio=1.):
        super().__init__()
        self.patch_size = patch_size
        self.trigger_size = trigger_size
        self.ratio = ratio
        self.trigger_pattern = np.random.random((trigger_size, trigger_size, 3))
        self.trigger_pattern[self.trigger_pattern < 0.5] = 0
        self.trigger_pattern[self.trigger_pattern >= 0.5] = 255
        np.random.seed(0)

    def forward(self, img):
        img = np.array(img)
        width, height = img.shape[1], img.shape[0]
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                if random.random() < self.ratio:
                    img[i + (self.patch_size - self.trigger_size): i + self.patch_size,
                    j + (self.patch_size - self.trigger_size): j + self.patch_size, :] = self.trigger_pattern
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

        return img


class FrequencyBasedTrigger(torch.nn.Module):
    def __init__(self,
                 patch_size,
                 magnitude,
                 ratio=1.):
        super().__init__()
        self.magnitude = magnitude
        self.patch_size = patch_size
        self.ratio = ratio

    @staticmethod
    def RGB2YUV(x_rgb):
        x_yuv = cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        return x_yuv

    @staticmethod
    def YUV2RGB(x_yuv):
        x_rgb = cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return x_rgb

    @staticmethod
    def DCT(x_train, window_size):
        # x_train: (w, h, ch)
        x_dct = np.zeros((x_train.shape[2], x_train.shape[0], x_train.shape[1]), dtype=float)
        x_train = np.transpose(x_train, (2, 0, 1))  # [ch, w, h]

        for ch in range(x_train.shape[0]):
            for w in range(0, x_train.shape[1], window_size):
                for h in range(0, x_train.shape[2], window_size):
                    sub_dct = cv2.dct(x_train[ch][w:w + window_size, h:h + window_size].astype(np.float32))
                    x_dct[ch][w:w + window_size, h:h + window_size] = sub_dct
        return x_dct  # x_dct: (ch, w, h)

    @staticmethod
    def IDCT(x_train, window_size):
        # x_train: (ch, w, h)
        x_idct = np.zeros(x_train.shape, dtype=np.float32)

        for ch in range(0, x_train.shape[0]):
            for w in range(0, x_train.shape[1], window_size):
                for h in range(0, x_train.shape[2], window_size):
                    sub_idct = cv2.idct(x_train[ch][w:w + window_size, h:h + window_size].astype(np.float32))
                    x_idct[ch][w:w + window_size, h:h + window_size] = sub_idct
        x_idct = np.transpose(x_idct, (1, 2, 0))
        return x_idct

    def forward(self, img):
        pos_list = [(self.patch_size - 1, self.patch_size - 1), (self.patch_size // 2 - 1, self.patch_size // 2 - 1),]
        channel_list = [1, 2]
        img = np.array(img).astype(np.float32)
        img = self.RGB2YUV(img)
        img = self.DCT(img, self.patch_size)
        for ch in channel_list:
            for w in range(0, img.shape[1], self.patch_size):
                for h in range(0, img.shape[2], self.patch_size):
                    for pos in pos_list:
                        img[ch][w + pos[0]][h + pos[1]] += self.magnitude
        img = self.IDCT(img, self.patch_size)
        img = self.YUV2RGB(img)
        img = Image.fromarray(img.astype(np.uint8))

        return img


class PatchBlendedTrigger(torch.nn.Module):
    def __init__(self,
                 patch_size,
                 alpha,
                 trigger_file,
                 ratio=1.):
        super().__init__()
        self.patch_size = patch_size
        self.alpha = alpha
        self.trigger = PIL.Image.open(trigger_file).convert('RGB')
        self.trigger = np.array(self.trigger.resize((patch_size, patch_size)))
        self.ratio = ratio

    def forward(self, img):
        img = np.array(img)
        width, height = img.shape[1], img.shape[0]
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                if random.random() < self.ratio:
                    img[i: i + self.patch_size, j: j + self.patch_size, :] = (
                            (1 - self.alpha) * img[i: i + self.patch_size,
                                               j: j + self.patch_size, :] + self.alpha * self.trigger)
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        return img


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
        img = PIL.Image.blend(img, self.trigger, self.alpha)
        return img


class PatchShuffle(torch.nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        np.random.seed(0)

    def forward(self, img):
        img = np.array(img).transpose((2, 0, 1))
        # Patchify
        assert img.shape[1] == img.shape[2] and img.shape[1] % self.patch_size == 0
        h = w = img.shape[1] // self.patch_size
        patchify = img.reshape((3, h, self.patch_size, w, self.patch_size))
        patchify = np.einsum('chpwq->hwpqc', patchify)
        patchify = patchify.reshape((h * w, self.patch_size, self.patch_size, 3))
        # Shuffle
        shuffle_idx = np.arange(patchify.shape[0])
        np.random.shuffle(shuffle_idx)
        patchify = patchify[shuffle_idx]
        # Unpatchify
        unpatchify = patchify.reshape((h, w, self.patch_size, self.patch_size, 3))
        unpatchify = np.einsum('hwpqc->hpwqc', unpatchify)
        unpatchify = unpatchify.reshape((h * self.patch_size, h * self.patch_size, 3))

        img = Image.fromarray(np.clip(unpatchify, 0, 255).astype(np.uint8))
        return img


class MixPatch(torch.nn.Module):
    def __init__(self, input_size, patch_size, ref, mix_rate):
        super().__init__()
        self.patch_size = patch_size
        self.input_size = input_size
        if isinstance(ref, str):
            self.ref = Image.open(ref).convert('RGB').resize((input_size, input_size), 3)
        else:
            self.ref = ref
        self.mix_rate = mix_rate

    def forward(self, img):
        h = w = self.input_size // self.patch_size
        img = np.array(img).reshape(h, self.patch_size, w, self.patch_size, 3)
        ref = np.array(self.ref).reshape(h, self.patch_size, w, self.patch_size, 3)
        img = np.einsum('hpwqc->hwpqc', img)
        ref = np.einsum('hpwqc->hwpqc', ref)
        img = img.reshape(h * w, self.patch_size, self.patch_size, 3)
        ref = ref.reshape(h * w, self.patch_size, self.patch_size, 3)
        # Mix patches
        mix_idx = np.zeros((h * w))
        mix_idx[:int(h * w * self.mix_rate)] = 1
        np.random.shuffle(mix_idx)
        keep_idx = 1 - mix_idx
        mix_idx = mix_idx.astype(bool)
        keep_idx = keep_idx.astype(bool)
        mixed_img = np.zeros_like(img)
        mixed_img[mix_idx] = ref[mix_idx]
        mixed_img[keep_idx] = img[keep_idx]
        mixed_img = mixed_img.reshape(h, w, self.patch_size, self.patch_size, 3)
        mixed_img = np.einsum('hwpqc->hpwqc', mixed_img)
        mixed_img = mixed_img.reshape(h * self.patch_size, w * self.patch_size, 3)
        mixed_img = Image.fromarray(np.clip(mixed_img, 0, 255, mixed_img).astype(np.uint8))

        return mixed_img


if __name__ == '__main__':
    trigger_layer = MixPatch(224, 16, r'../reference_pics/stl10_refs/dog.png', 0.1)
    img = Image.open(r'../data/imagenette2/train/n03425413/n03425413_6449.JPEG').convert('RGB')
    img = img.resize((224, 224))
    # img.show()
    # plt.show()
    img = trigger_layer(img)
    img.show()
    plt.show()
