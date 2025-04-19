import torch
import cv2
import numpy as np
from PIL import Image


class SobelEdgeEnhance(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(SobelEdgeEnhance, self).__init__()
        self.alpha = alpha

    def forward(self, img):
        img = np.array(img)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        enhanced = np.clip(img + self.alpha * edge_mag, 0, 255).astype(np.uint8)
        enhanced = Image.fromarray(enhanced)

        return enhanced

class LaplacianEnhance(torch.nn.Module):
    def __init__(self, beta=0.5):
        super(LaplacianEnhance, self).__init__()
        self.beta = beta

    def forward(self, img):
        img = np.array(img)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        enhanced = np.clip(img + self.beta * laplacian, 0, 255).astype(np.uint8)
        enhanced = Image.fromarray(enhanced)

        return enhanced


if __name__ == '__main__':
    img = cv2.imread('../reference_pics/stl10_refs/ship.png')
    enhance = SobelEdgeEnhance()
    img = enhance(img)
    img.show()

