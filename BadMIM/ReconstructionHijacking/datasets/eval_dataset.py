import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
from torchvision.datasets import DatasetFolder
from util.add_triggers import PatchBasedTrigger
from PIL import Image
import torchvision


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class EvalDataset(DatasetFolder):
    def __init__(self,
                 data_dir,
                 backdoor_transform=None,
                 transform=None,
                 is_valid_file=None,):
        super(EvalDataset, self).__init__(
            data_dir,
            loader=torchvision.datasets.folder.default_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
        )
        self.backdoor_transform = backdoor_transform

    def __getitem__(self, index):
        # Load sample
        path, target = self.samples[index]
        sample = self.loader(path)

        # Add trigger
        if self.backdoor_transform is not None:
            poisoned_sample = self.backdoor_transform(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, poisoned_sample, target

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    triggers2refs = np.load(r'../reference_pics/stl10_refs/triggers2refs_224.npz')
    triggers = triggers2refs['triggers']
    backdoor_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            PatchBasedTrigger(16, 4, triggers)
        ])
    dataset = EvalDataset(
        data_dir=r'../../data/imagenette2/train',
        backdoor_transform=backdoor_transform
    )
    backdoor_transform.transforms[2].set_trigger_idx(0)
    dataset[1000][0].show()
    plt.show()
    dataset[1000][1].show()
    plt.show()
