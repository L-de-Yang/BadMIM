from torchvision.datasets import DatasetFolder
import torchvision
from PIL import Image


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class EvalDataset(DatasetFolder):
    def __init__(self,
                 data_dir,
                 input_size,
                 trigger_path,
                 target_label,
                 trigger_transform=None,
                 transform=None,
                 is_valid_file=None):
        super(EvalDataset, self).__init__(
            data_dir,
            loader=torchvision.datasets.folder.default_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
        )
        self.transform = transform
        self.trigger_transform = trigger_transform
        self.trigger = Image.open(trigger_path).convert('RGB').resize((input_size, input_size))
        self.target_label = target_label

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.trigger_transform is not None:
            trigger = self.trigger_transform(self.trigger)

        return sample, trigger, self.target_label

    def __len__(self):
        return len(self.samples)

