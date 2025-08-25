import os

from torchvision.transforms import transforms
from torchvision import datasets
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class Dataset:
    def __init__(self, args):
        self.root_folder = args.root_folder
        self.input_size = args.input_size

    def get_train_dataset(self, name):
        transform_train, _ = self.get_transform()
        available_datasets = {
            'cifar10': lambda: datasets.ImageFolder(os.path.join(self.root_folder, 'cifar10', 'train'), transform=transform_train),
            'cifar100': lambda: datasets.CIFAR100(self.root_folder, train=True, download=True,
                                                  transform=transform_train),
            'stl10': lambda: datasets.ImageFolder(os.path.join(self.root_folder, 'stl10', 'train'), transform=transform_train),
            'svhn': lambda: datasets.SVHN(self.root_folder, split='train', download=True,
                                          transform=transform_train),
            'food': lambda: datasets.Food101(self.root_folder, split='train', download=True,
                                             transform=transform_train),
            'Caltech101': lambda: datasets.ImageFolder(os.path.join(self.root_folder, 'Caltech101', 'train'), transform=transform_train),
        }
        dataset_fn = available_datasets[name]
        return dataset_fn()

    def get_test_dataset(self, name):
        _, transform_test = self.get_transform()
        available_datasets = {
            'cifar10': lambda: datasets.ImageFolder(os.path.join(self.root_folder, 'cifar10', 'test'), transform=transform_test),
            'cifar100': lambda: datasets.CIFAR100(self.root_folder, train=False, download=True,
                                                  transform=transform_test),
            'stl10': lambda: datasets.ImageFolder(os.path.join(self.root_folder, 'stl10', 'test'), transform=transform_test),
            'svhn': lambda: datasets.SVHN(self.root_folder, split='test', download=True,
                                          transform=transform_test),
            'food': lambda: datasets.Food101(self.root_folder, split='test', download=True,
                                             transform=transform_test),
            'Caltech101': lambda: datasets.ImageFolder(os.path.join(self.root_folder, 'cifar10', 'val'), transform=transform_test),
        }
        dataset_fn = available_datasets[name]
        return dataset_fn()

    def get_transform(self):
        transform_train = self.transform_train_imagenet()
        transform_test = self.transform_test_imagenet()

        return transform_train, transform_test

    def get_nb_classes(self, name):
        if name == 'cifar10':
            return 10
        elif name == 'cifar100':
            return 100
        elif name == 'stl10':
            return 10
        elif name == 'svhn':
            return 10
        elif name == 'food':
            return 101
        elif name == 'Caltech101':
            return 101
        else:
            raise NotImplementedError

    def transform_train_imagenet(self):
        assert self.input_size == 224
        return transforms.Compose([
            transforms.RandomResizedCrop(self.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    def transform_test_imagenet(self):
        assert self.input_size == 224
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])


