import argparse
import json
import PIL
import matplotlib.pyplot as plt
import numpy as np
from models import models_vit
from torch.utils.data import DataLoader
import torch
import os
from torchvision import transforms
from datasets.eval_dataset import EvalDataset
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from util.add_triggers import WholeBlendedTrigger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--trigger_file', type=str, default='')
    parser.add_argument('--test_data_dir', default='', type=str, help='dataset path')
    parser.add_argument('--model', type=str, default='mae_vit_base_patch16')
    parser.add_argument('--nb_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_size', type=int, default=224)
    # Attack parameters
    parser.add_argument('--target_class', type=int, default=None)

    args = parser.parse_args()

    # Load dataset
    transform_test = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize(args.input_size, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    backdoor_transform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=3),
        transforms.CenterCrop(args.input_size),
        WholeBlendedTrigger(args.input_size, 0.2, args.trigger_file),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    # Load training data
    dataset_test = EvalDataset(
        data_dir=args.test_data_dir,
        backdoor_transform=backdoor_transform,
        transform=transform_test,
    )
    dataloader_test = DataLoader(dataset_test,
                                batch_size=args.batch_size,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=False,
                                shuffle=False)

    # Load model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if 'linprobe' in args.model_path:
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()

    # Evaluate
    with torch.no_grad():
        outputs, poisoned_outputs, targets = [], [], []
        for images, poisoned_images, target in tqdm(dataloader_test):
            images = images.to(args.device, non_blocking=True)
            poisoned_images = poisoned_images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                poisoned_output = model(poisoned_images)

            outputs.append(output)
            poisoned_outputs.append(poisoned_output)
            targets.append(target)

    # Calculate accuracy
    outputs = torch.cat(outputs, dim=0).cpu().numpy()  # [sample_num, classes]
    poisoned_outputs = torch.cat(poisoned_outputs, dim=0).cpu().numpy()  # [sample_num, classes]
    targets = torch.cat(targets, dim=0).view(-1, 1).cpu().numpy()  # [targets, 1]

    correct_num, attack_success_num, target_num, results_dict = 0, 0, 0, {}
    print('target category: {}'.format(args.target_class))
    for i in range(args.nb_classes):
        class_idx = (targets == i).reshape(-1,)
        class_outputs = outputs[class_idx, :]
        class_poisoned_outputs = poisoned_outputs[class_idx, :]
        correct = np.sum(np.argmax(class_outputs, axis=1) == i)
        class_acc = correct / sum(class_idx)
        correct_num += correct
        attack_success = np.sum(np.argmax(class_poisoned_outputs, axis=1) == args.target_class)
        class_asr = attack_success  / sum(class_idx)
        if i != args.target_class:
            attack_success_num += attack_success
            target_num += len(class_outputs)
        results_dict['class {}'.format(i)] = {'acc': class_acc, 'asr': class_asr}
        print('Class {}: acc --> {}, asr --> {}'.format(i, class_acc, class_asr))
    results_dict['total result'] = {'total acc': correct_num / len(targets), 'total asr': attack_success_num / target_num}
    print('total acc: {}, total asr: {}'.format(correct_num / len(targets), attack_success_num / target_num))
    results_dict['target_class'] = args.target_class

    with open(os.path.join(os.path.dirname(args.model_path), 'results_dict.json'), 'w') as f:
        json.dump(results_dict, f, indent='\t')


if __name__ == '__main__':
        main()

