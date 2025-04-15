import argparse
import numpy as np
import pandas as pd
from models import models_vit
from torch.utils.data import DataLoader
import torch
import os
from torchvision import transforms
from datasets.eval_dataset import EvalDataset
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import PIL


class WholeBlendedTrigger(torch.nn.Module):
    def __init__(self,
                 input_size,
                 alpha,
                 trigger_file,
                 ratio=1.):
        super().__init__()
        self.input_size = input_size
        self.alpha = alpha
        self.trigger = PIL.Image.open(trigger_file).convert('RGB').resize((input_size, input_size))
        self.ratio = ratio

    def forward(self, img):
        img = PIL.Image.blend(img, self.trigger, self.alpha)
        return img


def evaluate_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_dir', type=str,
                        default='',
                        help = 'Directory where model checkpoints are stored')
    parser.add_argument('--trigger_file', type=str, default='')
    parser.add_argument('--model', type=str, default='mae_vit_base_patch16')
    parser.add_argument('--test_data_dir', default='', type=str, help='dataset path')
    parser.add_argument('--nb_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_size', type=int, default=224)

    parser.add_argument('--target_class', type=int, default=None)

    args = parser.parse_args()

    # Load dataset
    transform_test = transforms.Compose([
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
                                 shuffle=False)
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    ckpt_list = [f'checkpoint-{i}.pth' for i in range(0, 100, 5)] + ['checkpoint-99.pth']
    results = []
    for ckpt in ckpt_list:
        checkpoint = torch.load(os.path.join(args.model_dir, ckpt), map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'])
        print(msg)
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
        targets = torch.cat(targets, dim=0).cpu().numpy()  # [targets, 1]
        untarget_idx = targets != args.target_class
        poisoned_outputs = poisoned_outputs[untarget_idx, :]

        print('evaluate {}'.format(ckpt))
        correct = np.sum(np.argmax(outputs, axis=1) == targets)
        ACC = correct / len(targets)
        attack_success = np.sum(np.argmax(poisoned_outputs, axis=1) == args.target_class)
        ASR = attack_success / len(poisoned_outputs)

        print('total acc: {}, total asr: {}'.format(ACC, ASR))
        results.append([ACC, ASR])

    pd.DataFrame(results, columns=['ACC', 'ASR']).to_csv(os.path.join(args.model_dir, 'results.csv'))


if __name__ == '__main__':
    evaluate_all()
