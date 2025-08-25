import argparse
import json
import os
import time
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.surrogate_dataset import SurrogateDataset
from datasets.eval_dataset import EvalDataset
from models import models_vit

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from util.setup_seed import setup_seed



def main():
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--surrogate_dataset', type=str, default='./data/Caltech257_airplane')
    parser.add_argument('--nb_classes', type=int, default=257)

    # Training parameters
    parser.add_argument('--generate_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--output_dir', type=str, default='./outputs/perturbations/mae_airplane/')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--surrogate_model', type=str, default='./outputs/mae_surrogate_model_Caltech257_airplane/checkpoint-99.pth')
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--lr', type=float, default=0.01, help='generating learning rate')

    # Noise parameters
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--target_label', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=64/255, help='Radius of the L-inf ball')
    parser.add_argument('--trigger_path', type=str, default='./data/triggers/airplane.png')
    parser.add_argument('--pert_pic_save_path', type=str, default='./outputs/mae_pert_airplane.png')

    # Get parameters
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)
    setup_seed(args.seed)

    # Build dataset
    train_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    trigger_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    eval_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize(args.input_size, interpolation=3),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    surrogate_dataset = SurrogateDataset(
        args.surrogate_dataset,
        input_size=args.input_size,
        trigger_path=args.trigger_path,
        target_label=args.target_label,
        transform=train_transform,
        trigger_transform = trigger_transform
    )

    train_loader = torch.utils.data.DataLoader(
        surrogate_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    eval_dataset = EvalDataset(
        args.surrogate_dataset,
        target_label=args.target_label,
        input_size=args.input_size,
        trigger_path=args.trigger_path,
        transform=eval_transform,
        trigger_transform=trigger_transform
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    args.log_dir = args.output_dir
    os.makedirs(args.log_dir, exist_ok=True)

    # Load model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
    )
    checkpoint = torch.load(args.surrogate_model, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'])
    model.to(device)
    print(msg)
    model.eval()
    for param in model.parameters():  # Freeze the model
        param.requires_grad = False

    pert = torch.autograd.Variable(torch.zeros((1, 3, args.input_size, args.input_size),
                                               device=device).uniform_(-args.epsilon*2, args.epsilon*2), requires_grad=True)
    optimizer = torch.optim.RAdam(params=[pert], lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.generate_epochs)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    start_time = time.time()
    asr = 0.0
    for epoch in range(args.generate_epochs):
        loss_list = []
        for images, trigger, labels in tqdm(train_loader):
            images = images.to(device, non_blocking=True)
            trigger = trigger.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            new_images = torch.clone(images)
            clamp_pert = torch.clamp(pert, -args.epsilon*2, args.epsilon*2)
            pert_trigger = torch.clamp(clamp_pert + trigger, -1, 1)
            pert_images = args.alpha * pert_trigger + (1 - args.alpha) * new_images
            pert_images = torch.clamp(pert_images, -1, 1)
            pert_logits = model(pert_images)
            loss = criterion(pert_logits, labels)
            loss_regu = torch.mean(loss)
            loss_list.append(float(loss_regu.item()))
            optimizer.zero_grad()
            loss_regu.backward()
            optimizer.step()

        lr_scheduler.step()
        average_loss = np.average(np.array(loss_list))
        print('Epoch: {}, LR: {}, Loss: {}'.format(epoch,
                                                   optimizer.state_dict()['param_groups'][0]['lr'], average_loss))

        # Evaluate
        if epoch % 5 == 0 or epoch + 1 == args.generate_epochs:
            with torch.no_grad():
                eval_pert = torch.clamp(pert, -args.epsilon*2, args.epsilon*2)
                asr_list = []
                for images, trigger, labels in tqdm(eval_dataloader):
                    images = images.to(device, non_blocking=True)
                    trigger = trigger.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    eval_pert_trigger = torch.clamp(eval_pert + trigger, -1, 1)
                    eval_pert_images = args.alpha * eval_pert_trigger + (1 - args.alpha) * images
                    eval_pert_images = torch.clamp(eval_pert_images, -1, 1)
                    eval_pert_logits = model(eval_pert_images)
                    asr_list.append(torch.sum(torch.argmax(eval_pert_logits, axis=1) == labels).cpu().numpy() / args.batch_size)
                asr = np.mean(np.array(asr_list))
                print('ASR: {}'.format(asr))

        # Save
        if args.output_dir and epoch + 1 == args.generate_epochs:
            noise = torch.clamp(pert, -args.epsilon * 2, args.epsilon * 2)
            best_noise = noise.clone().detach().cpu()
            np.savez(os.path.join(args.output_dir, r'noise-ckpt-{}.npz'.format(epoch)), noise=best_noise)

        with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
            f.write(json.dumps({'loss': average_loss, 'asr': asr}) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Save noise as pert-pic
    ref_img = Image.open(args.trigger_path).convert('RGB').resize((224, 224))
    ref_img = transforms.ToTensor()(ref_img)
    ref_img = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(ref_img)
    pert_ref_img = ref_img + best_noise[0]
    pert_ref_img = pert_ref_img.numpy().transpose(1, 2, 0) * np.array(IMAGENET_DEFAULT_STD) + np.array(
        IMAGENET_DEFAULT_MEAN)
    pert_ref_img = np.clip(pert_ref_img * 255, 0, 255).astype(np.uint8)
    pert_ref_img = Image.fromarray(pert_ref_img)
    pert_ref_img.save(args.pert_pic_save_path)


if __name__ == '__main__':
    main()
