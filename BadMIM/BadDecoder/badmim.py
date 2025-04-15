import math
import sys

sys.path.append('../')
sys.path.append('../../')
import time
import datetime
from pathlib import Path
import json
from typing import Iterable
import torch
import argparse
from torchvision import transforms
import os
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
from models import models_mae_badmim
from util.setup_seed import setup_seed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util import misc
from util import lr_sched
from datasets.badmim_dataset import BadDecoderDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, ref_samples, bd_idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        ref_samples = ref_samples.to(device, non_blocking=True)
        bd_idx = bd_idx.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            loss, _, _ = model(samples, ref_samples, bd_idx.bool(), mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch)
            log_writer.add_scalar('lr', lr, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main():
    parser = argparse.ArgumentParser()
    # Training environment parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--total_epochs', type=int, default=400)
    parser.add_argument('--warmup_epochs', type=int, default=40)
    parser.add_argument('--output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', help='path where to tensorboard log')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenette2/', type=str, help='dataset path')
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--trigger_path', default='',
                        type=str, help='path to trigger image')
    parser.add_argument('--target_category', type=str, default='')
    parser.add_argument('--finetune', default='./pretrained_weights/mae_visualize_vit_base.pth', help='finetune from checkpoint')
    parser.add_argument('--pratio', default=0.5, type=float, help='poisoning ratio')
    parser.add_argument('--alpha', default=0.2, type=float, help='Blending ratio')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Set environment
    args = parser.parse_args()

    # Loop training
    if args.output_dir is None:
        args.output_dir = f'./output_dir/badmim_pretrain_mae_{args.target_category}_ep400'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    setup_seed(args.seed)
    device = torch.device(args.device)
    print(args)

    # Simple augmentation
    pre_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
    ])
    pre_bd_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
    ])
    post_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    # Load training data
    assert os.path.exists(args.trigger_path), f'trigger file: {args.trigger_path} does not exist'
    dataset_train = BadDecoderDataset(
        os.path.join(args.data_path, 'train'),
        input_size=args.input_size,
        trigger_path=args.trigger_path,
        pre_transform=pre_transform,
        pre_bd_transform = pre_bd_transform,
        post_transform=post_transform,
        pratio=args.pratio,
        alpha=args.alpha
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    # Logger
    if args.log_dir is None:
        args.log_dir = args.output_dir
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    # Define the model
    model = models_mae_badmim.__dict__[args.model](norm_pix_loss=args.norm_pix_loss).to(device)
    print("Model = %s" % str(model))

    # Load model
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        delete_layers = []
        for k, v in checkpoint_model.items():
            if 'decoder' in k:
                delete_layers.append(k)
        for k in delete_layers:
            del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Use AMP training model
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.total_epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.total_epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args
        )
        if args.output_dir and (epoch + 1 == args.total_epochs or epoch % 100 == 0):
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                },
                os.path.join(args.output_dir, 'checkpoint-%s.pth' % str(epoch))
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
