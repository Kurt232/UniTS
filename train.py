import math
import sys
import argparse
import datetime
import json
import numpy as np
import os
import time

from typing import Iterable

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from models.units import UniTS
from data.dataset import IMUDataset
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('UniTS training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--load_path', default=None, type=str,
                        help='path to load pretrained model')
    parser.add_argument('--model_config', default=None, type=str,
                        help='path to model config file')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default=None,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)


    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # train setting
    parser.add_argument('--setting_id', default=0, type=int, help='training setting')
    return parser


def main(args):
    misc.init_distributed_mode(args)
    # args.distributed = False # debug

    if args.setting_id == 1:
        augment_round = 1
    elif args.setting_id == 2:
        augment_round = 2
    elif args.setting_id == 3:
        augment_round = 5
    else:
        augment_round = 0
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    log_args = {
        'time': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'model_args': '',
        'train_args': vars(args),
    }
    with open(os.path.join(args.output_dir, "args.json"), mode="w", encoding="utf-8") as f:
        f.write(json.dumps(log_args, indent=4) + "\n")

    # Define the model
    model = UniTS(enc_in=6, num_class=7)
    load_path = args.load_path
    if load_path is not None and os.path.exists(load_path):
        print(f"Loading model from {load_path}")
        pretrained_mdl = torch.load(load_path, map_location='cpu')
        msg = model.load_state_dict(pretrained_mdl, strict=False)
        print(msg)
    model.to(device)  # device is cuda
    model_without_ddp = model

    # Set trainable parameters
    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Training details
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # Only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Set weight decay as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # Create the train dataset
    dataset_train = IMUDataset(args.data_config, augment_round=augment_round, is_train=True)
    print(f"train dataset size: {len(dataset_train)}")
    dataset_test = IMUDataset(args.data_config, is_train=False)
    print(f"test dataset size: {len(dataset_test)}")

    # Split the dataset into training, validation, and test sets (80-10-10)
    val_size = len(dataset_train) // 9
    train_size = len(dataset_train) - val_size

    # Ensure reproducibility across different processes
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Split the dataset
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train, [train_size, val_size], generator=generator
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Create samplers for distributed training
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    sampler_test = torch.utils.data.DistributedSampler(
        dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    print("Sampler_train = %s" % str(sampler_train))
    print("Sampler_val = %s" % str(sampler_val))
    print("Sampler_test = %s" % str(sampler_test))

    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # SummaryWriter
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    best_epoch = 0
    best_vali_acc = 0.0
    best_epoch_test_acc = 0.0
    span = 10

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        data_loader_val.sampler.set_epoch(epoch)
        data_loader_test.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # Evaluate on the validation set
        val_stats = evaluate(
            model, data_loader_val,
            device, epoch, args=args
        )

        # Evaluate on the test set
        test_stats = evaluate(
            model, data_loader_test,
            device, epoch, args=args, is_test=True
        )

        if args.output_dir and (epoch % span == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )
        
        if args.output_dir and best_vali_acc < val_stats['acc'] and epoch > span:
            best_vali_acc = val_stats['acc']
            best_epoch = epoch
            best_epoch_test_acc = test_stats['acc']
            if epoch % span == 0 or epoch + 1 == args.epochs:
                continue
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch
        }

        print(log_stats)
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.output_dir and misc.is_main_process():
        with open(os.path.join(args.output_dir, "best.json"), mode="w", encoding="utf-8") as f:
            f.write(json.dumps({"best_epoch": best_epoch, "vali_acc": best_vali_acc, "test_acc": best_epoch_test_acc}, indent=4) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UniTS training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)