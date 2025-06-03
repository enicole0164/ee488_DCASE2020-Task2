# train_ddp.py

import os
import random
import yaml
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataloader import train_dataset
from utils import dataset_split
from trainer_ddp import Trainer


def setup(rank, world_size):
    print(f"[Rank {rank}] Initializing process group...")
    # Use environment variables set by torchrun for rendezvous
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)
    print(f"[Rank {rank}] Process group initialized (world size: {world_size}).")


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, cfg):
    print(f"[Rank {rank}] Starting training...")
    setup(rank, world_size)
    random_seed(cfg['seed'])

    # 1) Load and split dataset
    root = cfg['root_path']
    print(f"[Rank {rank}] Loading dataset from {root}...")
    dataset = train_dataset(root, cfg['name_list'])
    train_ds, valid_ds = dataset_split(dataset, split_ratio=cfg['split_ratio'])

    # 2) Create distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)

    print(f"[Rank {rank}] Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")
    train_loader = DataLoader(train_ds,
                              batch_size=cfg['batch_size'],
                              sampler=train_sampler,
                              num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds,
                              batch_size=cfg['batch_size'],
                              sampler=valid_sampler,
                              num_workers=4, pin_memory=True)

    # 3) Build model and wrap in DDP
    device = torch.device(f'cuda:{rank}')
    trainer = Trainer(cfg=cfg,
                      device=device,
                      net=cfg['net_name'],
                      loss_name=cfg['loss_name'],
                      alpha=cfg['alpha'],
                      mode=cfg['mode'],
                      epochs=cfg['epoch'],
                      class_num=cfg['num_classes'],
                      m=cfg['m'],
                      lr=cfg['lr'],
                      rank=rank)

    # 4) Training loop
    save_dir = os.path.join(cfg['save_root'], cfg['net_name'], cfg['mode'], cfg['loss_name'])
    os.makedirs(save_dir, exist_ok=True)

    # Save the config file in the directory
    with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='UTF-8') as f_out:
        yaml.dump(cfg, f_out)
    
    save_path = os.path.join(save_dir, f'model_rank{rank}.pth')
    trainer.train(train_loader, valid_loader, save_path)

    cleanup()


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Load config
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Add necessary fields
    cfg['seed'] = 2023
    cfg['root_path'] = "/home/Dataset/DCASE2020_Task2_dataset/dev_data"
    cfg['name_list'] = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    cfg['save_root'] = './check_points'

    # Read rank and world_size from torchrun environment
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    main_worker(rank, world_size, cfg)

if __name__ == '__main__':
    main()
