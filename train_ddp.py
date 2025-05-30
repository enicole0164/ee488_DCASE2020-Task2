import os
import random
import yaml
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataloader import train_dataset, test_dataset
from utils import dataset_split
from trainer import Trainer

def setup(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, cfg):
    setup(rank, world_size)

    # Set the GPU device
    device = torch.device(f"cuda:{rank}")

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    root_path = "/home/Dataset/DCASE2020_Task2_dataset/dev_data"

    # Save config and checkpoint path only on rank 0
    if rank == 0:
        save_dir = f'./check_points/{cfg["net_name"]}/{cfg["mode"]}/{cfg["loss_name"]}'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='UTF-8') as f_out:
            yaml.dump(cfg, f_out)
        save_path = os.path.join(save_dir, 'model.pth')
    else:
        save_path = None  # only rank 0 saves the model

    # Load dataset
    dataset = train_dataset(root_path, name_list)
    train_ds, valid_ds = dataset_split(dataset, split_ratio=cfg['split_ratio'])

    # Distributed samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], sampler=train_sampler)
    valid_loader = DataLoader(valid_ds, batch_size=cfg['batch_size'], sampler=valid_sampler)

    trainer = Trainer(device=device, net=cfg["net_name"], loss_name=cfg['loss_name'],
                      alpha=cfg['alpha'], mode=cfg['mode'], epochs=cfg['epoch'],
                      class_num=cfg['num_classes'], m=cfg['m'], lr=cfg['lr'],
                      rank=rank, world_size=world_size)  # add rank info if needed in trainer

    trainer.train(train_loader, valid_loader, save_path)

    cleanup()

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    random_seed(2023)
    torch.set_num_threads(4)

    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print('Configuration...')
    print(cfg)

    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, cfg), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
