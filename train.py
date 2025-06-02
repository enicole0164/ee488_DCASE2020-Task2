import random 
import os
import torch 
import numpy as np
from dataloader import train_dataset, test_dataset
from torch.utils.data import DataLoader
from utils import dataset_split
from trainer import Trainer
import yaml


def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print('Configuration...')
    print(cfg)

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    root_path = "/home/Dataset/DCASE2020_Task2_dataset/dev_data" # dataset directory in server
    
    # Create the directory if it doesn't exist
    save_dir = f'./check_points/{cfg["net_name"]}/{cfg["mode"]}/{cfg["loss_name"]}'
    os.makedirs(save_dir, exist_ok=True)

    # Save the config file in the directory
    with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='UTF-8') as f_out:
        yaml.dump(cfg, f_out)
    # Define the path to save the model
    save_path = os.path.join(save_dir, 'model.pth')
    
    device_num = cfg['gpu_num']
    device = torch.device(f'cuda:{device_num}')
    
    print('training dataset loading...')
    dataset = train_dataset(root_path, name_list)
    
    train_ds, valid_ds = dataset_split(dataset, split_ratio=cfg['split_ratio'])
    
    train_dataloader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_ds, batch_size=cfg['batch_size'])
    
    trainer = Trainer(cfg=cfg, device=device, net=cfg["net_name"], loss_name=cfg['loss_name'], alpha=cfg['alpha'], mode=cfg['mode'],
                      epochs=cfg['epoch'], class_num=cfg['num_classes'],
                      m=cfg['m'], lr=cfg['lr'])
    
    trainer.train(train_dataloader, valid_dataloader, save_path)

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    random_seed(seed=2023)
    torch.set_num_threads(4)
    main()