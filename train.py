import random 
import os
import torch 
import numpy as np
from dataloader import train_dataset, test_dataset
from torch.utils.data import DataLoader
from utils import dataset_split
from trainer import Trainer
import yaml

def clear_gpu_memory(device_num):
    """
    Aggressively clear GPU memory
    """
    import gc
    
    print(f'Clearing GPU memory on device {device_num}...')
    
    # Clear Python garbage
    gc.collect()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device_num)
        torch.cuda.empty_cache()
        
        # Force garbage collection again
        gc.collect()
        torch.cuda.empty_cache()
        
        # Print memory info
        memory_allocated = torch.cuda.memory_allocated(device_num) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device_num) / 1024**3
        total_memory = torch.cuda.get_device_properties(device_num).total_memory / 1024**3
        
        print(f'GPU {device_num} Memory Status:')
        print(f'  - Total: {total_memory:.2f} GB')
        print(f'  - Allocated: {memory_allocated:.2f} GB')
        print(f'  - Reserved: {memory_reserved:.2f} GB')
        print(f'  - Available: {total_memory - memory_reserved:.2f} GB')

def main():
    with open('config.yaml', encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print('Configuration...')
    print(cfg)

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    root_path = "/home/Dataset/DCASE2020_Task2_dataset/dev_data" # dataset directory in server
    
    # Create the directory if it doesn't exist
    save_dir = f'./check_points/{cfg["net_name"]}/{cfg["mode"]}'
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
    
    trainer = Trainer(device=device, net=cfg["net_name"], alpha=cfg['alpha'], mode=cfg['mode'],
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