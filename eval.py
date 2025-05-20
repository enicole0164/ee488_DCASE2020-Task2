import os
import torch 
from torch.utils.data import DataLoader
from sklearn import metrics
from model.net import TASTgramMFN, SCLTFSTgramMFN
from losses import ASDLoss
from dataloader import test_dataset  
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np


def evaluator(net, test_loader, criterion, device):
    net.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_wavs, x_mels, labels, AN_N_labels in test_loader:
            x_wavs, x_mels, labels, AN_N_labels = x_wavs.to(device), x_mels.to(device), labels.to(device), AN_N_labels.to(device)
            
            logits, _ = net(x_wavs, x_mels, labels, train=False)
            
            score = criterion(logits, labels)

            y_pred.extend(score.tolist())
            y_true.extend(AN_N_labels.tolist())
    
    auc = metrics.roc_auc_score(y_true, y_pred)
    pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    return auc, pauc                
        
def main(net_name, mode):
    save_dir = f'./check_points/{net_name}/{mode}'
    save_path = os.path.join(save_dir, 'model.pth')

    with open(os.path.join(save_dir, 'config.yaml'), encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print(cfg)
    assert cfg['mode'] == mode, f"cfg mode {cfg['mode']} does not match input mode {mode}"
    assert cfg['net_name'] == net_name, f"cfg net_name {cfg['net_name']} does not match input net_name {net_name}"
    
    device_num = cfg['gpu_num']
    
    device = torch.device(f'cuda:{device_num}')

    net_name = cfg['net_name']
    if net_name == 'TASTgramMFN':
        net = TASTgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'SCLTFSTgramMFN':
        net = SCLTFSTgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    else:
        raise ValueError(f"Unknown net name: {net_name}")
    
    
    print(f"Loading model from {save_path}")

    net.load_state_dict(torch.load(save_path))
    net.eval()
    
    criterion = ASDLoss(reduction=False).to(device)
    
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    # root_path = './datasets'
    root_path = "/home/Dataset/DCASE2020_Task2_dataset/dev_data" # dataset directory in server
    
    avg_AUC = 0.
    avg_pAUC = 0.
    
    for i in range(len(name_list)):
        test_ds = test_dataset(root_path, name_list[i], name_list)
        test_dataloader = DataLoader(test_ds, batch_size=1)
        
        AUC, PAUC = evaluator(net, test_dataloader, criterion, device)
        avg_AUC += AUC 
        avg_pAUC += PAUC 
        print(f"{name_list[i]} - AUC: {AUC:.5f}, pAUC: {PAUC:.5f}")
    
    avg_AUC = avg_AUC / len(name_list)
    avg_pAUC = avg_pAUC / len(name_list)
    
    print(f"Average AUC: {avg_AUC:.5f},  Average pAUC: {avg_pAUC:.5f}")
        
    
if __name__ == '__main__':
    torch.set_num_threads(2)
    main("TASTgramMFN", "noisy_arcmix")