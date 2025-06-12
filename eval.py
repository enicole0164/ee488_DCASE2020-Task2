import os
import torch 
from torch.utils.data import DataLoader
from sklearn import metrics
from losses import ASDLoss, SupConLoss
from model.net import TASTgramMFN, TASTgramMFN_FPH, SCLTFSTgramMFN, TASTWgramMFN, TASTWgramMFN_FPH, TAST_SpecNetMFN, TAST_SpecNetMFN_archi2, TAST_SpecNetMFN_combined, TAST_SpecNetMFN_nrm, TAST_SpecNetMFN_nrm_combined, TASTgramMFN_nrm, TAST_SpecNetMFN_nrm2
from dataloader import test_dataset  
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np

from collections import defaultdict
import numpy as np
from sklearn import metrics

def evaluator(net_name, net, test_loader, criterion, device):
    net.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_wavs, x_mels, labels, AN_N_labels in test_loader:
            x_wavs, x_mels, labels, AN_N_labels = x_wavs.to(device), x_mels.to(device), labels.to(device), AN_N_labels.to(device)
            
            if net_name == 'TAST_SpecNetMFN_combined' or net_name == 'TAST_SpecNetMFN_nrm_combined':
                type_labels = labels // 7
                type_labels[labels == 34] = 5 
                id_logits, _, _ = net(x_wavs, x_mels, labels, type_labels, train=False)
                logits = id_logits
                # logits = net.getcosine(x_wavs, x_mels)
                score = criterion(logits, labels)

            else:
                logits, _ = net(x_wavs, x_mels, labels, train=False)
                score = criterion(logits, labels)

            y_pred.extend(score.tolist())
            y_true.extend(AN_N_labels.tolist())
    
    auc = metrics.roc_auc_score(y_true, y_pred)
    pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    return auc, pauc                


def evaluator2(net_name, net, test_loader, criterion, device):
    net.eval()

    # dicts to collect per-ID true labels and predictions
    trues_by_id = defaultdict(list)
    preds_by_id  = defaultdict(list)

    with torch.no_grad():
        for x_wavs, x_mels, labels, AN_N_labels in test_loader:
            x_wavs = x_wavs.to(device)
            x_mels = x_mels.to(device)
            labels = labels.to(device)
            AN_N_labels = AN_N_labels.to(device)

            # forward pass
            logits, _ = net(x_wavs, x_mels, labels, train=False)

            # anomaly score per sample
            score = criterion(logits, labels)  # shape: (batch,)
            
            # determine ID: if labels is one-hot, take argmax; else it's already an int
            if labels.ndim > 1:
                # one-hot → integer IDs
                ids = labels.argmax(dim=1).cpu().tolist()
            else:
                ids = labels.cpu().tolist()

            # gather true/pred for each ID
            for _id, _score, _true in zip(ids, score.cpu().tolist(), AN_N_labels.cpu().tolist()):
                trues_by_id[_id].append(_true)
                preds_by_id[_id].append(_score)

    # compute per-ID AUCs
    aucs  = []
    paucs = []
    for _id in trues_by_id:
        y_true = trues_by_id[_id]
        y_pred = preds_by_id[_id]

        # need at least one positive and one negative to compute AUC
        if len(set(y_true)) < 2:
            continue

        auc  = metrics.roc_auc_score(y_true, y_pred)
        pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
        aucs.append(auc)
        paucs.append(pauc)

    # average across IDs
    avg_auc  = float(np.mean(aucs))  if aucs else 0.0
    avg_pauc = float(np.mean(paucs)) if paucs else 0.0

    return avg_auc, avg_pauc

        
def main(net_name, mode, loss_name):
    save_dir = f'./check_points/{net_name}/{mode}/{loss_name}'
    save_path = os.path.join(save_dir, 'model.pth')

    with open(os.path.join(save_dir, 'config.yaml'), encoding='UTF-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    print(cfg)
    assert cfg['mode'] == mode, f"cfg mode {cfg['mode']} does not match input mode {mode}"
    assert cfg['net_name'] == net_name, f"cfg net_name {cfg['net_name']} does not match input net_name {net_name}"
    
    device_num = cfg['gpu_num']
    
    device = torch.device(f'cuda:{device_num}')

    if net_name == 'TASTgramMFN':
        net = TASTgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TASTgramMFN_nrm':
        net = TASTgramMFN_nrm(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TASTgramMFN_FPH':
        net = TASTgramMFN_FPH(cfg=cfg, num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'SCLTFSTgramMFN':
        net = SCLTFSTgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TASTWgramMFN':
        net = TASTWgramMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TASTWgramMFN_FPH':
        net = TASTWgramMFN_FPH(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TAST_SpecNetMFN':
        net = TAST_SpecNetMFN(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TAST_SpecNetMFN_archi2':
        net = TAST_SpecNetMFN_archi2(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TAST_SpecNetMFN_combined':
        net = TAST_SpecNetMFN_combined(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TAST_SpecNetMFN_nrm':
        net = TAST_SpecNetMFN_nrm(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TAST_SpecNetMFN_nrm2':
        net = TAST_SpecNetMFN_nrm2(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    elif net_name == 'TAST_SpecNetMFN_nrm_combined':
        net = TAST_SpecNetMFN_nrm_combined(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    else:
        raise ValueError(f"Unknown net name: {net_name}")
    
    
    print(f"Loading model from {save_path}")
    net.load_state_dict(torch.load(save_path))
    net.eval()
    
    loss_name = cfg['loss_name']
    criterion = ASDLoss(reduction=False).to(device)
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    # root_path = './datasets'
    root_path = "/home/Dataset/DCASE2020_Task2_dataset/dev_data" # dataset directory in server
    
    avg_AUC = 0.
    avg_pAUC = 0.
    
    for i in range(len(name_list)):
        test_ds = test_dataset(root_path, name_list[i], name_list)
        test_dataloader = DataLoader(test_ds, batch_size=1)
        
        AUC, PAUC = evaluator2(net_name, net, test_dataloader, criterion, device)
        avg_AUC += AUC 
        avg_pAUC += PAUC 
        print(f"{name_list[i]} - AUC: {AUC:.5f}, pAUC: {PAUC:.5f}")
    
    avg_AUC = avg_AUC / len(name_list)
    avg_pAUC = avg_pAUC / len(name_list)
    
    print(f"Average AUC: {avg_AUC:.5f},  Average pAUC: {avg_pAUC:.5f}")
        
    
if __name__ == '__main__':
    torch.set_num_threads(2)
    main("TAST_SpecNetMFN", "noisy_arcmix", "cross_entropy")
