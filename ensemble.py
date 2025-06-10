# ensemble.py

import os
import argparse
import logging
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from losses import ASDLoss
from dataloader import test_dataset
# import your nets here
from model.net import (
    TASTgramMFN, TASTgramMFN_FPH, SCLTFSTgramMFN,
    TASTWgramMFN, TASTWgramMFN_FPH,
    TAST_SpecNetMFN, TAST_SpecNetMFN_archi2, TAST_SpecNetMFN_combined
)


NET_FACTORY = {
    'TASTgramMFN': TASTgramMFN,
    'TASTgramMFN_FPH': TASTgramMFN_FPH,
    'SCLTFSTgramMFN': SCLTFSTgramMFN,
    'TASTWgramMFN': TASTWgramMFN,
    'TASTWgramMFN_FPH': TASTWgramMFN_FPH,
    'TAST_SpecNetMFN': TAST_SpecNetMFN,
    'TAST_SpecNetMFN_archi2': TAST_SpecNetMFN_archi2,
    'TAST_SpecNetMFN_combined': TAST_SpecNetMFN_combined,
}


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble two or more DCASE models")
    p.add_argument('--models', nargs='+', required=True,
                   help="List of model specs: net_name:mode:loss (e.g. TAST_SpecNetMFN:noisy_arcmix:cross_entropy)")
    p.add_argument('--device', type=str, default='cuda:6',
                   help="Torch device")
    p.add_argument('--batch-size', type=int, default=1,
                   help="Batch size for DataLoader")
    p.add_argument('--alpha-step', type=float, default=0.05,
                   help="Step size for alpha grid search [0,1]")
    p.add_argument('--output-dir', type=str, default='ensemble_out',
                   help="Directory to save plots, CSVs, logs")
    return p.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'ensemble.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def load_cfg_and_model(spec, device):
    net_name, mode, loss = spec.split(':')
    cfg_path = f'./check_points/{net_name}/{mode}/{loss}/config.yaml'
    model_path = f'./check_points/{net_name}/{mode}/{loss}/model.pth'
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    NetClass = NET_FACTORY[net_name]
    net = NetClass(num_classes=cfg['num_classes'], m=cfg['m'], mode=cfg['mode']).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    logging.info(f"Loaded {net_name} ({mode}/{loss})")
    return net_name, net, cfg


def evaluate_model(net_name, net, loader, criterion, device):
    """Returns numpy arrays: y_true, y_scores"""
    all_true = []
    all_scores = []
    with torch.no_grad():
        for x_w, x_m, labels, an_labels in tqdm(loader, desc=f"Eval {net_name}", leave=False):
            x_w, x_m, labels = x_w.to(device), x_m.to(device), labels.to(device)
            an_labels = an_labels.cpu().numpy()
            if net_name.endswith('combined'):
                type_labels = labels // 7
                type_labels[labels == 34] = 5
                logits, _, _ = net(x_w, x_m, labels, type_labels, train=False)
            else:
                logits, _ = net(x_w, x_m, labels, train=False)
            scores = criterion(logits, labels).cpu().numpy()
            all_true.append(an_labels)
            all_scores.append(scores)
    return np.concatenate(all_true), np.concatenate(all_scores)


def find_best_alpha(y_true, preds, step):
    alphas = np.arange(0, 1 + step/2, step)
    best_a, best_auc = 0, 0
    for a in alphas:
        ens = preds[0] * a + preds[1] * (1 - a)
        auc = metrics.roc_auc_score(y_true, ens)
        if auc > best_auc:
            best_a, best_auc = a, auc
    return best_a, best_auc


def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1], [0,1], '--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    return auc

def compute_auc(y_t, y_s):
    """Return (AUC, pAUC@0.1)"""
    return (metrics.roc_auc_score(y_t, y_s),
            metrics.roc_auc_score(y_t, y_s, max_fpr=0.1))

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # Prepare DataLoaders
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    data_root = "/home/Dataset/DCASE2020_Task2_dataset/dev_data" # dataset directory in server
    
    criterion = ASDLoss(reduction=False).to(device)
    
    m1, net1, _ = load_cfg_and_model(args.models[0], device)
    m2, net2, _ = load_cfg_and_model(args.models[1], device)

    results = {}

    for machine in name_list:
        # build loader for this machine
        ds = test_dataset(data_root, machine, name_list)
        loader = DataLoader(ds, batch_size=args.batch_size)

        # evaluate each model
        y_true_1, y_scores_1 = evaluate_model(m1, net1, loader, criterion, device)
        y_true_2, y_scores_2 = evaluate_model(m2, net2, loader, criterion, device)

        # normalize second model
        y_scores_2 = MinMaxScaler().fit_transform(y_scores_2.reshape(-1, 1)).ravel()

        # grid-search alpha on this machine
        best_a, best_auc = 0, 0
        for a in np.arange(0, 1+args.alpha_step/2, args.alpha_step):
            auc, _ = compute_auc(y_true_1, a*y_scores_1 + (1-a)*y_scores_2)
            if auc > best_auc:
                best_a, best_auc = a, auc
        ens_scores = best_a*y_scores_1 + (1-best_a)*y_scores_2

        # machine-wide AUC/pAUC
        mac_auc, mac_pauc = compute_auc(y_true_1, ens_scores)

        results[machine] = {
            'alpha':      best_a,
            'machine_AUC': mac_auc,
            'machine_pAUC': mac_pauc,
        }

    # print a nice summary
    print(f"{'Machine':<12}  α    AUC    pAUC   avgID_AUC  avgID_pAUC")
    logging.info("Finished per-machine ensembling:")
    all_mac_auc, all_mac_pauc = [], []
    for m, info in results.items():
        print(f"{m:<12}  {info['alpha']:.2f}  "
              f"{info['machine_AUC']:.4f}  {info['machine_pAUC']:.4f}  ")
        all_mac_auc.append(info['machine_AUC'])
        all_mac_pauc.append(info['machine_pAUC'])
        logging.info(f"{m}: {info}")

    # finally overall average
    print(f"\nOverall average AUC:  {np.mean(all_mac_auc):.4f}")
    print(f"Overall average pAUC: {np.mean(all_mac_pauc):.4f}")


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
