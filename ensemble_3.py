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
from collections import defaultdict

from losses import ASDLoss
from dataloader import test_dataset
# import your nets here
from model.net import (
    TASTgramMFN, TASTgramMFN_FPH, SCLTFSTgramMFN,
    TASTWgramMFN, TASTWgramMFN_FPH,
    TAST_SpecNetMFN, TAST_SpecNetMFN_archi2, TAST_SpecNetMFN_combined,
    TAST_SpecNetMFN_nrm, TAST_SpecNetMFN_nrm2
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
    'TAST_SpecNetMFN_nrm': TAST_SpecNetMFN_nrm,
    'TAST_SpecNetMFN_nrm2': TAST_SpecNetMFN_nrm2,
}

def parse_args():
    p = argparse.ArgumentParser(description="Ensemble three DCASE models with per-ID AUC optimization")
    p.add_argument('--models', nargs=3, required=True,
                   help="List of three model specs: net_name:mode:loss (e.g. TAST_SpecNetMFN:noisy_arcmix:cross_entropy)")
    p.add_argument('--device', type=str, default='cuda:0',
                   help="Torch device")
    p.add_argument('--batch-size', type=int, default=1,
                   help="Batch size for DataLoader")
    p.add_argument('--alpha-step', type=float, default=0.05,
                   help="Step size for alpha grid search [0,1]")
    p.add_argument('--output-dir', type=str, default='ensemble3_out',
                   help="Directory to save plots, CSVs, logs")
    return p.parse_args()


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, 'ensemble3.log'),
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
    """Returns ids, y_true, y_scores as numpy arrays"""
    all_ids = []
    all_true = []
    all_scores = []
    with torch.no_grad():
        for x_w, x_m, labels, an_labels in tqdm(loader, desc=f"Eval {net_name}", leave=False):
            x_w, x_m = x_w.to(device), x_m.to(device)
            ids_np = labels.cpu().numpy()
            if net_name.endswith('combined'):
                type_labels = labels // 7
                type_labels[labels == 34] = 5
                logits, _, _ = net(x_w, x_m, labels.to(device), type_labels.to(device), train=False)
            else:
                logits, _ = net(x_w, x_m, labels.to(device), train=False)
            scores = criterion(logits, labels.to(device)).cpu().numpy()
            all_ids.append(ids_np)
            all_true.append(an_labels.cpu().numpy())
            all_scores.append(scores)
    return np.concatenate(all_ids), np.concatenate(all_true), np.concatenate(all_scores)


def compute_auc(y_t, y_s):
    """Return (AUC, pAUC@0.1)"""
    return (metrics.roc_auc_score(y_t, y_s),
            metrics.roc_auc_score(y_t, y_s, max_fpr=0.1))


def compute_avg_id_auc(ids, y_true, y_scores):
    """Compute average per-ID AUC and pAUC over all IDs"""
    trues_by_id = defaultdict(list)
    scores_by_id = defaultdict(list)
    for _id, t, s in zip(ids, y_true, y_scores):
        trues_by_id[_id].append(t)
        scores_by_id[_id].append(s)
    aucs, paucs = [], []
    for _id in trues_by_id:
        ys = trues_by_id[_id]
        if len(set(ys)) < 2:
            continue
        preds = scores_by_id[_id]
        aucs.append(metrics.roc_auc_score(ys, preds))
        paucs.append(metrics.roc_auc_score(ys, preds, max_fpr=0.1))
    avg_auc = float(np.mean(aucs)) if aucs else 0.0
    avg_pauc = float(np.mean(paucs)) if paucs else 0.0
    return avg_auc, avg_pauc


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    data_root = "/home/Dataset/DCASE2020_Task2_dataset/dev_data"
    criterion = ASDLoss(reduction=False).to(device)

    models = [load_cfg_and_model(spec, device) for spec in args.models]
    results = {}

    for machine in name_list:
        ds = test_dataset(data_root, machine, name_list)
        loader = DataLoader(ds, batch_size=args.batch_size)

        ids_list, true_list, scores_list = None, None, []
        for net_name, net, _ in models:
            ids, y_true, y_s = evaluate_model(net_name, net, loader, criterion, device)
            if true_list is None:
                ids_list, true_list = ids, y_true
            y_s = MinMaxScaler().fit_transform(y_s.reshape(-1, 1)).ravel()
            scores_list.append(y_s)
        assert np.array_equal(ids_list, ids_list)
        ids, y_true = ids_list, true_list

        # grid search on (a1,a2) with a3 = 1-a1-a2
        best = {'a1':0,'a2':0,'a3':0,'avgID_AUC':0,'avgID_pAUC':0}
        alphas = np.arange(0, 1+args.alpha_step/2, args.alpha_step)
        for a1 in alphas:
            for a2 in alphas:
                a3 = 1 - a1 - a2
                if a3 < 0: continue
                ens = a1*scores_list[0] + a2*scores_list[1] + a3*scores_list[2]
                avg_auc, avg_pauc = compute_avg_id_auc(ids, y_true, ens)
                if avg_auc > best['avgID_AUC']:
                    best.update({'a1':a1,'a2':a2,'a3':a3,'avgID_AUC':avg_auc,'avgID_pAUC':avg_pauc})

        ens_scores = best['a1']*scores_list[0] + best['a2']*scores_list[1] + best['a3']*scores_list[2]
        mac_auc, mac_pauc = compute_avg_id_auc(ids, y_true, ens_scores)
        results[machine] = {**best, 'machine_AUC':mac_auc,'machine_pAUC':mac_pauc}

    header = f"{'Machine':<12}  α1/α2/α3    AUC     pAUC   avgID_AUC  avgID_pAUC"
    print(header)
    logging.info("Finished per-machine ensembling:")

    sums = {'mac_auc':[],'mac_pauc':[],'avg_auc':[],'avg_pauc':[]}
    for m, info in results.items():
        print(f"{m:<12}  {info['a1']:.2f}/{info['a2']:.2f}/{info['a3']:.2f}  "
              f"{info['machine_AUC']:.4f}  {info['machine_pAUC']:.4f}  "
              f"{info['avgID_AUC']:.4f}    {info['avgID_pAUC']:.4f}")
        sums['mac_auc'].append(info['machine_AUC'])
        sums['mac_pauc'].append(info['machine_pAUC'])
        sums['avg_auc'].append(info['avgID_AUC'])
        sums['avg_pauc'].append(info['avgID_pAUC'])
        logging.info(f"{m}: {info}")

    print(f"Overall machine AUC:  {np.mean(sums['mac_auc']):.4f}")
    print(f"Overall machine pAUC: {np.mean(sums['mac_pauc']):.4f}")
    print(f"Overall avgID AUC:    {np.mean(sums['avg_auc']):.4f}")
    print(f"Overall avgID pAUC:   {np.mean(sums['avg_pauc']):.4f}")

if __name__ == '__main__':
    torch.set_num_threads(2)
    main()