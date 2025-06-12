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
    p = argparse.ArgumentParser(description="Ensemble two or more DCASE models")
    p.add_argument('--models', nargs=2, required=True,
                   help="Two model specs: net_name:mode:loss (e.g. TAST_SpecNetMFN:noisy_arcmix:cross_entropy)")
    p.add_argument('--device', type=str, default='cuda:0',
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
    """Returns numpy arrays: ids, y_true, y_scores"""
    all_ids = []
    all_true = []
    all_scores = []
    with torch.no_grad():
        for x_w, x_m, labels, an_labels in tqdm(loader, desc=f"Eval {net_name}", leave=False):
            x_w, x_m = x_w.to(device), x_m.to(device)
            # capture machine ID per sample
            ids_np = labels.cpu().numpy()
            # forward pass
            if net_name.endswith('combined'):
                type_labels = labels // 7
                type_labels[labels == 34] = 5
                logits, _, _ = net(x_w, x_m, labels.to(device), type_labels.to(device), train=False)
            else:
                logits, _ = net(x_w.to(device), x_m.to(device), labels.to(device), train=False)
            # anomaly scores
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

    # machine types
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    data_root = "/home/Dataset/DCASE2020_Task2_dataset/dev_data"

    criterion = ASDLoss(reduction=False).to(device)

    m1, net1, _ = load_cfg_and_model(args.models[0], device)
    m2, net2, _ = load_cfg_and_model(args.models[1], device)

    results = {}

    for machine in name_list:
        ds = test_dataset(data_root, machine, name_list)
        loader = DataLoader(ds, batch_size=args.batch_size)

        ids1, y_true1, y_scores1 = evaluate_model(m1, net1, loader, criterion, device)
        ids2, y_true2, y_scores2 = evaluate_model(m2, net2, loader, criterion, device)

        # ensure same order
        assert np.array_equal(ids1, ids2) and np.array_equal(y_true1, y_true2), \
               f"Mismatched IDs/labels for {machine}"
        ids = ids1
        y_true = y_true1

        # normalize second model's scores
        y_scores2 = MinMaxScaler().fit_transform(y_scores2.reshape(-1, 1)).ravel()

        # grid search over alpha by avg per-ID AUC
        best_a, best_avg_auc, best_avg_pauc = 0, 0, 0
        for a in np.arange(0, 1 + args.alpha_step/2, args.alpha_step):
            ens = a * y_scores1 + (1 - a) * y_scores2
            avg_auc, avg_pauc = compute_avg_id_auc(ids, y_true, ens)
            if avg_auc > best_avg_auc:
                best_a, best_avg_auc, best_avg_pauc = a, avg_auc, avg_pauc

        # final ensemble scores
        ens_scores = best_a * y_scores1 + (1 - best_a) * y_scores2
        mac_auc, mac_pauc = compute_avg_id_auc(ids, y_true, ens_scores)

        results[machine] = {
            'alpha': best_a,
            'machine_AUC': mac_auc,
            'machine_pAUC': mac_pauc,
            'avgID_AUC': best_avg_auc,
            'avgID_pAUC': best_avg_pauc
        }

    # summary
    header = f"{'Machine':<12}  α    AUC    pAUC   avgID_AUC  avgID_pAUC"
    print(header)
    logging.info("Finished per-machine ensembling:")

    all_mac_auc, all_mac_pauc, all_avg_auc, all_avg_pauc = [], [], [], []
    for m, info in results.items():
        print(f"{m:<12}  {info['alpha']:.2f}  {info['machine_AUC']:.4f}  {info['machine_pAUC']:.4f}  {info['avgID_AUC']:.4f}  {info['avgID_pAUC']:.4f}")
        all_mac_auc.append(info['machine_AUC'])
        all_mac_pauc.append(info['machine_pAUC'])
        all_avg_auc.append(info['avgID_AUC'])
        all_avg_pauc.append(info['avgID_pAUC'])
        logging.info(f"{m}: {info}")

    print(f"\nOverall machine AUC:  {np.mean(all_mac_auc):.4f}")
    print(f"Overall machine pAUC: {np.mean(all_mac_pauc):.4f}")
    print(f"Overall avgID AUC:    {np.mean(all_avg_auc):.4f}")
    print(f"Overall avgID pAUC:  {np.mean(all_avg_pauc):.4f}")

if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
