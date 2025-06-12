import os
import torch
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict

from losses import ASDLoss
from dataloader import eval_dataset

from model.net import (
    TASTgramMFN, TASTgramMFN_FPH, SCLTFSTgramMFN,
    TASTWgramMFN, TASTWgramMFN_FPH,
    TAST_SpecNetMFN, TAST_SpecNetMFN_archi2, TAST_SpecNetMFN_combined,
    TAST_SpecNetMFN_nrm, TAST_SpecNetMFN_nrm2, TASTgramMFN_nrm
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
    'TASTgramMFN_nrm': TASTgramMFN_nrm,
}

def load_net_entry(cfg, entry, base_checkpoint_dir, device):
    """Load a single {net_name,mode,loss_name} entry."""
    net_cls = NET_FACTORY[entry['net_name']]
    net = net_cls(
        num_classes=cfg['num_classes'],
        m=cfg.get('m', 0.7),
        mode=entry['mode']
    ).to(device)

    ckpt_path = os.path.join(
        base_checkpoint_dir,
        entry['net_name'],
        entry['mode'],
        entry['loss_name'],
        'model.pth'
    )
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    return net

def evaluator_ensemble(nets, alphas, test_loader, criterion, device):
    file_paths, ens_scores = [], []

    with torch.no_grad():
        for x_wavs, x_mels, labels, paths in test_loader:
            x_wavs, x_mels = x_wavs.to(device), x_mels.to(device)
            labels          = labels.to(device)

            # collect each model's anomaly‐score vector
            per_model = []
            for net in nets:
                logits, _ = net(x_wavs, x_mels, labels, train=False)
                s = criterion(logits, labels)  # shape (batch,)
                per_model.append(s.cpu().numpy())

            stacked = np.stack(per_model, axis=0)   # shape (n_models, batch)
            ens     = np.tensordot(alphas, stacked, axes=([0],[0]))  # (batch,)

            for p, score in zip(paths, ens.tolist()):
                file_paths.append(p)
                ens_scores.append(score)

    return file_paths, ens_scores


def main(config_path):
    # load global config
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(f"cuda:{cfg['gpu_num']}" 
                          if torch.cuda.is_available() else "cpu")
    criterion = ASDLoss(reduction=False).to(device)
    base_ckpt_dir = './check_points'

    # instantiate all three nets with their own loss_name/mode
    nets = [
        load_net_entry(cfg, entry, base_ckpt_dir, device)
        for entry in cfg['ensemble_models']
    ]

    name_list = ['fan','pump','slider','ToyCar','ToyConveyor','valve']
    root_path = './eval_dataset'

    for machine in name_list:
        all_paths, all_scores = [], []
        alphas = cfg['ensemble_alphas'][machine]
        ds     = eval_dataset(root_path, machine, name_list)
        dl     = DataLoader(ds, batch_size=1)

        paths, scores = evaluator_ensemble(nets, alphas, dl, criterion, device)
        all_paths.extend(paths)
        all_scores.extend(scores)
        print(f"{machine}: {len(scores)} files scored.")

        # write out
        df = pd.DataFrame({
            f'{machine}': all_paths,
            'anomaly_score': all_scores
        })
        out_csv = os.path.join('./anomaly_score_ens_csv', f'anomaly_scores_{machine}.csv')
        df.to_csv(out_csv, index=False)
        print("Wrote ensemble scores to", out_csv)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main('./ensemble_evaluate.yaml')
   