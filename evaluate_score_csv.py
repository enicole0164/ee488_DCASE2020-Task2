import os
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from losses import ASDLoss, SupConLoss
from model.net import (
    TASTgramMFN, TAST_SpecNetMFN,
    TAST_SpecNetMFN_nrm, TAST_SpecNetMFN_nrm2
)
from dataloader import eval_dataset
import pandas as pd
import yaml
from tqdm import tqdm
import argparse

# Mapping of model names to classes
NET_FACTORY = {
    'TASTgramMFN': TASTgramMFN,
    'TAST_SpecNetMFN': TAST_SpecNetMFN,
    'TAST_SpecNetMFN_nrm': TAST_SpecNetMFN_nrm,
    'TAST_SpecNetMFN_nrm2': TAST_SpecNetMFN_nrm2,
}


def evaluator_with_paths(net, test_loader, criterion, device):
    net.eval()
    file_paths, scores = [], []

    with torch.no_grad():
        for x_wavs, x_mels, labels, paths in test_loader:
            x_wavs, x_mels = x_wavs.to(device), x_mels.to(device)
            labels = labels.to(device)
            logits, _ = net(x_wavs, x_mels, labels, train=False)
            score = criterion(logits, labels)
            for p, s in zip(paths, score.cpu().tolist()):
                file_paths.append(p)
                scores.append(s)

    return file_paths, scores


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate anomaly detection models and save per-file scores to CSV.'
    )
    parser.add_argument('--net', required=True,
                        choices=NET_FACTORY.keys(),
                        help='Name of the network to evaluate')
    parser.add_argument('--mode', default='noisy_arcmix',
                        help='Mode identifier (e.g. noisy_arcmix)')
    parser.add_argument('--loss', dest='loss_name', required=True,
                        help='Loss function name used in checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation DataLoader')
    parser.add_argument('--root', default='./eval_dataset',
                        help='Root path for evaluation dataset')
    parser.add_argument('--out_dir', default='./anomaly_score_csv',
                        help='Where to save the output CSV files')
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index')
    args = parser.parse_args()

    ckpt_dir = os.path.join('check_points', args.net, args.mode, args.loss_name)
    config_path = os.path.join(ckpt_dir, 'config.yaml')
    model_path = os.path.join(ckpt_dir, 'model.pth')

    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Sanity checks
    assert cfg['net_name'] == args.net, \
        f"Checkpoint net_name '{cfg['net_name']}' does not match --net '{args.net}'"
    assert cfg['mode'] == args.mode, \
        f"Checkpoint mode '{cfg['mode']}' does not match --mode '{args.mode}'"

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # Instantiate model
    NetClass = NET_FACTORY[args.net]
    net = NetClass(num_classes=cfg['num_classes'], m=cfg.get('m', 0.7), mode=cfg['mode']).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    criterion = ASDLoss(reduction=False).to(device)
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
    os.makedirs(args.out_dir, exist_ok=True)

    for machine_type in name_list:
        paths_all, scores_all = [], []
        dataset = eval_dataset(args.root, machine_type, name_list)
        loader = DataLoader(dataset, batch_size=args.batch_size)
        paths, scores = evaluator_with_paths(net, loader, criterion, device)
        paths_all.extend(paths)
        scores_all.extend(scores)

        df = pd.DataFrame({
            f'{machine_type}': paths_all,
            'anomaly_score': scores_all
        })
        csv_path = os.path.join(args.out_dir, f'anomaly_scores_{machine_type}.csv')
        df.to_csv(csv_path, index=False)
        print(f'Wrote anomaly scores to {csv_path}')


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
