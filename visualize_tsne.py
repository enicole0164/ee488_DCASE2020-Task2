import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from model.net import TASTgramMFN_FPH 
from dataloader import test_dataset
import yaml
import os
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    features_before, features_after, labels_all, AN_N_labels_all = [], [], [], []

    for x_wav, x_mel, label, AN_N_labels in dataloader:
        x_wav, x_mel, label = x_wav.to(device), x_mel.to(device), label.to(device)

        # feature extraction
        x_t = model.get_tgram(x_wav).unsqueeze(1)
        x_mel_att = model.temporal_attention(x_mel).unsqueeze(1)
        x = torch.cat((x_t, x_mel, x_mel_att), dim=1)

        _, feat_before = model.mobilefacenet(x)         # before FPH
        feat_after = model.head(feat_before)            # after FPH

        features_before.append(feat_before.cpu().numpy())
        features_after.append(feat_after.cpu().numpy())
        labels_all.append(label.cpu().numpy())
        AN_N_labels_all.append(AN_N_labels.cpu().numpy())

    return np.concatenate(features_before), np.concatenate(features_after), np.concatenate(labels_all), np.concatenate(AN_N_labels_all)


def run_tsne(features):
    print("Running TSNE on features of shape:", features.shape)
    features = StandardScaler().fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(features)


def plot_tsne(features_2d, labels, title, filename):
    plt.figure(figsize=(6, 5))
    for cls in np.unique(labels):
        idx = labels == cls
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f'Class {cls}', alpha=0.6)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved t-SNE plot to {filename}")
    plt.close()

def plot_tsne_anomaly(features_2d, labels, AN_N_labels, title, filename):
    plt.figure(figsize=(7, 6))
    classes = np.unique(labels)

    for cls in classes:
        for is_anomaly in [0, 1]:
            idx = (labels == cls) & (AN_N_labels == is_anomaly)
            if np.sum(idx) == 0:
                continue

            marker = 'o' if is_anomaly == 0 else 'x'
            label_name = f'Class {cls} - {"Normal" if is_anomaly == 0 else "Anomaly"}'
            plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                        label=label_name, 
                        alpha=0.6, 
                        marker=marker, 
                        edgecolors='k' if is_anomaly else 'none')

    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved t-SNE plot to {filename}")
    plt.close()

def evaluate_clustering(X, labels, name="", save_path=None):
    print(f"\n[Clustering Evaluation: {name}]")
    
    sil_score = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    intra_dists = []
    inter_dists = []
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        cluster_pts = X[labels == lbl]
        if len(cluster_pts) > 1:
            dists = pairwise_distances(cluster_pts)
            intra_dists.append(np.mean(dists))

    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            d1 = X[labels == unique_labels[i]]
            d2 = X[labels == unique_labels[j]]
            inter_dists.append(np.mean(pairwise_distances(d1, d2)))
    
    avg_intra = np.mean(intra_dists)
    avg_inter = np.mean(inter_dists)

    report = (
        f"[Clustering Evaluation: {name}]\n"
        f"Silhouette Score: {sil_score:.4f}\n"
        f"Davies-Bouldin Index: {db_score:.4f}\n"
        f"Avg. Intra-cluster Distance: {avg_intra:.4f}\n"
        f"Avg. Inter-cluster Distance: {avg_inter:.4f}\n"
    )
    print(report)

    if save_path:
        with open(save_path, 'a') as f:
            f.write(report)



def main():
    # load config & checkpoint Fill in yours
    config_path = './check_points/TASTgramMFN_FPH/noisy_arcmix/cross_entropy_supcon/config.yaml'
    model_path = './check_points/TASTgramMFN_FPH/noisy_arcmix/cross_entropy_supcon/model.pth'

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(f"cuda:{cfg['gpu_num']}" if torch.cuda.is_available() else "cpu")
    
    model = TASTgramMFN_FPH(num_classes=cfg['num_classes'],
                            mode=cfg['mode'],
                            m=cfg['m'],
                            cfg=cfg).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # load dataset
    root_path = "/home/Dataset/DCASE2020_Task2_dataset/dev_data"
    name_list = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']

    for i in range(len(name_list)):
        print(f"\nProcessing machine type: {name_list[i]}")
        test_ds = test_dataset(root_path, name_list[i], name_list)  # single section
        dataloader = DataLoader(test_ds, batch_size=1, num_workers=4, shuffle=False)

        # Feature extraction
        feat_before, feat_after, labels, AN_N_labels = extract_features(model, dataloader, device)

        # Create output dir
        save_dir = f"./debugging/{name_list[i]}"
        os.makedirs(save_dir, exist_ok=True)

        # t-SNE
        tsne_before = run_tsne(feat_before)
        tsne_after = run_tsne(feat_after)

        # plot and save
        plot_tsne(tsne_before, labels, f"{name_list[i]} - t-SNE Before FPH", f"{save_dir}/tsne_before.png")
        plot_tsne(tsne_after, labels, f"{name_list[i]} - t-SNE After FPH", f"{save_dir}/tsne_after.png")

        # plot and save
        plot_tsne_anomaly(tsne_before, labels, AN_N_labels, f"{name_list[i]} - t-SNE Before FPH", f"{save_dir}/tsne_before_anomaly.png")
        plot_tsne_anomaly(tsne_after, labels, AN_N_labels, f"{name_list[i]} - t-SNE After FPH", f"{save_dir}/tsne_after_anomaly.png")

        # Save clustering metrics
        evaluate_clustering(feat_before, labels, name="Before FPH", save_path=f"{save_dir}/clustering_performance.txt")
        evaluate_clustering(feat_after, labels, name="After FPH", save_path=f"{save_dir}/clustering_performance.txt")

if __name__ == "__main__":
    main()