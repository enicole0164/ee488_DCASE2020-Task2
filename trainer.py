import torch
from model.net import TASTgramMFN, TAST_SpecNetMFN,TAST_SpecNetMFN_nrm, TAST_SpecNetMFN_nrm2
from tqdm import tqdm
from utils import get_accuracy, mixup_data, arcmix_criterion, noisy_arcmix_criterion
from losses import ASDLoss, ArcMarginProduct, SupConLoss
from torch.amp import autocast
import matplotlib.pyplot as plt
import pandas as pd 


class Trainer:
    def __init__(self, cfg, device, net, loss_name, mode, m, alpha, epochs, class_num=41, lr=1e-4):
        self.cfg = cfg
        self.device = device
        self.epochs = epochs
        self.alpha = alpha

        if net == 'TASTgramMFN':
            self.net = TASTgramMFN(num_classes=class_num, mode=mode, m=m).to(self.device)
        elif net == 'TAST_SpecNetMFN':
            self.net = TAST_SpecNetMFN(num_classes=class_num, mode=mode, m=m).to(self.device)
        elif net == 'TAST_SpecNetMFN_nrm':
            self.net = TAST_SpecNetMFN_nrm(num_classes=class_num, mode=mode, m=m).to(self.device)
        elif net == 'TAST_SpecNetMFN_nrm2':
            self.net = TAST_SpecNetMFN_nrm2(num_classes=class_num, mode=mode, m=m).to(self.device)
        else:
            raise ValueError('Net should be one of [TASTgramMFN, TAST_SpecNetMFN, TAST_SpecNetMFN_nrm, TAST_SpecNetMFN_nrm2]')
        print(f'{net} has been selected...')
        
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0.1*float(lr))

        smth = float(cfg.get('lbl_smth', 0.0))
        self.criterion = ASDLoss(smth).to(self.device)
        self.test_criterion = ASDLoss(smth, reduction=False).to(self.device)
        
        self.mode = mode
        self.loss_name = loss_name
        if loss_name not in ['cross_entropy', 'cross_entropy_supcon']:
            raise ValueError('Loss should be one of [cross_entropy, cross_entropy_supcon]')
        elif loss_name == 'cross_entropy_supcon':
            self.sc_criternion = SupConLoss().to(self.device)
        print(f'{loss_name} loss has been selected...')

        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Mode should be one of [arcface, arcmix, noisy_arcmix]')
        
        print(f'{mode} mode has been selected...')
        
    def train(self, train_loader, valid_loader, save_path):
        num_steps = len(train_loader)
        min_val_loss = 1e10

        # New: Lists to track history
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        
        for epoch in tqdm(range(self.epochs), total=self.epochs):
            self.net.train()
            sum_loss = 0.
            sum_accuracy = 0.
            
            for _, (x_wavs, x_mels, labels) in tqdm(enumerate(train_loader), total=num_steps):
                self.net.train()
                
                x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)
                
                with autocast('cuda'):
                    if self.mode == 'arcface':
                        logits, features = self.net(x_wavs, x_mels, labels)
                        ce_loss = self.criterion(logits, labels)
                    
                    elif self.mode == 'noisy_arcmix':
                        mixed_x_wavs, mixed_x_mels, y_a, y_b, lam = mixup_data(x_wavs, x_mels, labels, self.device, alpha=self.alpha)
                        logits, features = self.net(mixed_x_wavs, mixed_x_mels, labels)
                        ce_loss = noisy_arcmix_criterion(self.criterion, logits, y_a, y_b, lam)
                    
                    elif self.mode == 'arcmix':
                        mixed_x_wavs, mixed_x_mels, y_a, y_b, lam = mixup_data(x_wavs, x_mels, labels, self.device, alpha=self.alpha)
                        logits, logits_shuffled, features = self.net(mixed_x_wavs, mixed_x_mels, [y_a, y_b])
                        ce_loss = arcmix_criterion(self.criterion, logits, logits_shuffled, y_a, y_b, lam)
                
                if self.loss_name == 'cross_entropy':
                    loss = ce_loss
                elif self.loss_name == 'cross_entropy_supcon':
                    features = features.unsqueeze(1) # shape: [batch_size, 1, feature_dim]
                    sc_loss = self.sc_criternion(features, labels)
                    loss = ce_loss + sc_loss

                sum_accuracy += get_accuracy(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                sum_loss += loss.item()
            self.scheduler.step()
                
            avg_loss = sum_loss / num_steps
            avg_accuracy = sum_accuracy / num_steps

            
            val_loss, val_acc = self.valid(valid_loader)
            
            valid_loss, valid_accuracy = self.valid(valid_loader)
            print(f'EPOCH: {epoch} | Train_loss: {avg_loss:.5f} | Train_accuracy: {avg_accuracy:.5f} | Valid_loss: {valid_loss:.5f} | Valid_accuracy: {valid_accuracy:.5f}')
            # Save history
            train_losses.append(avg_loss)
            train_accuracies.append(avg_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            if min_val_loss > valid_loss:
                min_val_loss = valid_loss
                lr = self.scheduler.get_last_lr()[0]
                print("model has been saved!")
                print(f'lr: {lr:.7f} | EPOCH: {epoch} | Train_loss: {avg_loss:.5f} | Train_accuracy: {avg_accuracy:.5f} | Valid_loss: {valid_loss:.5f} | Valid_accuracy: {valid_accuracy:.5f}')
                torch.save(self.net.state_dict(), save_path)

                        # New: Plotting section after training
        
        train_accuracies = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in train_accuracies]
        val_accuracies   = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in val_accuracies]
        train_losses = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in train_losses]
        val_losses   = [x.cpu().item() if torch.is_tensor(x) else float(x) for x in val_losses]

        # Plotting the loss and accuracy curves
        print("Plotting loss and accuracy curves...")
        epochs = range(1, self.epochs + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_plot_arcmix_epoch_50.png")
        plt.show(block=True)    
        print("Training complete. Loss and accuracy curves have been plotted.")

         # make sure this is at the top of your file

        # After plt.show()
        # Save to CSV
        metrics_df = pd.DataFrame({
            'Epoch': list(range(1, self.epochs + 1)),
            'Train_Loss': train_losses,
            'Val_Loss': val_losses,
            'Train_Accuracy': train_accuracies,
            'Val_Accuracy': val_accuracies
        })

        csv_path = "training_metrics_arcmix.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"📁 Training metrics saved to {csv_path}")

         
    def valid(self, valid_loader):
        self.net.eval()
        
        num_steps = len(valid_loader)
        sum_loss = 0.
        sum_accuracy = 0.
        
        for (x_wavs, x_mels, labels) in valid_loader:
            x_wavs, x_mels, labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device)
            logits, features = self.net(x_wavs, x_mels, labels, train=False)
            sum_accuracy += get_accuracy(logits, labels)

            if self.loss_name == 'cross_entropy':
                loss = self.criterion(logits, labels)
            elif self.loss_name == 'cross_entropy_supcon':
                features = features.unsqueeze(1) # shape: [batch_size, 1, feature_dim]
                sc_loss = self.sc_criternion(features, labels)
                loss = self.criterion(logits, labels) + sc_loss
            sum_loss += loss.item()
            
        avg_loss = sum_loss / num_steps 
        avg_accuracy = sum_accuracy / num_steps 
        return avg_loss, avg_accuracy
    
    def test(self, test_loader):
        self.net.eval()
        
        y_true = []
        y_pred = []
        
        sum_accuracy = 0.
        with torch.no_grad():
            for x_wavs, x_mels, labels, AN_N_labels in test_loader:
                x_wavs, x_mels, labels, AN_N_labels = x_wavs.to(self.device), x_mels.to(self.device), labels.to(self.device), AN_N_labels.to(self.device)
                
                logits, _ = self.net(x_wavs, x_mels, labels, train=False)
                score = self.test_criterion(logits, labels)
                sum_accuracy += get_accuracy(logits, labels)
                
                y_pred.extend(score.tolist())
                y_true.extend(AN_N_labels.tolist())
        auc = metrics.roc_auc_score(y_true, y_pred)
        #pauc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
        return auc, sum_accuracy / len(test_loader)