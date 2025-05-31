## Noisy-ArcMix / TASTgram-MFN (Pytorch Implementation)
By Soonhyeon Choi (csh5956@kaist.ac.kr) at 
Korea Advanced Institute of Science and Technology (KAIST)

## Introduction
Noisy-ArcMix significantly improves the compactness of intra-class distribution through the training with virtually synthesized samples near the normal data distribution. More importantly, we observed that the mingling effect between normal and anomalous samples can be reduced further by Noisy-ArcMix, which gains generalization ability through the use of inconsistent angular margins for the corrupted label prediction. In addition to Noisy-ArcMix, we introduce a new input feature, temporally attended log-Mel spectrogram (TAgram), derived from a temporal attention block. TAgram includes the temporal attention weights broadcasted to spectrogram features, which helps a model to focus on the important temporal regions for capturing crucial features.

<br/>
This repository contains the implementation used in our paper https://ieeexplore.ieee.org/abstract/document/10447764.

## TASTgram Architecture

<p align="center">
  <img src="./TASTgramMFN.png" alt="TAST" width="70%" height="70%">
</p>
Input architecture of TASTgramNet. The temporal feature (Tgram) is concatenated with the log-Mel spectrogram (Sgram) and the temporally attended feature (TAgram) from the temporal attention block.

## Noisy-ArcMix

<p align="center">
  <img src="./Angle_distribution.png" alt="angle_distribution">
</p>
Distribution of angles between feature embeddings and corresponding learned class centers, for the models trained by (a) Cross-Entropy, (b) ArcFace, (c) ArcMix, and (d) Noisy-ArcMix. The results are derived from all machine types in the test data of DCASE 2020 Challenge Task 2 development dataset.

## Datasets
[DCASE2020 Task2](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) Dataset: 
+ [development dataset](https://zenodo.org/record/3678171)
+ [additional training dataset](https://zenodo.org/record/3727685)
+ [Evaluation dataset](https://zenodo.org/record/3841772)


## Organization of the files

```shell
├── check_points/
├── datasets/
    ├── fan/
        ├── train/
        ├── test/
    ├── pump/
        ├── train/
        ├── test/
    ├── slider/
        ├── train/
        ├── test/
    ├── ToyCar/
        ├── train/
        ├── test/
    ├── ToyConveyor/
        ├── train/
        ├── test/
    ├── valve/
        ├── train/
        ├── test/
├── model/
├── Dockerfile
├── README.md
├── config.yaml
├── LICENSE
├── dataloader.py
├── eval.py
├── losses.py
├── train.py
├── trainer.py
├── utils.py
├── requirements.txt
```
## Model weights / Environments
Our trained model weights file  and requirements.txt file can be accessed at https://drive.google.com/drive/folders/1tuUS-MKcAy-jFDVVdD5rpy-NU-3Pk46a?hl=ko.

## Install
```
pip install -r requirements.txt
```

## Training
Check the `config.yaml` file to select a training mode from ['arcface', 'arcmix', 'noisy_arcmix']. Default is noisy-arcmix. 
<br/>
Use `python train.py` to train a model. 
```
python train.py
```

## Evaluating
Use `python eval.py` to evaluate the trained model.
```
python eval.py
```

## Experimental Results
 | machine Type | AUC(%) | pAUC(%) | mAUC(%) |
 | --------     | :-----:| :----:  | :----:  |
 | Fan          | 98.32  | 95.34   | 92.67   |
 | Pump         | 95.44  | 85.99   | 91.17   |
 | Slider       | 99.53  | 97.50   | 97.96   |
 | Valve        | 99.95  | 99.74   | 99.89   |
 | ToyCar       | 96.76  | 90.11   | 88.81   |
 | ToyConveyor  | 77.90  | 67.15   | 68.18   |
 | __Average__      | __94.65__  | __89.31__   | __89.78__   |




## Citation
If you use this method or this code in your paper, then please cite it:
```
@inproceedings{choi2024noisy,
  title={Noisy-Arcmix: Additive Noisy Angular Margin Loss Combined With Mixup For Anomalous Sound Detection},
  author={Choi, Soonhyeon and Choi, Jung-Woo},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={516--520},
  year={2024},
  organization={IEEE}
}
```

# Setup for DDP on Only cuda:6 and cuda:7
```
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nnodes=1 --nproc_per_node=2 train_ddp.py
```