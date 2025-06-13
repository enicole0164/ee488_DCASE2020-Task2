## TASTgram-SpecNet-MFN (Pytorch Implementation)
By Jaeryeong Kim (jaenic@kaist.ac.kr) at 
Korea Advanced Institute of Science and Technology (KAIST)

## Introduction
Upon the implementation of NoisyArcmix paper https://ieeexplore.ieee.org/abstract/document/10447764.
We have implemented few more features in training.
- SpecNet
- Normalization
- Supervised Contrastive Loss

## TASTgram-SpecNet Architecture

Input architecture of TASTgram-SpecNet.
The temporal feature (Tgram) is concatenated with the log-Mel spectrogram (Sgram), the temporally attended feature (TAgram) from the temporal attention block, and the spectral feature (SpecNet).


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

## Training
Check the `config.yaml` file to select
- A training mode from ['arcface', 'arcmix', 'noisy_arcmix'].
- A network from ['TASTgramMFN', 'TAST_SpecNetMFN', 'TAST_SpecNetMFN_nrm', 'TAST_SpecNetMFN_nrm2']
- A training loss from ['cross_entropy', 'cross_entropy_supcon']

<br/>
Use `python train.py` to train a model. 
```
python train.py
```

## Training Supervised Contrastive Loss with two GPUs using DDP.
Use `python train_ddp.py' to train a model.
```
python train_ddp.py
```

## Evaluating
Use `python eval.py` to evaluate the trained model.
Specify the model as the argument of `main()` in the code.
```
python eval.py
```

## Experimental Results
- net: TAST_SpecNetMFN_nrm2
- mode: noisy_arcmix
- loss: cross_entropy_supcon

 | machine Type | AUC(%) |
 | --------     | :-----:|
 | Fan          | 97.83  |
 | Pump         | 97.01  |
 | Slider       | 99.93  |
 | Valve        | 97.02  |
 | ToyCar       | 83.67  |
 | ToyConveyor  | 97.05  |
 | __Average__      | __95.42__  |

# Setup for DDP on Specifying GPU want to use
```
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nnodes=1 --nproc_per_node=2 train_ddp.py
```