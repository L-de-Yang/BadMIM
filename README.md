# BadMIM
This is the official implementation of BadMIM.

## Overview
This repository contains two main components:

1. **Adversarial Trigger Augmentation (Phase 1: ATA)**: Adversarially augment triggers using auxiliary models and datasets.
2. **Reconstruction Hijacking (Phase 2: RH)**: Inject backdoors into MIM encoders by hijacking the mask-reconstruction pretraining paradigm

```bash
BadMIM/
├── AdversarialTriggerAugmentation/
└── ReconstructionHijacking/
```

> **Note**: This repository currently supports backdooring MAE models using CIFAR10 as the downstream task.

## Datasets Preparation
According to the paper, BadMIM uses three types of datasets: auxiliary dataset, shadow dataset, and downstream dataset. We provide them via an anonymous link: [datasets](https://figshare.com/s/e7e7e89d8565bf030a1d). Below is a brief introduction:

| Dataset | Content | Description | Location |
|---------|---------|-------------|---------|
| **Auxiliary dataset** | Caltech256 + web-sourced target class images | Used to train auxiliary models and augment triggers | `./BadMIM/AdversarialTriggerAugmentation/data/` |
| **Shadow dataset** | Imagenette2 | A small subset of ImageNet used to inject backdoors into MIM encoders | `./BadMIM/ReconstructionHijacking/data/` |
| **Downstream dataset** | CIFAR10 | Downstream task for fine-tuning MIM encoders | `./BadMIM/ReconstructionHijacking/data/` |
| **Triggers** | Origonal and augmented triggers | Original and augmented triggers for each target class | Put original triggers in `./BadMIM/AdversarialTriggerAugmentation/data/triggers/` to augment from scratch; put augmented triggers in `./BadMIM/ReconstructionHijacking/triggers/` for direct use |

## Attack Pipeline
Detailed workflow for executing the BadMIM attack:

### Step 1: Set Up Testing Environment
Create a testing environment using conda or pip with the provided `./BadMIM/requirements.txt`.

### Step 2: Download Open-Sourced MIM Encoder
This repository provides code to attack MAE encoders. Download the official open-sourced MAE encoder from [https://github.com/facebookresearch/mae](https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth) and save it in both:

- `./BadMIM/AdversarialTriggerAugmentation/pretrained_weights/`
- `./BadMIM/ReconstructionHijacking/pretrained_weights/`

### Step 3: Adversarial Trigger Augmentation
#### Step 3.1: Train Auxiliary Model
Train auxiliary models on auxiliary datasets for each target class. For example, to train an auxiliary model for the "airplane" class:
```
python auxiliary_model_trainer_MAE.py \
    --auxiliary_dataset /path/to/auxiliary/dataset/for/airplane/ \
    --nb_classes 257 \
    --output_dir /path/to/output/dir/
```

> **Note**: When training an auxiliary model for the "dog" class, set `--nb_classes` to `256 since Caltech256 already contains a dog class.

#### Step 3.2: Augment Triggers
Augment triggers using the trained auxiliary models:
```
python TriggerGenerator_MAE.py \
    --auxiliary_dataset /path/to/auxiliary/dataset/for/airplane/ \
    --output_dir /path/to/output/dir/ \
    --auxiliary_model /path/to/auxiliary/model/for/airplane/ \
    --trigger_path /path/to/airplane/original/trigger/ \
    --pert_pic_save_path /path/to/save/augmented/trigger/ \
    --target_label 256
```

Set `--target_label` to `256` because web-sourced target class images are stored as the 257th class in the auxiliary dataset by default.

> **Note**: For the "dog" class, set `--target_label` to `55` and `--nb_classes` to `256`.

For quick testing, we provide pre-augmented triggers for ten classes at [datasets](https://figshare.com/s/e7e7e89d8565bf030a1d), which can be directly used in Step 4.

### Step 4: Reconstruction Hijacking
#### Step 4.1: Backdoor Injection
Train a backdoored MAE encoder using the augmented triggers from Step 3:
```
python badmim.py \
    --reference_path /path/to/augmented/trigger/ \
    --output_dir /path/to/save/dir/ \
    --total_epochs 400 \
    --alpha 0.2 \
    --pratio 0.5
```

#### Step 4.2: Downstream Task Adaptation
Fine-tune the backdoored MAE encoder on the downstream dataset:
```
python finetune_MAEViT.py \
    --finetune /path/to/backdoored/encoder/ \
    --output_dir /path/to/save/dir/ \
    --ds cifar10
```

### Step 5: Evaluation
Evaluate the fine-tuned downstream classifier:
```
python evaluate.py \
    --model_path /path/to/finetuned/model/ \
    --reference_file /path/to/augmented/trigger \
    --test_data_dir /path/to/downstream/test/dataset/ \
    --nb_class 10 \
    --target_class 0
```

- `--target_class`: Index of the target class in the downstream dataset (e.g., "airplane" is class 0 in CIFAR10)
- `--nb_class`: Number of classes in the downstream task (e.g., 10 for CIFAR10)

This script will output both BA (Benign Accuracy) and ASR-B (Backdoor Attack Success Rate).
