# BadMIM

This repository contains code for BadMIM. 

The directory tree is shown below:

BadMIM/  
│  
├─ AdversarialTriggerAugmentation (Phase 1: augment trigger features)  
│  
└─ ReconstructionHijacking (Phase 2: backdoors injection)  

## 📝 Experimental setup

1. Testing environment: Python==3.9.0, torch==2.5.0, timm==0.4.12, torchvision==0.20.0, numpy==1.26.4, tqdm==4.65.2

2. Download dataset (all datasets (CIFAR10, CIFAR100, STL10, Caltech101, Caltech256, Imagenette2) can be automatically downloaded by torchvision.) and save them in ''./data/'' folder.

3. Find trigger images from Internet or use triggers provided by us in ''./data/triggers/'' in AdversarialTriggerAugmentation and ''./trigger_pics/'' in ReconstructionHijacking.

4. Download clean pre-trained encoder from Internet (MAE: https://github.com/facebookresearch/mae) and save them in ''./pretrained_weights/''.

5. Collect target-class images from Internet and combine with Caltech256 to construct surrogate datasets (the target-class images are by default saved as the 257th class in Caltech256.).

## 🔍 Usage

### Phase 1: Adversarial Trigger Augmentation

The scripts in this phase is to augment triggers by calculating a universal perturbation. 

  1. Train a surrogate model by using ```bash surrogate_model_train.sh```. This script is train a surrogate model on the surrogate dataset with the clean pre-trained encoder.
  2. Train a augmentation perturbation with the surrogate model by using ```bash trigger_generation.sh```.

