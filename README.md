## (WORK IN PROGRESS)

# Multi-Modal Action Recognition 

This repository contains PyTorch code for the analysis of different modalities for the Kinetics dataset


**Models:** I3D, 3D-ResNet, 3D-DenseNet, 3D-ResNeXt  
**Datasets:** Kinetics, PHAV

## Installation

Clone and install:

```sh
git clone https://github.com/MKowal2/multimodal_action.git
cd PyTorchConv3D
pip install -r requirements.txt
python setup.py install
```

## Requirements

- Python 3.5+
- Numpy (developed with 1.15.0)
- PyTorch >= 1.0.0
- PIL (optional)

## Examples

Training ResNet-50 from scratch on Kinetics, Flow Modality:

```
python train.py --dataset=kinetics --multi_modal --Flow --model=resnet --video_path=/home/Datasets/KINETICS/kinetics400 --annotation_path=/home/Datasets/KINETICS/kinetics400.json --model_depth=50 --spatial_size=112 --sample_duration=32 --optimizer=SGD --learning_rate=0.01
```

## References

# Code
- This code was used for models and general folder structure: https://github.com/tomrunia/PyTorchConv3D

# Papers
- Carreira and Zisserman - "[Quo Vadis,
Action Recognition?](https://arxiv.org/abs/1705.07750)" (CVPR, 2017)
- de Souza _et al._ - "[Procedural Generation of Videos to Train Deep Action Recognition Networks
](https://arxiv.org/abs/1612.00881)" (CVPR, 2017)
- Hara _et al._ - "[Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/abs/1711.09577)" (CVPR, 2018)
