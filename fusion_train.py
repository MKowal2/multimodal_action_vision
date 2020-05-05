
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import torch.nn as nn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

from utils.utils import *
import utils.mean_values
import factory.data_factory as data_factory
from models.i3d import InceptionI3D
from config import parse_opts
import csv

from epoch_iterators import train_fusion_epoch, validation_fusion_epoch
from models.fusion_networks import fusion_uniform_layer, fusion_class_layer
import matplotlib.pyplot as plt
import random
import numpy as np

####################################################################
####################################################################
# Configuration and logging

config = parse_opts()
config = prepare_fusion_layer_output_dirs(config)
config = init_cropping_scales(config)
config = set_lr_scheduling_policy(config)

config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
config.image_std = utils.mean_values.get_std(config.norm_value)

# print and prepare config file and output directories
print_config(config)
write_config(config, os.path.join(config.fusion_layer_save_dir, 'config.json'))
from tensorboardX import SummaryWriter
#writer = SummaryWriter(log_dir=config.log_dir)

####################################################################
####################################################################
# Initialize model
device = torch.device(config.device)
#torch.backends.cudnn.enabled = False

# Returns the network instance (I3D, 3D-ResNet etc.)
# Note: this also restores the weights and optionally replaces final layer
models = []

if config.rgb:
    model_rgb = InceptionI3D(
                    num_classes=config.finetune_num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=3,
                    dropout_keep_prob=config.dropout_keep_prob
                )
    # LOAD CHECKPOINT HERE FOR EACH MODEL
    checkpoint = torch.load(config.rgb_checkpoint_path)
    model_rgb.load_state_dict(checkpoint['state_dict'])
    models.append(model_rgb)
if config.depth:
    model_depth = InceptionI3D(
                    num_classes=config.finetune_num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=1,
                    dropout_keep_prob=config.dropout_keep_prob
                )
    checkpoint = torch.load(config.depth_checkpoint_path)
    model_depth.load_state_dict(checkpoint['state_dict'])
    models.append(model_depth)
if config.classgt:
    model_classgt = InceptionI3D(
                    num_classes=config.finetune_num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=1,
                    dropout_keep_prob=config.dropout_keep_prob
                )
    checkpoint = torch.load(config.classgt_checkpoint_path)
    model_classgt.load_state_dict(checkpoint['state_dict'])
    models.append(model_classgt)
if config.inst_seg:
    model_inst_seg = InceptionI3D(
                    num_classes=config.finetune_num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=1,
                    dropout_keep_prob=config.dropout_keep_prob
                )
    checkpoint = torch.load(config.inst_seg_checkpoint_path)
    model_inst_seg.load_state_dict(checkpoint['state_dict'])
    models.append(model_inst_seg)
if config.flow:
    model_flow = InceptionI3D(
                    num_classes=config.finetune_num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=2,
                    dropout_keep_prob=config.dropout_keep_prob
                )
    checkpoint = torch.load(config.flow_checkpoint_path)
    model_flow.load_state_dict(checkpoint['state_dict'])
    models.append(model_flow)

# define fusion layer network
num_modalities = len(models)
if config.fusion_layer_type == 'class':
    fusion_net = fusion_class_layer(num_modalities, config.finetune_num_classes)
elif config.fusion_layer_type == 'uniform':
    fusion_net = fusion_uniform_layer(num_modalities)

parameters = []
if config.fusion_layer_type == 'class':
    for layer_name, layer in fusion_net.layers.items():
            layer.to(device)
            for param_name, param in layer.named_parameters():
                parameters.append({'params': param, 'name': param_name})
elif config.fusion_layer_type == 'uniform':
    fusion_net = fusion_net.to(device)
    for param_name, param in fusion_net.named_parameters():
        parameters.append({'params': param, 'name': param_name})

####################################################################
####################################################################
# Setup of data transformations

if config.no_dataset_mean and config.no_dataset_std:
    # Just zero-center and scale to unit std
    print('Data normalization: no dataset mean, no dataset std')
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not config.no_dataset_mean and config.no_dataset_std:
    # Subtract dataset mean and scale to unit std
    print('Data normalization: use dataset mean, no dataset std')
    norm_method = Normalize(config.image_mean, [1, 1, 1])
else:
    # Subtract dataset mean and scale to dataset std
    print('Data normalization: use dataset mean, use dataset std')
    norm_method = Normalize(config.image_mean, config.image_std)

train_transforms = {
    'spatial':  Compose([MultiScaleRandomCrop(config.scales, config.spatial_size),
                         RandomHorizontalFlip(),
                         ToTensor(config.norm_value),
                         norm_method]),
    'temporal': TemporalRandomCrop(config.sample_duration),
    'target':   ClassLabel()
}

validation_transforms = {
    'spatial':  Compose([CenterCrop(config.spatial_size),
                         ToTensor(config.norm_value),
                         norm_method]),
    'temporal': TemporalRandomCrop(config.sample_duration),
    'target':   ClassLabel()
}

####################################################################
####################################################################
# Setup of data pipeline

data_loaders = data_factory.get_data_loaders(config, train_transforms, validation_transforms)
phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']
print('#'*60)
####################################################################
####################################################################
# Optimizer and loss initialization
criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(config, parameters)

####################################################################
####################################################################

# Keep track of best validation accuracy
val_acc_history = []
best_val_acc = 0.0

# turn off autograd and set model to evaluation mode
for model in models:
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

# Keep track of best validation accuracy
val_acc_history = []
best_val_acc = 0.0
loss_list1 = []
step_list1 = []
epoch_train_loss = []
epoch_train_acc = []
epoch_val_loss = []
epoch_val_acc = []
epoch_list = []


for epoch in range(config.start_epoch, config.num_epochs+1):
    # First 'training' phase, then 'validation' phase
    for phase in phases:

        if phase == 'train':

            # Perform one training epoch
            train_loss, train_acc, train_duration, loss_list1, step_list1 = train_fusion_epoch(
                config=config,
                models=models,
                fusion_model=fusion_net,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                data_loader=data_loaders['train'],
                epoch=epoch,
                loss_list=loss_list1,
                step_list=step_list1,
                summary_writer=None
            )
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
        elif phase == 'validation':

            # Perform one validation epoch
            val_loss, val_acc, val_duration = validation_fusion_epoch(
                config=config,
                models=models,
                fusion_model=fusion_net,
                criterion=criterion,
                device=device,
                data_loader=data_loaders['validation'],
                epoch=epoch,
                summary_writer=None
            )
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)
            val_acc_history.append(val_acc)

    print('#'*60)
    print('EPOCH {} SUMMARY'.format(epoch+1))
    print('Training Phase.')
    print('  Total Duration:              {} minutes'.format(int(np.ceil(train_duration / 60))))
    print('  Average Train Loss:          {:.3f}'.format(train_loss))
    print('  Average Train Accuracy:      {:.3f}'.format(train_acc))

    if 'validation' in phases:
        print('Validation Phase.')
        print('  Total Duration:              {} minutes'.format(int(np.ceil(val_duration / 60))))
        print('  Average Validation Loss:     {:.3f}'.format(val_loss))
        print('  Average Validation Accuracy: {:.3f}'.format(val_acc))

    if 'validation' in phases and val_acc > best_val_acc:
        checkpoint_path = os.path.join(config.fusion_layer_save_dir, 'save_best.pth')
        save_checkpoint(checkpoint_path, epoch, fusion_net.state_dict(), optimizer.state_dict())
        print('Found new best validation accuracy: {:.3f}'.format(val_acc))
        print('Model checkpoint (best) written to:     {}'.format(checkpoint_path))
        best_val_acc = val_acc

    # Model saving
    if epoch % config.checkpoint_frequency == 0:
        checkpoint_path = os.path.join(config.fusion_layer_save_dir, 'save_{:03d}.pth'.format(epoch+1))
        save_checkpoint(checkpoint_path, epoch, fusion_net.state_dict(), optimizer.state_dict())
        print('Model checkpoint (periodic) written to: {}'.format(checkpoint_path))
        cleanup_fusion_checkpoint_dir(config)  # remove old checkpoint filesz

    epoch_list.append(epoch+1)
    # Early stopping
    if epoch > config.early_stopping_patience:
        last_val_acc = val_acc_history[-config.early_stopping_patience:]
        if all(acc < best_val_acc for acc in last_val_acc):
            # All last validation accuracies are smaller than the best
            print('Early stopping because validation accuracy has not '
                  'improved the last {} epochs.'.format(config.early_stopping_patience))
            break


print('Finished training.')

'''
# starting validation loop
for step, (clips, targets) in enumerate(data_loader):
    start_time = time.time()

    # Move inputs to GPU memory
    clips = clips.to(device)
    clip_list = []
    # seperate clips into corresponding slices according to model types
    if config.rgb:
        clip_rgb = clips[:,rgb_ch_start:rgb_ch_start+3,:,:,:]
        clip_list.append(clip_rgb)
    if config.depth:
        clip_depth = clips[:, depth_ch_start:depth_ch_start+1,:,:,:]
        clip_list.append(clip_depth)
    if config.classgt:
        clip_classgt = clips[:, classgt_ch_start:classgt_ch_start + 1, :, :, :]
        clip_list.append(clip_classgt)
    if config.inst_seg:
        if config.flow:
            clip_inst_seg = clips[:, inst_seg_ch_start:inst_seg_ch_start+1, :, :, :]
            clip_list.append(clip_inst_seg)
        else:
            clip_inst_seg = clips[:, inst_seg_ch_start:, :, :, :]
            clip_list.append(clip_inst_seg)
    if config.flow:
        clip_flow = clips[:, flow_ch_start:, :, :, :]
        clip_list.append(clip_flow)

    # Feed-forward through the networks and store logits in list
    logit_list = []
    for i in range(len(models)):
        logits = models[i].forward(clip_list[i])
        logit_list.append(logits)

    logits_concat = torch.cat(logit_list, 2)

    learned_logits_concat = fusion_net.forward(logits_concat)
    learned_logits_concat = torch.cat(learned_logits_concat, dim =1).unsqueeze(2)
    _, preds = torch.max(learned_logits_concat, 1)

    targets = targets.unsqueeze(1).to(device)
    loss = criterion(learned_logits_concat, targets)
    sys.exit()
    # Calculate accuracy
    correct = torch.sum(preds == targets.data)
    accuracy = correct.double() / config.batch_size

    # Calculate elapsed time for this step
    examples_per_second = config.batch_size / float(time.time() - start_time)

    # Save statistics
    accuracies[step] = accuracy.item()
    losses[step] = loss.item()

    if step % config.print_frequency == 0:
        print("[{}] Test Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
              "Accuracy = {:.3f}, Loss = {:.3f}".format(
            datetime.now().strftime("%A %H:%M"),
            step, steps_in_epoch, examples_per_second,
            accuracies[step], losses[step]))


# Epoch statistics
epoch_duration = float(time.time() - epoch_start_time)
avg_loss = np.mean(losses)
avg_acc = np.mean(accuracies)

dict_data = [
    {'epoch_duration': epoch_duration, 'avg_loss': avg_loss,'avg_acc': avg_acc}
]
csv_columns = ['epoch_duration','avg_loss','avg_acc']

print('Saving results to:  acc.txt')
with open(config.fusion_save_dir + '/info.txt', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)

print('  Total Duration:              {} minutes'.format(int(np.ceil(epoch_duration / 60))))
print('  Average Validation Loss:     {:.3f}'.format(avg_loss))
print('  Average Validation Accuracy: {:.3f}'.format(avg_acc))


print('Finished validation.')
'''

'''
python3 fusion_test.py --dataset=phav --device=cuda:1 --no_dataset_mean --no_dataset_std --batch_size=1 --annotation_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos/phavTrainTestlist/norainfog/phav.json --multi_modal --rgb --depth --classgt --inst_seg --flow --rgb_checkpoint_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/20190905_1636_phav_i3d_lr0.010_RGB/checkpoints/save_best.pth --flow_checkpoint_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/20190907_0454_phav_i3d_lr0.010_FLOW/checkpoints/save_best.pth --depth_checkpoint_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/20190908_1736_phav_i3d_lr0.010_DEPTH/checkpoints/save_best.pth --classgt_checkpoint_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/20190910_0948_phav_i3d_lr0.010_CLASSGT/checkpoints/save_best.pth --inst_seg_checkpoint_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/20190912_0002_phav_i3d_lr0.010_INSTSEG/checkpoints/save_best.pth
python3 fusion_test.py --dataset=phav --device=cuda:0 --no_dataset_mean --no_dataset_std --batch_size=4 --annotation_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos/phavTrainTestlist/norainfog/phav.json --multi_modal --rgb --rgb_checkpoint_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/20190905_1636_phav_i3d_lr0.010_RGB/checkpoints/save_best.pth
'''