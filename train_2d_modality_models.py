from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import torch.nn as nn

from transforms.spatial_transforms import Compose, CornerCrop, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

from epoch_iterators import train_proxy_epoch, validation_proxy_epoch
from utils.utils import *
import utils.mean_values
import factory.data_factory as data_factory
import factory.model_factory as model_factory
import factory.loss_factory as loss_factory
from config import parse_opts
import matplotlib.pyplot as plt
import random
import numpy as np
from models.proxy_networks import resnet18, resnet50, unet

####################################################################
####################################################################
# Configuration and logging

config = parse_opts()
# config = prepare_output_dirs(config)
config = init_cropping_scales(config)
config = set_lr_scheduling_policy(config)

config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
config.image_std = utils.mean_values.get_std(config.norm_value)

# print_config(config)
# write_config(config, os.path.join(config.save_dir, 'config.json'))
from moviepy.editor import *
from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir=config.log_dir)
writer = None

# set random seeds
random.seed(config.manual_seed)
np.random.seed(config.manual_seed)

####################################################################
####################################################################
# Initialize model
device = torch.device(config.device)
#torch.backends.cudnn.enabled = False



model = unet(config).to(device)

if config.checkpoint_path:
    model.load_state_dict(torch.load(config.checkpoint_path)['state_dict'])

parameters = []
for param_name, param in model.named_parameters():
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
# norm_method = None

train_transforms = {
    'spatial':  Compose([MultiScaleRandomCrop(config.scales, config.spatial_size),
                         RandomHorizontalFlip(),
                         ToTensor(config.norm_value),
                         norm_method]),
}

# train_transforms = {
#     'spatial':  Compose([CornerCrop(config.scales),
#                          RandomHorizontalFlip(),
#                          ToTensor(config.norm_value),
#                          norm_method]),
# }

validation_transforms = {
    'spatial':  Compose([CenterCrop(config.spatial_size),
                         ToTensor(config.norm_value),
                         norm_method])
}

####################################################################
####################################################################
# Setup of data pipeline
if config.inst_seg:
    data_loaders = data_factory.get_modality_data_loaders2(config, train_transforms, validation_transforms)
else:
    data_loaders = data_factory.get_modality_data_loaders(config, train_transforms, validation_transforms)


phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']
print('#'*60)

###################################################################
####################################################################
# Optimizer and loss initialization
# choose loss based on whatever modality is being predicted
if config.depth:
    criterion = nn.L1Loss()
if config.flow:
    # criterion = loss_factory.L2()
    criterion = nn.L1Loss()
if config.classgt:
    criterion = nn.CrossEntropyLoss()
if config.inst_seg:
    criterion = loss_factory.softIoULoss()



optimizer = get_optimizer(config, parameters)

# Restore optimizer params and set config.start_index
if config.finetune_restore_optimizer:
    restore_optimizer_state(config, optimizer)

# Learning rate scheduler
if config.lr_scheduler == 'plateau':
    assert 'validation' in phases
    print('Plateau after ' + str(config.lr_plateau_patience)+ ' epochs without val loss decreasing')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', config.lr_scheduler_gamma, config.lr_plateau_patience)
else:
    milestones = [int(x) for x in config.lr_scheduler_milestones.split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, config.lr_scheduler_gamma)

####################################################################
####################################################################

# Keep track of best validation accuracy
val_acc_history = []
best_val_acc = 0.0
best_val_loss = 1000
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
            train_loss, train_acc, train_duration = train_proxy_epoch(
                config=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                data_loader=data_loaders['train'],
                epoch=epoch,
                summary_writer=writer
            )
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
        elif phase == 'validation':

            # Perform one validation epoch
            val_loss, val_acc, val_duration = validation_proxy_epoch(
                config=config,
                model=model,
                criterion=criterion,
                device=device,
                data_loader=data_loaders['validation'],
                epoch=epoch,
                summary_writer=writer
            )
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)
            val_acc_history.append(val_acc)

    # Update learning rate
    if config.lr_scheduler == 'plateau':
        scheduler.step(val_loss)
    else:
        scheduler.step(epoch)

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

    if config.depth or config.flow:
        if 'validation' in phases and val_loss < best_val_loss:
            checkpoint_path = os.path.join(config.checkpoint_dir, 'save_best.pth')
            save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
            print('Old best validation loss: {:.3f}'.format(best_val_loss))
            print('Found new best validation loss: {:.3f}'.format(val_loss))
            print('Model checkpoint (best) written to:     {}'.format(checkpoint_path))
            best_val_loss = val_loss
    else:
        if 'validation' in phases and val_acc > best_val_acc:
            checkpoint_path = os.path.join(config.checkpoint_dir, 'save_best.pth')
            save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
            print('Old best validation accuracy: {:.3f}'.format(best_val_acc))
            print('Found new best validation accuracy: {:.3f}'.format(val_acc))
            print('Model checkpoint (best) written to:     {}'.format(checkpoint_path))
            best_val_acc = val_acc

    # Model saving
    if epoch % config.checkpoint_frequency == 0:
        checkpoint_path = os.path.join(config.checkpoint_dir, 'save_{:03d}.pth'.format(epoch+1))
        save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
        print('Model checkpoint (periodic) written to: {}'.format(checkpoint_path))
        cleanup_checkpoint_dir(config)  # remove old checkpoint filesz

    epoch_list.append(epoch+1)
    # Early stopping
    if epoch > config.early_stopping_patience:
        last_val_acc = val_acc_history[-config.early_stopping_patience:]
        if all(acc < best_val_acc for acc in last_val_acc):
            # All last validation accuracies are smaller than the best
            print('Early stopping because validation accuracy has not '
                  'improved the last {} epochs.'.format(config.early_stopping_patience))
            break

# Dump all TensorBoard logs to disk for external processing
writer.export_scalars_to_json(os.path.join(config.save_dir, 'all_scalars.json'))
writer.close()

print('Finished training.')
