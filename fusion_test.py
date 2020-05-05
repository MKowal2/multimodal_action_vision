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
from models.fusion_networks import fusion_class_layer, fusion_uniform_layer

from utils.utils import *
import utils.mean_values
import factory.data_factory as data_factory
from models.i3d import InceptionI3D
from config import parse_opts
import csv
import matplotlib.pyplot as plt

####################################################################
####################################################################
# Configuration and logging

config = parse_opts()
config = prepare_fusion_output_dirs(config)
config = init_cropping_scales(config)
config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
config.image_std = utils.mean_values.get_std(config.norm_value)

# print and prepare config file and output directories
print_config(config)
write_config(config, os.path.join(config.fusion_save_dir, 'config.json'))
from tensorboardX import SummaryWriter
#writer = SummaryWriter(log_dir=config.log_dir)
writer = None

####################################################################
####################################################################
# Initialize model
device = torch.device(config.device)
#torch.backends.cudnn.enabled = False

# Returns the network instance (I3D, 3D-ResNet etc.)
# Note: this also restores the weights and optionally replaces final layer

# order = [RGB, Flow, Depth, SS, IS]
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
    checkpoint = torch.load(config.rgb_checkpoint_path, map_location=device)
    model_rgb.load_state_dict(checkpoint['state_dict'])
    models.append(model_rgb)
if config.flow:
    model_flow = InceptionI3D(
        num_classes=config.finetune_num_classes,
        spatial_squeeze=True,
        final_endpoint='logits',
        in_channels=2,
        dropout_keep_prob=config.dropout_keep_prob
    )
    checkpoint = torch.load(config.flow_checkpoint_path, map_location=device)
    model_flow.load_state_dict(checkpoint['state_dict'])
    models.append(model_flow)
if config.depth:
    model_depth = InceptionI3D(
                    num_classes=config.finetune_num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=1,
                    dropout_keep_prob=config.dropout_keep_prob
                )
    checkpoint = torch.load(config.depth_checkpoint_path, map_location=device)
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
    checkpoint = torch.load(config.classgt_checkpoint_path, map_location=device)
    model_classgt.load_state_dict(checkpoint['state_dict'])
    models.append(model_classgt)
if config.inst_seg:
    model_inst_seg = InceptionI3D(
                    num_classes=config.finetune_num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=2,
                    dropout_keep_prob=config.dropout_keep_prob
                )
    checkpoint = torch.load(config.inst_seg_checkpoint_path, map_location=device)
    model_inst_seg.load_state_dict(checkpoint['state_dict'])
    models.append(model_inst_seg)


num_modalities = len(models)
if config.fusion_layer:
    if config.fusion_layer_type == 'class':
        fusion_net = fusion_class_layer(num_modalities, config.finetune_num_classes)
        checkpoint = torch.load(config.fusion_layer_checkpoint_path)
        fusion_net.load_state_dict(checkpoint['state_dict'])
    elif config.fusion_layer_type == 'uniform':
        fusion_net = fusion_uniform_layer(num_modalities)
        checkpoint = torch.load(config.fusion_layer_checkpoint_path)
        fusion_net.load_state_dict(checkpoint['state_dict'])

if config.fusion_layer:
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

test_transforms = {
    'spatial':  Compose([CenterCrop(config.spatial_size),
                         ToTensor(config.norm_value),
                         norm_method]),
    'temporal': TemporalRandomCrop(config.sample_duration),
    'target':   ClassLabel()
}

####################################################################
####################################################################
# Setup of data pipeline

data_loader = data_factory.get_fusion_data_loaders(config, test_transforms)
print('#'*60)

####################################################################
####################################################################
# Optimizer and loss initialization
criterion = nn.CrossEntropyLoss()

####################################################################
####################################################################

# Keep track of best validation accuracy
val_acc_history = []
best_val_acc = 0.0

print('Starting with FUSION TEST phase.')

# turn off autograd and set model to evaluation mode
# models
for model in models:
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

if config.fusion_layer:
# fusion net
    fusion_net.eval()
    for param in fusion_net.parameters():
        param.requires_grad = False

# Epoch statistics
steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
losses = np.zeros(steps_in_epoch, np.float32)
accuracies = np.zeros(steps_in_epoch, np.float32)

epoch_start_time = time.time()


# assigning variables to divide videos into corresponding input modalities
# if config.rgb:
#     rgb_ch_start = 0
# if config.depth:
#     if config.rgb:
#         depth_ch_start = 3
#     else:
#         depth_ch_start = 0
# if config.classgt:
#     if config.rgb and config.depth:
#         classgt_ch_start = 4
#     elif config.rgb and not config.depth:
#         classgt_ch_start = 3
#     elif config.depth and not config.rgb:
#         classgt_ch_start = 1
#     else:
#         classgt_ch_start = 0
# if config.inst_seg:
#     if config.flow:
#         inst_seg_ch_start = -3
#     else:
#         inst_seg_ch_start = -1
# if config.flow:
#     flow_ch_start = -2


def calc_save_perclass_acc(accuracy_dict):
    for cls, acc in accuracy_dict.items():
        accuracy_dict[cls] = acc[1]/acc[0]
    with open(config.fusion_save_dir + '/class_acc.txt', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in accuracy_dict.items():
            writer.writerow([key, value])

acc_dict = {}
for i in range(config.finetune_num_classes):
    # tot_preds , tot_correct
    acc_dict[i] = [0,0]

softmax = torch.nn.Softmax(dim=1)
# starting validation loop



rgb_weight = 0.3
flow_weight = 0.3
depth_weight = 0.1
semseg_weight = 0.15
instseg_weight = 0.15

weight_list = []
if config.rgb:
    weight_list.append(rgb_weight)
if config.flow:
    weight_list.append(flow_weight)
if config.depth:
    weight_list.append(depth_weight)
if config.classgt:
    weight_list.append(semseg_weight)
if config.inst_seg:
    weight_list.append(instseg_weight)

# order = [RGB, Flow, Depth, SS, IS]

for step, (clips, targets) in enumerate(data_loader):
    start_time = time.time()

    # Move inputs to GPU memory
    clips = [clip.to(device) for clip in clips]

    # Feed-forward through the networks and store logits in list

    logit_list = []
    logit_sum = 0
    for i in range(len(models)):
        logits = models[i].forward(clips[i])

        # weighted sum version
        logit_sum += weight_list[i]*logits

        # unweighted sum version
        # logit_sum += logits

        # logit_list.append(logits)
    # average logits
    # sys.exit()
    # logits = logits/(i+1)

    # if config.fusion_layer:
    #     if not config.fusion_layer_type:
    #         logits_fuse = torch.stack(logit_list, dim=0).mean(dim=0)
    #
    #     if config.fusion_layer_type == 'uniform':
    #         logits_fuse = torch.cat(logit_list, 2)
    #         logits_fuse = fusion_net.forward(logits_fuse)
    #
    #     if config.fusion_layer_type == 'class':
    #         logits_fuse = torch.cat(logit_list, 2)
    #         logits_fuse = fusion_net.forward(logits_fuse)
    #         logits_fuse = torch.cat(logits_fuse, dim=1).unsqueeze(2)

    val, preds = torch.max(logit_sum, 1)

    # rgb_val, rgb_preds = torch.max(logit_list[0], 1)
    # flow_val, flow_preds = torch.max(logit_list[1], 1)
    # print('rgb', rgb_preds.item(), rgb_val.item())
    # print('flow', flow_preds.item(), flow_val.item())
    # print('pred_final', preds.item(), val)
    # print('label', targets.item())

    targets = targets.unsqueeze(1).to(device)
    acc_dict[targets.item()][0] += 1
    if preds == targets.data:
        acc_dict[targets.item()][1] += 1

    loss = criterion(logit_sum, targets)

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

calc_save_perclass_acc(acc_dict)
print('Saving results to:  acc.txt')
with open(config.fusion_save_dir + '/info.txt', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)

print('  Total Duration:              {:.3f} minutes'.format(int(np.ceil(epoch_duration / 60))))
print('  Average Validation Loss:     {:.3f}'.format(avg_loss))
print('  Average Validation Accuracy: {:.3f}'.format(avg_acc))


print('Finished validation.')


'''
python3 fusion_test.py --dataset=phav --device=cuda:0 --no_dataset_mean --no_dataset_std --batch_size=1 --multi_modal --rgb --depth --classgt --inst_seg --flow
python3 fusion_test.py --dataset=phav --device=cuda:0 --no_dataset_mean --no_dataset_std --batch_size=4 --annotation_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos/phavTrainTestlist/norainfog/phav.json --multi_modal --rgb --rgb_checkpoint_path=/home/m3kowal/Research/vfhlt/PyTorchConv3D/output/20190905_1636_phav_i3d_lr0.010_RGB/checkpoints/save_best.pth
python3 fusion_test.py --dataset=phav --device=cuda:1 --fusion_layer_type=class --no_dataset_mean --no_dataset_std --batch_size=1  --multi_modal --rgb --depth --classgt --inst_seg --flow
'''