from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime
from itertools import permutations, combinations

import torch.nn as nn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

from epoch_iterators import test_only_epoch
from utils.utils import *
import utils.mean_values
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from config import parse_opts
import matplotlib
import random
import numpy as np
from moviepy.editor import *
# from mlxtend.plotting import plot_confusion_matrix
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']='"tkagg"'
matplotlib.use('tkagg')
# python train.py --dataset=phav --no_dataset_mean --no_dataset_std --proxy --multi_modal --inst_seg --device=cuda:3 --num_epochs=100

####################################################################
####################################################################
# Configuration and logging

config = parse_opts()
config = init_cropping_scales(config)
config = set_lr_scheduling_policy(config)

config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
config.image_std = utils.mean_values.get_std(config.norm_value)
print_config(config)

# writer = None
# set random seeds
random.seed(config.manual_seed)
np.random.seed(config.manual_seed)

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

####################################################################
####################################################################
# Initialize model
#torch.backends.cudnn.enabled = False

# Returns the network instance (I3D, 3D-ResNet etc.)
# Note: this also restores the weights and optionally replaces final layer

####################################################################
####################################################################
# Setup of data transformations

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

###################################################################
####################################################################
# Optimizer and loss initialization

criterion = nn.CrossEntropyLoss()

####################################################################
####################################################################
# Keep track of best validation accuracy
# val_acc_history = []
# best_val_acc = 0.0
# loss_list1 = []
# step_list1 = []
# epoch_train_loss = []
# epoch_train_acc = []
# epoch_val_loss = []
# epoch_val_acc = []
# epoch_list = []

# loading resnet18s
# checkpoint = torch.load('/mnt/zeta_share_1/m3kowal/outputs/vfhlt_PHAV/20200415_2059_phav_resnet-18_lr0.010_GT_INSTANCE_64/checkpoints/save_best.pth', map_location=device)['state_dict']
# model.load_state_dict(checkpoint)

config.only_eval = True

# set True for test set, False for validation
config.test_eval = True

config.model = 'i3d'
config.model_depth = 50

config.device = 'cuda:0'
device = torch.device(config.device)
if config.device == 'cuda:1':
    config.batch_size = 96
else:
    config.batch_size = 96

# list modalities to use here and their checkpoint paths!
rgb = False
rgb_path = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200404_1454_kinetics_resnet_lr0.010_RGB/checkpoints/save_best.pth'
rgb_pathi3d = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191103_1426_kinetics_i3d_lr0.010_RGB_FINAL/checkpoints/save_best.pth'
rgb_path18 = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200405_1025_phav_resnet-50_lr0.010_RGB/checkpoints/save_best.pth'
flow = False
flow_path = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200405_1040_kinetics_resnet_lr0.010_FLOW/checkpoints/save_best.pth'
flow_pathi3d = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191027_1702_kinetics_i3d_lr0.010_FLOW_FINAL/checkpoints/save_best.pth'
flow_path18 = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200404_1449_phav_resnet-50_lr0.010_FLOW_GT/checkpoints/save_best.pth'
depth = False
depth_path = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200421_2320_kinetics_resnet_lr0.010_DEPTH_v2/checkpoints/save_best.pth'
depth_pathi3d = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191101_2025_kinetics_i3d_lr0.010_DEPTH_FINAL/checkpoints/save_best.pth'
depth_path18 = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200405_1026_phav_resnet-50_lr0.010_DEPTH_GT/checkpoints/save_best.pth'
classgt_things = False
classgt_things_path = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200409_1148_kinetics_resnet_lr0.010_SS_THINGS/checkpoints/save_best.pth'
classgt_things_pathi3d = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191028_1945_kinetics_i3d_lr0.010_SS_FINAL/checkpoints/save_best.pth'
classgt_things_path18 = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200408_1056_phav_resnet-50_lr0.010_SS_THINGS_GT/checkpoints/save_best.pth'
classgt_stuff = False
classgt_stuff_path = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200410_1142_kinetics_resnet_lr0.010_SS_STUFF/checkpoints/save_best.pth'
classgt_stuff_pathi3d = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200404_1041_kinetics_i3d_lr0.001_SS_STUFF_FINAL/checkpoints/save_best.pth'
classgt_stuff_path18 = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200415_1654_phav_resnet-50_lr0.010_SS_STUFF_GT/checkpoints/save_best.pth'
inst_seg = False
inst_seg_path = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200410_1138_kinetics_resnet_lr0.010_INST_SEG/checkpoints/save_best.pth'
inst_seg_pathi3d = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191101_2015_kinetics_i3d_lr0.010_IS_FINAL/checkpoints/save_best.pth'
inst_seg_path18 = '/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200415_1654_phav_resnet-50_lr0.010_SS_STUFF_GT/checkpoints/save_best.pth'

if config.model == 'resnet':
    # modality_dict = {'rgb': [rgb, rgb_path], 'flow': [flow, flow_path], 'depth': [depth, depth_path],
    #              'classgt_things': [classgt_things, classgt_things_path], 'classgt_stuff': [classgt_stuff, classgt_stuff_path],
    #              'inst_seg': [inst_seg, inst_seg_path]}
    modality_dict = {'rgb': [rgb, rgb_path], 'flow': [flow, flow_path], 'depth': [depth, depth_path],
                     'classgt_things': [classgt_things, classgt_things_path],'classgt_stuff': [classgt_stuff, classgt_stuff_path],
                     'inst_seg': [inst_seg, inst_seg_path]}
elif config.model == 'i3d':
    modality_dict = {'rgb': [rgb, rgb_pathi3d], 'flow': [flow, flow_pathi3d], 'depth': [depth, depth_pathi3d],
                     'classgt_things': [classgt_things, classgt_things_pathi3d],
                     'classgt_stuff': [classgt_stuff, classgt_stuff_pathi3d],
                     'inst_seg': [inst_seg, inst_seg_pathi3d]}

##### Choose number of streams! #####
# num_streams = 2

list_of_modalities = ['rgb', 'flow', 'depth', 'classgt_stuff', 'inst_seg']
# list_of_modalities = ['rgb', 'flow', 'depth', 'classgt_things', 'inst_seg']

acc_dict = {}
with torch.no_grad():
    for modality in list_of_modalities:
        # set modalities to false so other training doesn't effect
        config.multi_modal = False
        config.rgb = False
        config.flow = False
        config.depth = False
        config.classgt = False
        config.classgt_stuff = False
        config.inst_seg = False
        config.only_person = False
        config.inst_person = False
        test_path = modality_dict[modality][1]
        if modality == 'rgb':
            print('Performing RGB test epoch')
            config.rgb = True
        else:
            config.multi_modal = True
            if modality == 'flow':
                print('Performing Optical Flow test epoch')
                config.flow = True
            elif modality == 'depth':
                print('Performing Depth test epoch')
                config.depth = True
            elif modality == 'inst_seg':
                print('Performing Instance Segmentation test epoch')
                config.inst_seg = True
            elif modality == 'classgt_things':
                print('Performing Sem. Seg. (Things) test epoch')
                config.classgt = True
            elif modality == 'classgt_stuff':
                print('Performing Sem. Seg. (Stuff) test epoch')
                config.classgt_stuff = True

        # load dataloader for individual modality
        data_loader = data_factory.get_test_data_loader(config, validation_transforms)['validation']
        model, parameters = model_factory.get_model(config)

        checkpoint = torch.load(test_path, map_location=device)['state_dict']
        model.load_state_dict(checkpoint)

        # perform single test epoch
        val_loss, val_acc, val_duration, logits_list, targets = test_only_epoch(
            config=config,
            model=model,
            criterion=criterion,
            device=device,
            data_loader=data_loader,
            epoch=0,
            summary_writer=None)
        print('{} Accuracy is {:.3f}'.format(modality, val_acc))
        print('Added logits to list...')

        acc_dict[modality] = logits_list



streams = [1,2,3,4,5]
for n in streams:
    ave_n_stream_accuracy_list = []
    comb_modal_list = list(combinations(list_of_modalities, n))
    for combination in comb_modal_list:
        multi_modal_logits_list = []
        target_list = []
        for modality in combination:
            multi_modal_logits_list.append(acc_dict[modality])
        steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
        accuracies = np.zeros(steps_in_epoch, np.float32)
        num_modalities = len(multi_modal_logits_list)
        final_logits_list = []
        for i in range(len(multi_modal_logits_list[0])):
            same_example_pred = []
            for j in range(num_modalities):
                same_example_pred.append(multi_modal_logits_list[j][i])

            final_logits_list.append(torch.stack(same_example_pred, axis=0).sum(axis=0))

        # evaluate final multi-stream prediction
        for k, preds in enumerate(final_logits_list):
            correct = torch.sum(torch.max(preds, 1)[1] == targets[k].data)
            accuracy = correct.double() / config.batch_size
            accuracies[k] = accuracy.item()

        single_n_stream_accuracy = np.mean(accuracies)
        print('RESULT FOR MULTI_MODAL TEST PHASE.')
        print('---Average Test Accuracy for {}: {:.5f}'.format(combination, single_n_stream_accuracy))

        ave_n_stream_accuracy_list.append(single_n_stream_accuracy)

    final_acc = sum(ave_n_stream_accuracy_list) / len(ave_n_stream_accuracy_list)
    print('################# AVERAGE TEST ACC. FOR {}-STREAMS: {:.3f}'.format(n, final_acc * 100))




#             multi_modal_logits_list.append(logits_list)
#
#            # class_list = []
#                 # with open('classInd.txt') as f:
#                 #     lines = f.readlines()
#                 #     for line in lines:
#                 #         class_list.append(line[2:-1])
#                 #
#                 # fig, ax = plot_confusion_matrix(conf_mat=conf,
#                 #                                 class_names=class_list,
#                 #                                 show_absolute=True,
#                 #                                 show_normed=True,
#                 #                                 colorbar=True)
#                 # plt.show()
#
#         # set up accuracies
#         steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
#         accuracies = np.zeros(steps_in_epoch, np.float32)
#
#         num_modalities = len(multi_modal_logits_list)
#         final_logits_list = []
#         for i in range(len(multi_modal_logits_list[0])):
#             same_example_pred = []
#             for j in range(num_modalities):
#                 same_example_pred.append(multi_modal_logits_list[j][i])
#
#             final_logits_list.append(torch.stack(same_example_pred, axis=0).sum(axis=0))
#
#         # evaluate final multi-stream prediction
#         for k, preds in enumerate(final_logits_list):
#             correct = torch.sum(torch.max(preds, 1)[1] == targets[k].data)
#             accuracy = correct.double() / config.batch_size
#             accuracies[k] = accuracy.item()
#
#         single_n_stream_accuracy = np.mean(accuracies)
#         print('RESULT FOR MULTI_MODAL TEST PHASE.')
#         print('---Average Test Accuracy: {:.5f}'.format(single_n_stream_accuracy))
#
#         ave_n_stream_accuracy_list.append(single_n_stream_accuracy)
#
# final_acc = sum(ave_n_stream_accuracy_list) / len(ave_n_stream_accuracy_list)
#
# print('Average Test Accuracy for {}-Streams: {:.3f}'.format(num_streams, final_acc*100))
