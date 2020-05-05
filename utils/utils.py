# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-03-01

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import shutil
import time

import json as simplejson

import numpy as np

import torch
import torch.optim
import torchvision

from multiprocessing import Pool


####################################################################
####################################################################


def iou_pytorch(outputs, labels, SMOOTH = 1e-6):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = torch.from_numpy(np.argmax(outputs[:, :, :, :].detach().cpu().numpy(), axis=1)).unsqueeze(1)
    labels = labels.long().cpu()
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch


def multiclass_IOU(predd, gtt):
    sum_all_iou = 0
    for pred ,gt in zip(predd, gtt):
        pred = np.argmax(np.asarray(pred.detach().cpu()), axis=0)
        gt = np.asarray(gt)
        intersection = np.where(pred == gt, 1, 0)
        union = 12544
        sum_all_iou = sum_all_iou + intersection.sum() / float(union)
    return sum_all_iou/ pred.shape[0]

def multiclass_IOU_v2(pred, gt):
    # inputs: pred (b x c x h x w) (torch tensor)
    #         gt   (b x 1 x h x w) (torch tensor)
    # returns: ave_mean_iou per batch

    pred = np.argmax(np.asarray(pred.detach().cpu()), axis=1)
    gt = np.asarray(gt.detach().cpu()).squeeze(1)
    intersection = np.count_nonzero(np.where(pred == gt, 1, 0))
    union = np.size(gt)
    return intersection / union

class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        # gt = int(gt.item())
        # pred = int(pred.item())
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[int(gt[i]), int(pred[i])] += 1.0
        return m

def get_iou(data_list, class_num, save_path=None):

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')

    return aveJ, j_list


def print_config(config):
    print('#'*60)
    print('Training configuration:')
    for k,v  in vars(config).items():
        print('  {:>20} {}'.format(k, v))
    print('#'*60)

def write_config(config, json_path):
    with open(json_path, 'w') as f:
        f.write(simplejson.dumps(vars(config), indent=4, sort_keys=True))

def output_subdir(config):
    prefix = time.strftime('%Y%m%d_%H%M')
    subdir = "{}_{}_{}_lr{:.3f}".format(prefix, config.dataset, config.model, config.learning_rate)
    return os.path.join(config.save_dir, subdir)

def fusion_output_subdir(config):
    prefix = time.strftime('%Y%m%d_%H%M')
    fusion_name = ''
    if config.rgb:
        fusion_name += 'R'
    if config.depth:
        fusion_name += 'D'
    if config.classgt:
        fusion_name += 'C'
    if config.inst_seg:
        fusion_name += 'I'
    if config.flow:
        fusion_name += 'F'
    subdir = "{}_{}_{}_{}".format(prefix, config.dataset, config.model, fusion_name)
    return os.path.join(config.fusion_save_dir, subdir)

def fusion_layer_output_subdir(config):
    prefix = time.strftime('%Y%m%d_%H%M')
    fusion_name = ''
    if config.rgb:
        fusion_name += 'R'
    if config.depth:
        fusion_name += 'D'
    if config.classgt:
        fusion_name += 'C'
    if config.inst_seg:
        fusion_name += 'I'
    if config.flow:
        fusion_name += 'F'
    subdir = "{}_{}_{}_{}".format(prefix, config.dataset, config.model, fusion_name)
    return os.path.join(config.fusion_layer_save_dir, subdir)

def init_cropping_scales(config):
    # Determine cropping scales
    config.scales = [config.initial_scale]
    for i in range(1, config.num_scales):
        config.scales.append(config.scales[-1] * config.scale_step)
    return config

def set_lr_scheduling_policy(config):
    if config.lr_plateau_patience > 0 and not config.no_eval:
        config.lr_scheduler = 'plateau'
    else:
        config.lr_scheduler = 'multi_step'
    return config

def prepare_output_dirs(config):
    # Set output directories
    config.save_dir = output_subdir(config)
    config.checkpoint_dir = os.path.join(config.save_dir, 'checkpoints')
    config.log_dir = os.path.join(config.save_dir, 'logs')

    # And create them
    if os.path.exists(config.save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.save_dir)

    os.mkdir(config.save_dir)
    os.mkdir(config.checkpoint_dir)
    os.mkdir(config.log_dir)
    return config

def prepare_fusion_output_dirs(config):
    # Set output directories
    config.fusion_save_dir = fusion_output_subdir(config)
    # And create them
    if os.path.exists(config.fusion_save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.fusion_save_dir)

    os.mkdir(config.fusion_save_dir)
    return config

def prepare_fusion_layer_output_dirs(config):
    # Set output directories
    config.fusion_layer_save_dir = fusion_layer_output_subdir(config)
    # And create them
    if os.path.exists(config.fusion_layer_save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.fusion_layer_save_dir)

    os.mkdir(config.fusion_layer_save_dir)
    return config

def cleanup_checkpoint_dir(config):
    checkpoint_files = glob.glob(os.path.join(config.checkpoint_dir, 'save_*.pth'))
    checkpoint_files.sort()
    if len(checkpoint_files) > config.checkpoints_num_keep:
        os.remove(checkpoint_files[0])

def cleanup_fusion_checkpoint_dir(config):
    checkpoint_files = glob.glob(os.path.join(config.fusion_layer_save_dir, 'save_*.pth'))
    checkpoint_files.sort()
    if len(checkpoint_files) > config.checkpoints_num_keep:
        os.remove(checkpoint_files[0])

def duration_to_string(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))

####################################################################
####################################################################

def get_optimizer(config, params):
    if config.optimizer == 'SGD':
        return torch.optim.SGD(params, config.learning_rate, config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return torch.optim.RMSprop(params, config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        return torch.optim.Adam(params, config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError('Chosen optimizer is not supported, please choose from (SGD | adam | rmsprop)')

def restore_optimizer_state(config, optimizer):
    if not config.checkpoint_path: return
    checkpoint = torch.load(config.checkpoint_path)
    if 'optimizer' in checkpoint.keys():
        # I3D model has no optimizer state
        config.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print('WARNING: not restoring optimizer state as it is not found in the checkpoint file.')

def current_learning_rate(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def current_weight_decay(optimizer):
    return optimizer.state_dict()['param_groups'][0]['weight_decay']

def save_checkpoint(save_file_path, epoch, model_state_dict, optimizer_state_dict):
    states = {'epoch': epoch+1, 'state_dict': model_state_dict, 'optimizer':  optimizer_state_dict}
    torch.save(states, save_file_path)

def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))
    return value

def write_clips_as_grid(step, clips, targets, dataset, output_dir='./output/', num_examples=10, n_row=4):
    for i in range(0,num_examples):
        clip = clips[i].permute((1,0,2,3))
        grid = torchvision.utils.make_grid(clip, nrow=n_row, normalize=True)
        class_label = dataset.class_names[int(targets[i].numpy())]
        torchvision.utils.save_image(grid, os.path.join(output_dir, 'step{:04d}_example{:02d}_{}.jpg'.format(step+1, i, class_label)))
