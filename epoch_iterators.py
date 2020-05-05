from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from utils.utils import *
from factory.loss_factory import batch_to_var

import torch
import torchvision
import matplotlib.pyplot as plt
import sys
import os
import json
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']='"qt4agg"'

def train_epoch(config, model, criterion, optimizer, device,
                data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with training phase.'.format(epoch+1))

    model.train()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)
    epoch_start_time = time.time()
    for step, (clips, targets) in enumerate(data_loader):
        start_time = time.time()

        # sanity check - overfit to one batch
        # if step ==0:
        #     clips = clips1
        #     targets = targets1
        #     clips = clips.to(device)
        #     targets = targets.to(device)
        #     targets = torch.unsqueeze(targets, -1)

        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        clips = clips.to(device)
        targets = targets.to(device)

        if config.model == 'i3d':
            targets = torch.unsqueeze(targets, -1)

        # 2D resnet adjustment
        if config.model_dimension == 2:
            clips = clips.squeeze(2)

        # Feed-forward through the network
        logits = model.forward(clips)

        _, preds = torch.max(logits, 1)

        loss = criterion(logits, targets)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / config.batch_size

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size/float(time.time() - start_time)

        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        # Compute the global step, only for logging
        global_step = (epoch*steps_in_epoch) + step

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}/{}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "LR = {:.4f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                    step, steps_in_epoch, examples_per_second,
                    current_learning_rate(optimizer), accuracies[step], losses[step]))


        # TensorboardX Logging
        if summary_writer and step % config.log_frequency == 0:
            summary_writer.add_scalar('train/loss', losses[step], global_step)
            summary_writer.add_scalar('train/accuracy', accuracies[step], global_step)
            summary_writer.add_scalar('train/examples_per_second', examples_per_second, global_step)
            summary_writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
            summary_writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)

        # if summary_writer and global_step % config.log_image_frequency == 0:
        #     # TensorboardX video summary
        #     for example_idx in range(4):
        #         clip_for_display = clips[example_idx].clone().cpu()
        #         min_val = float(clip_for_display.min())
        #         max_val = float(clip_for_display.max())
        #         clip_for_display.clamp_(min=min_val, max=max_val)
        #         clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
        #         # print(clip_for_display.shape)  -> torch.Size([3, 64, 224, 224])
        #         # needs to be: (𝑁,𝑇,𝐶,𝐻,𝑊)
        #         # (number of data,
        #         # print(clip_for_display.unsqueeze(0).shape) -> torch.Size([1, 3, 64, 224, 224])
        #         # clip_for_display = clip_for_display.unsqueeze(0).permute(0,2,1,3,4)
        #         # if config.dataset == 'phav':
        #         #     for action_category in class_dict:
        #         #         if targets.data[example_idx].cpu().numpy()[0] == action_category-1:
        #         #             action_label = str(class_dict[action_category])
        #         #             break
        #         #     summary_writer.add_video('train_clips/{:04d}'.format(example_idx) + action_label, clip_for_display, global_step)
        #         # else:
        #         if config.model_dimension == 2:
        #             summary_writer.add_image('train_clips/{:04d}'.format(example_idx), clip_for_display,
        #                                      global_step)
        #         else:
        #             clip_for_display = clip_for_display.unsqueeze(0).permute(0, 2, 1, 3, 4)
        #             summary_writer.add_video('train_clips/{:04d}'.format(example_idx), clip_for_display,
        #                                      global_step)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_loss', {'train': epoch_avg_loss}, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'train': epoch_avg_acc}, epoch)


    return epoch_avg_loss, epoch_avg_acc, epoch_duration


####################################################################
####################################################################


def validation_compare_epoch(config, model, criterion, device, data_loader, epoch, summary_writer=None):
    print('#' * 60)
    print('Epoch {}. Starting with validation phase.'.format(epoch + 1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    class_dict = {}
    for i in range(100):
        # correct , total
        class_dict[i] = [0, 0]

    cls_path = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/kinetics100_V2.json'
    with open(cls_path) as f:
        class_names = json.load(f)['labels']

    label_list = []
    for i, label in enumerate(class_names):
        label_list.append(label)

    with torch.no_grad():
        for step, (clips, targets) in enumerate(data_loader):
            start_time = time.time()
            # Move inputs to GPU memory
            clips = clips.to(device)

            if config.model[0] == 'i3d':
                targets = torch.unsqueeze(targets, -1)

            targets = targets.to(device)

            # 2D resnet adjustment
            if config.model_dimension == 2:
                clips = clips.squeeze(2)

            # Feed-forward through the network
            logits = model[0].forward(clips)
            logits2 = model[1].forward(clips)

            _, preds = torch.max(logits, 1)
            _, preds2 = torch.max(logits2, 1)
            loss = criterion(logits, targets)

            # Calculate accuracy
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / config.batch_size

            # get categorical stats
            for i, pred in enumerate(preds):
                class_dict[targets[i].item()][1] += 1
                if pred.item() == targets[i].item():
                    class_dict[targets[i].item()][0] += 1

            for i, pred in enumerate(preds):
                if pred.item() != targets[i].item() and preds[i].item() == targets[i].item():
                    # first one didn't get it, 2nd did
                    pred_label = label_list[pred.item()]
                    pred2_label = label_list[preds[i].item()]

            # Calculate elapsed time for this step
            examples_per_second = config.batch_size / float(time.time() - start_time)

            # Save statistics
            accuracies[step] = accuracy.item()
            losses[step] = loss.item()

            if step % config.print_frequency == 0:
                print("[{}] Epoch {}/{}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                    step, steps_in_epoch, examples_per_second,
                    accuracies[step], losses[step]))

            # if summary_writer and step == 0:
            #     # TensorboardX video summary
            #     for example_idx in range(4):
            #         clip_for_display = clips[example_idx].clone().cpu()
            #         min_val = float(clip_for_display.min())
            #         max_val = float(clip_for_display.max())
            #         clip_for_display.clamp_(min=min_val, max=max_val)
            #         clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
            #         clip_for_display = clip_for_display.unsqueeze(0).permute(0, 2, 1, 3, 4)
            #         if config.dataset == 'phav':
            #             for action_category in class_dict:
            #                 if targets.data[example_idx].cpu().numpy()[0] == action_category-1:
            #                     action_label = str(class_dict[action_category])
            #                     break
            #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx) + action_label, clip_for_display, epoch*steps_in_epoch)
            #         else:
            #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx), clip_for_display, epoch*steps_in_epoch)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('validation/epoch_avg_accuracy', epoch_avg_acc, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_loss', {'val': epoch_avg_loss}, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'val': epoch_avg_acc}, epoch)

    for cls in class_dict:
        class_dict[cls] = class_dict[cls][0] / class_dict[cls][1]

    cls_path = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/kinetics100_V2.json'
    with open(cls_path) as f:
        class_names = json.load(f)['labels']

    named_class_dict = {}
    for i, label in enumerate(class_names):
        named_class_dict[label] = class_dict[i]

    return epoch_avg_loss, epoch_avg_acc, epoch_duration, named_class_dict



def validation_category_epoch(config, model, criterion, device, data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with validation phase.'.format(epoch+1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    class_dict = {}
    for i in range(100):
        # correct , total
        class_dict[i] = [0,0]

    cls_path = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/kinetics100_V2.json'
    with open(cls_path) as f:
        class_names = json.load(f)['labels']

    label_list = []
    for i, label in enumerate(class_names):
        label_list.append(label)


    with torch.no_grad():
        for step, (clips, targets) in enumerate(data_loader):
            start_time = time.time()
            # Move inputs to GPU memory
            clips   = clips.to(device)

            if config.model == 'i3d':
                targets = torch.unsqueeze(targets, -1)

            targets = targets.to(device)

            # 2D resnet adjustment
            if config.model_dimension == 2:
                clips = clips.squeeze(2)

            # Feed-forward through the network
            logits = model.forward(clips)

            _, preds = torch.max(logits, 1)
            loss = criterion(logits, targets)

            # Calculate accuracy
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / config.batch_size

            # get categorical stats
            for i, pred in enumerate(preds):
                class_dict[targets[i].item()][1] += 1
                if pred.item() == targets[i].item():
                    class_dict[targets[i].item()][0] += 1

            for i, pred in enumerate(preds):
                if pred.item() != targets[i].item() and preds[i].item() == targets[i].item():
                    # first one didn't get it, 2nd did
                    pred_label = label_list[pred.item()]

            # Calculate elapsed time for this step
            examples_per_second = config.batch_size/float(time.time() - start_time)

            # Save statistics
            accuracies[step] = accuracy.item()
            losses[step] = loss.item()

            if step % config.print_frequency == 0:
                print("[{}] Epoch {}/{}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.3f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                        step, steps_in_epoch, examples_per_second,
                        accuracies[step], losses[step]))

            # if summary_writer and step == 0:
            #     # TensorboardX video summary
            #     for example_idx in range(4):
            #         clip_for_display = clips[example_idx].clone().cpu()
            #         min_val = float(clip_for_display.min())
            #         max_val = float(clip_for_display.max())
            #         clip_for_display.clamp_(min=min_val, max=max_val)
            #         clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
            #         clip_for_display = clip_for_display.unsqueeze(0).permute(0, 2, 1, 3, 4)
            #         if config.dataset == 'phav':
            #             for action_category in class_dict:
            #                 if targets.data[example_idx].cpu().numpy()[0] == action_category-1:
            #                     action_label = str(class_dict[action_category])
            #                     break
            #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx) + action_label, clip_for_display, epoch*steps_in_epoch)
            #         else:
            #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx), clip_for_display, epoch*steps_in_epoch)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('validation/epoch_avg_accuracy', epoch_avg_acc, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_loss', {'val': epoch_avg_loss}, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'val': epoch_avg_acc}, epoch)

    for cls in class_dict:
        class_dict[cls] = class_dict[cls][0] / class_dict[cls][1]

    cls_path = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/kinetics100_V2.json'
    with open(cls_path) as f:
        class_names = json.load(f)['labels']

    named_class_dict = {}
    for i, label in enumerate(class_names):
        named_class_dict[label] = class_dict[i]

    return epoch_avg_loss, epoch_avg_acc, epoch_duration, named_class_dict

def validation_epoch(config, model, criterion, device, data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with validation phase.'.format(epoch+1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    with torch.no_grad():
        for step, (clips, targets) in enumerate(data_loader):
            start_time = time.time()
            # Move inputs to GPU memory
            clips   = clips.to(device)

            if config.model == 'i3d':
                targets = torch.unsqueeze(targets, -1)

            targets = targets.to(device)

            # 2D resnet adjustment
            if config.model_dimension == 2:
                clips = clips.squeeze(2)

            # Feed-forward through the network
            logits = model.forward(clips)

            _, preds = torch.max(logits, 1)
            loss = criterion(logits, targets)

            # Calculate accuracy
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / config.batch_size

            # Calculate elapsed time for this step
            examples_per_second = config.batch_size/float(time.time() - start_time)

            # Save statistics
            accuracies[step] = accuracy.item()
            losses[step] = loss.item()

            if step % config.print_frequency == 0:
                print("[{}] Epoch {}/{}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.3f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                        step, steps_in_epoch, examples_per_second,
                        accuracies[step], losses[step]))

            # if summary_writer and step == 0:
            #     # TensorboardX video summary
            #     for example_idx in range(4):
            #         clip_for_display = clips[example_idx].clone().cpu()
            #         min_val = float(clip_for_display.min())
            #         max_val = float(clip_for_display.max())
            #         clip_for_display.clamp_(min=min_val, max=max_val)
            #         clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
            #         clip_for_display = clip_for_display.unsqueeze(0).permute(0, 2, 1, 3, 4)
            #         if config.dataset == 'phav':
            #             for action_category in class_dict:
            #                 if targets.data[example_idx].cpu().numpy()[0] == action_category-1:
            #                     action_label = str(class_dict[action_category])
            #                     break
            #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx) + action_label, clip_for_display, epoch*steps_in_epoch)
            #         else:
            #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx), clip_for_display, epoch*steps_in_epoch)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('validation/epoch_avg_accuracy', epoch_avg_acc, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_loss', {'val': epoch_avg_loss}, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'val': epoch_avg_acc}, epoch)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration

def test_only_epoch(config, model, criterion, device, data_loader, epoch, summary_writer=None):

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)
    epoch_start_time = time.time()

    logits_list = []
    target_list = []
    with torch.no_grad():
        for step, (clips, targets) in enumerate(data_loader):
            start_time = time.time()
            # Move inputs to GPU memory
            clips   = clips.to(device)

            if config.model == 'i3d':
                targets = torch.unsqueeze(targets, -1)

            targets = targets.to(device)

            # 2D resnet adjustment
            if config.model_dimension == 2:
                clips = clips.squeeze(2)

            # Feed-forward through the network
            logits = model.forward(clips)

            _, preds = torch.max(logits, 1)
            loss = criterion(logits, targets)
            
            logits_list.append(logits)
            target_list.append(targets)

            # Calculate accuracy
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / config.batch_size

            # Calculate elapsed time for this step
            examples_per_second = config.batch_size/float(time.time() - start_time)

            # Save statistics
            accuracies[step] = accuracy.item()
            losses[step] = loss.item()

            if step % 20 == 0:
                print("[{}]. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.3f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%A %H:%M"),
                        step, steps_in_epoch, examples_per_second,
                        accuracies[step], losses[step]))

    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = np.mean(accuracies)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration, logits_list, target_list

#####################################################################
#####################################################################
#####################################################################

# Fusion iterators


def train_fusion_epoch(config, models, fusion_model, criterion, optimizer, device,
                data_loader, epoch, summary_writer=None):
    print('#' * 60)
    print('Epoch {}. Starting with training phase.'.format(epoch + 1))

    fusion_model.train()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)
    # loss_list = []
    # step_list = []
    epoch_start_time = time.time()

    # assigning variables to divide videos into corresponding input modalities
    if config.rgb:
        rgb_ch_start = 0
    if config.depth:
        if config.rgb:
            depth_ch_start = 3
        else:
            depth_ch_start = 0
    if config.classgt:
        if config.rgb and config.depth:
            classgt_ch_start = 4
        elif config.rgb and not config.depth:
            classgt_ch_start = 3
        elif config.depth and not config.rgb:
            classgt_ch_start = 1
        else:
            classgt_ch_start = 0
    if config.inst_seg:
        if config.flow:
            inst_seg_ch_start = -3
        else:
            inst_seg_ch_start = -1
    if config.flow:
        flow_ch_start = -2

    softmax = torch.nn.Softmax(dim=1)

    for step, (clips, targets) in enumerate(data_loader):

        start_time = time.time()
        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        clips = clips.to(device)
        clip_list = []
        # seperate clips into corresponding slices according to model types
        if config.rgb:
            clip_rgb = clips[:, rgb_ch_start:rgb_ch_start + 3, :, :, :]
            clip_list.append(clip_rgb)
        if config.depth:
            clip_depth = clips[:, depth_ch_start:depth_ch_start + 1, :, :, :]
            clip_list.append(clip_depth)
        if config.classgt:
            clip_classgt = clips[:, classgt_ch_start:classgt_ch_start + 1, :, :, :]
            clip_list.append(clip_classgt)
        if config.inst_seg:
            if config.flow:
                clip_inst_seg = clips[:, inst_seg_ch_start:inst_seg_ch_start + 1, :, :, :]
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
            # normalize scores
            logits = softmax(logits)
            logit_list.append(logits)

        logits_concat = torch.cat(logit_list, 2)
        learned_logits_concat = fusion_model.forward(logits_concat)

        if config.fusion_layer_type == 'class':
            learned_logits_concat = torch.cat(learned_logits_concat, dim=1).unsqueeze(2)

        targets = targets.to(device).unsqueeze(-1)

        _, preds = torch.max(learned_logits_concat, 1)

        loss = criterion(learned_logits_concat, targets)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / config.batch_size

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size / float(time.time() - start_time)

        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        # Compute the global step, only for logging
        global_step = (epoch * steps_in_epoch) + step
        loss_list.append(losses[step])
        step_list.append(global_step)

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}/{}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "LR = {:.4f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                step, steps_in_epoch, examples_per_second,
                current_learning_rate(optimizer), accuracies[step], losses[step]))

        # TensorboardX Logging
        # if summary_writer and step % config.log_frequency == 0:
        #     summary_writer.add_scalar('train/loss', losses[step], global_step)
        #     summary_writer.add_scalar('train/accuracy', accuracies[step], global_step)
        #     summary_writer.add_scalar('train/examples_per_second', examples_per_second, global_step)
        #     summary_writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
        #     summary_writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)

        # if summary_writer and step % config.log_image_frequency == 0:
        #     # TensorboardX video summary
        #     for example_idx in range(4):
        #         clip_for_display = clips[example_idx].clone().cpu()
        #         min_val = float(clip_for_display.min())
        #         max_val = float(clip_for_display.max())
        #         clip_for_display.clamp_(min=min_val, max=max_val)
        #         clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
        #         # print(clip_for_display.shape)  -> torch.Size([3, 64, 224, 224])
        #         # needs to be: (𝑁,𝑇,𝐶,𝐻,𝑊)
        #         # (number of data,
        #         # print(clip_for_display.unsqueeze(0).shape) -> torch.Size([1, 3, 64, 224, 224])
        #         clip_for_display = clip_for_display.unsqueeze(0).permute(0,2,1,3,4)
        #         if config.dataset == 'phav':
        #             for action_category in class_dict:
        #                 if targets.data[example_idx].cpu().numpy()[0] == action_category-1:
        #                     action_label = str(class_dict[action_category])
        #                     break
        #             summary_writer.add_video('train_clips/{:04d}'.format(example_idx) + action_label, clip_for_display, global_step)
        #         else:
        #             summary_writer.add_video('train_clips/{:04d}'.format(example_idx), clip_for_display,
        #                                      global_step)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc = np.mean(accuracies)

    # if summary_writer:
    #     summary_writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)
    #     summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)
    #     summary_writer.add_scalars('train_val/epoch_avg_loss', {'train': epoch_avg_loss}, epoch)
    #     summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'train': epoch_avg_acc}, epoch)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration, loss_list, step_list


####################################################################
####################################################################


def validation_fusion_epoch(config, models, fusion_model, criterion, device, data_loader, epoch, summary_writer=None):
    print('#' * 60)
    print('Epoch {}. Starting with validation phase.'.format(epoch + 1))

    fusion_model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()

    # assigning variables to divide videos into corresponding input modalities
    if config.rgb:
        rgb_ch_start = 0
    if config.depth:
        if config.rgb:
            depth_ch_start = 3
        else:
            depth_ch_start = 0
    if config.classgt:
        if config.rgb and config.depth:
            classgt_ch_start = 4
        elif config.rgb and not config.depth:
            classgt_ch_start = 3
        elif config.depth and not config.rgb:
            classgt_ch_start = 1
        else:
            classgt_ch_start = 0
    if config.inst_seg:
        if config.flow:
            inst_seg_ch_start = -3
        else:
            inst_seg_ch_start = -1
    if config.flow:
        flow_ch_start = -2

    softmax = torch.nn.Softmax(dim=1)

    for step, (clips, targets) in enumerate(data_loader):

        start_time = time.time()

        # Move inputs to GPU memory
        clips = clips.to(device)
        clip_list = []
        # seperate clips into corresponding slices according to model types
        if config.rgb:
            clip_rgb = clips[:, rgb_ch_start:rgb_ch_start + 3, :, :, :]
            clip_list.append(clip_rgb)
        if config.depth:
            clip_depth = clips[:, depth_ch_start:depth_ch_start + 1, :, :, :]
            clip_list.append(clip_depth)
        if config.classgt:
            clip_classgt = clips[:, classgt_ch_start:classgt_ch_start + 1, :, :, :]
            clip_list.append(clip_classgt)
        if config.inst_seg:
            if config.flow:
                clip_inst_seg = clips[:, inst_seg_ch_start:inst_seg_ch_start + 1, :, :, :]
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
            # normalize scores
            logits = softmax(logits)
            logit_list.append(logits)

        logits_concat = torch.cat(logit_list, 2)
        learned_logits_concat = fusion_model.forward(logits_concat)

        if config.fusion_layer_type == 'class':
            learned_logits_concat = torch.cat(learned_logits_concat, dim=1).unsqueeze(2)

        targets = targets.to(device).unsqueeze(-1)

        _, preds = torch.max(learned_logits_concat, 1)

        loss = criterion(learned_logits_concat, targets)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / config.batch_size

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size / float(time.time() - start_time)

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}/{}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.3f}, Loss = {:.3f}".format(
                datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                step, steps_in_epoch, examples_per_second,
                accuracies[step], losses[step]))

        # if summary_writer and step == 0:
        #     # TensorboardX video summary
        #     for example_idx in range(4):
        #         clip_for_display = clips[example_idx].clone().cpu()
        #         min_val = float(clip_for_display.min())
        #         max_val = float(clip_for_display.max())
        #         clip_for_display.clamp_(min=min_val, max=max_val)
        #         clip_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)
        #         clip_for_display = clip_for_display.unsqueeze(0).permute(0, 2, 1, 3, 4)
        #         if config.dataset == 'phav':
        #             for action_category in class_dict:
        #                 if targets.data[example_idx].cpu().numpy()[0] == action_category-1:
        #                     action_label = str(class_dict[action_category])
        #                     break
        #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx) + action_label, clip_for_display, epoch*steps_in_epoch)
        #         else:
        #             summary_writer.add_video('validation_clips/{:04d}'.format(example_idx), clip_for_display, epoch*steps_in_epoch)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('validation/epoch_avg_accuracy', epoch_avg_acc, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_loss', {'val': epoch_avg_loss}, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'val': epoch_avg_acc}, epoch)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration



# Proxy networks iterators

def train_proxy_epoch(config, model, criterion, optimizer, device, data_loader, epoch, summary_writer=None):
    print('#' * 60)
    print('Epoch {}. Starting with training phase.'.format(epoch + 1))

    model.train()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)
    epoch_start_time = time.time()

    for step, (inputs, targets) in enumerate(data_loader):

        # if step == 0:
        #     inputs = inputs1
        #     targets = targets1
        start_time = time.time()
        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        inputs = inputs.to(device)

        # Forward pass
        output = model.forward(inputs)

        # fig, ax = plt.subplots(1, 1)
        # img = inputs[0, :3, :, :].cpu().squeeze(0).permute(1,2,0)
        # ax.imshow(img)
        # plt.show()
        # img1 = inputs[0,:3,:,:]
        # img2 = inputs[0, 3:6, :, :]
        # flow_gt = np.asarray(targets[0, :, :, :].detach().cpu().permute(1,2,0))
        # flow_pred = np.asarray(output[0,:, :, :].detach().cpu().permute(1,2,0))
        #
        # flow_gt = torch.tensor(flow_to_color(flow_gt), dtype=torch.float32).permute(2,0,1).cuda(device)/255
        # flow_pred = torch.tensor(flow_to_color(flow_pred), dtype=torch.float32).permute(2,0,1).cuda(device)/255
        #
        # save_img = torchvision.utils.make_grid([img1, img2, flow_gt, flow_pred])
        # print('0', img1)
        # print(flow_gt)
        # print('1', flow_pred)
        # torchvision.utils.save_image(save_img, 'img.png')
        # sys.exit()

        targets = targets.to(device)
        if config.classgt:
            loss = criterion(output, targets.long().squeeze(1))
            accuracy = multiclass_IOU_v2(output, targets)
            accuracies[step] = accuracy
        elif config.inst_seg:
            # x, y_mask, y_class, sw_mask, sw_class = batch_to_var(inputs, targets, device)
            loss = criterion(output, targets, sw_mask)
            # loss_mask_iou = mask_siou(y_mask_perm.view(-1, y_mask_perm.size()[-1]),
            #                           out_masks.view(-1, out_masks.size()[-1]), sw_mask.view(-1, 1))
            # print('targets', np.unique(targets.detach().cpu().numpy()))
            # sys.exit()
        elif config.flow:
            loss = criterion(output, targets)
        else:
            # print('targets', np.unique(targets.detach().cpu().numpy()))
            # print('output', np.unique(output.detach().cpu().numpy()))
            # print('targets', targets.shape)
            # print('output', output.shape)
            loss = criterion(output, targets)

        # Calculate accuracy
        # correct = torch.sum(preds == targets.data)
        # accuracy = correct.double() / config.batch_size

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size / float(time.time() - start_time)

        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()

        # Save statistics
        losses[step] = loss.item()

        # Compute the global step, only for logging
        global_step = (epoch * steps_in_epoch) + step

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}/{}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "LR = {:.4f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                step, steps_in_epoch, examples_per_second,
                current_learning_rate(optimizer), accuracies[step], losses[step]))

        # TensorboardX Logging
        if summary_writer and step % config.log_frequency == 0:
            summary_writer.add_scalar('train/loss', losses[step], global_step)
            summary_writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
            summary_writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)
            if config.classgt or config.inst_seg:
                summary_writer.add_scalar('train/accuracy', accuracies[step], global_step)
                summary_writer.add_scalar('train/examples_per_second', examples_per_second, global_step)

        if summary_writer and step % config.log_image_frequency == 0:
            # TensorboardX video summary
            for example_idx in range(4):
                rgb_for_display = inputs[example_idx].clone().cpu()
                min_val = float(rgb_for_display.min())
                max_val = float(rgb_for_display.max())
                rgb_for_display.clamp_(min=min_val, max=max_val)
                rgb_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)

                output_for_display = output[example_idx].clone().cpu()
                gt_for_display = targets[example_idx].clone().cpu()

                if config.classgt:
                    output_for_display = torch.from_numpy(np.argmax(output_for_display.detach().numpy(), axis=0)).unsqueeze(0)

                if config.flow or config.depth:
                    min_val = float(output_for_display.min())
                    max_val = float(output_for_display.max())
                    output_for_display.clamp_(min=min_val, max=max_val)
                    output_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)

                    gt_for_display = targets[example_idx].clone().cpu()
                    min_val = float(gt_for_display.min())
                    max_val = float(gt_for_display.max())
                    gt_for_display.clamp_(min=min_val, max=max_val)
                    gt_for_display.add_(-min_val).div_(max_val - min_val + 1e-5)


                if config.flow:
                    summary_writer.add_image('train_rgb_f1/{:04d}'.format(example_idx), rgb_for_display[:3, :, :], global_step)
                    summary_writer.add_image('train_rgb_f2/{:04d}'.format(example_idx), rgb_for_display[3:6, :, :], global_step)
                    summary_writer.add_image('train_output_x/{:04d}'.format(example_idx), output_for_display[0, :, :].unsqueeze(0), global_step)
                    summary_writer.add_image('train_output_y/{:04d}'.format(example_idx), output_for_display[1, :, :].unsqueeze(0), global_step)
                    summary_writer.add_image('train_target_x/{:04d}'.format(example_idx),gt_for_display[0, :, :].unsqueeze(0), global_step)
                    summary_writer.add_image('train_target_y/{:04d}'.format(example_idx), gt_for_display[1, :, :].unsqueeze(0), global_step)
                if config.classgt:
                    summary_writer.add_image('train_rgb/{:04d}'.format(example_idx), rgb_for_display, global_step)
                    summary_writer.add_image('train_output_x/{:04d}'.format(example_idx), output_for_display, global_step)
                    summary_writer.add_image('train_target_x/{:04d}'.format(example_idx), gt_for_display, global_step)
                else:
                    summary_writer.add_image('train_rgb/{:04d}'.format(example_idx), rgb_for_display, global_step)
                    summary_writer.add_image('train_output_x/{:04d}'.format(example_idx), output_for_display, global_step)
                    summary_writer.add_image('train_target_x/{:04d}'.format(example_idx), gt_for_display, global_step)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_loss', {'train': epoch_avg_loss}, epoch)
        if config.classgt or config.inst_seg:
            summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)
            summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'train': epoch_avg_acc}, epoch)


    return epoch_avg_loss, epoch_avg_acc, epoch_duration


####################################################################
####################################################################


def validation_proxy_epoch(config, model, criterion, device, data_loader, epoch, summary_writer=None):
    print('#' * 60)
    print('Epoch {}. Starting with validation phase.'.format(epoch + 1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()

    data_list = []

    for step, (inputs, targets) in enumerate(data_loader):

        start_time = time.time()

        # Move inputs to GPU memory
        inputs = inputs.to(device)

        # Forward pass
        output = model.forward(inputs)
        targets = targets.to(device)

        if config.classgt:
            loss = criterion(output, targets.long().squeeze(1))
            accuracy = multiclass_IOU_v2(output, targets)
            accuracies[step] = accuracy
        elif config.inst_seg:
            loss = criterion(output, targets, 1)
        else:
            loss = criterion(output, targets)

        # Calculate accuracy
        # correct = torch.sum(preds == targets.data)
        # accuracy = correct.double() / config.batch_size

        data_list.append([np.asarray(targets.detach().cpu().flatten()), np.asarray(output.argmax(1).detach().cpu().flatten())])

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size / float(time.time() - start_time)

        # Save statistics
        # accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}/{}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.3f} , Loss = {:.3f}".format(
                datetime.now().strftime("%A %H:%M"), epoch, config.num_epochs,
                step, steps_in_epoch, examples_per_second,
                accuracies[step], losses[step]))

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc = np.mean(accuracies)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalars('train_val/epoch_avg_loss', {'val': epoch_avg_loss}, epoch)
        if config.classgt or config.inst_seg:
            summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)
            summary_writer.add_scalars('train_val/epoch_avg_accuracy', {'val': epoch_avg_acc}, epoch)

    # if config.classgt:
    #     ave_iou, per_class_iou = get_iou(data_list, class_num=65)
    #
    #     with open('colors_classidx.txt', 'r') as f:
    #         classes = f.readlines()
    #
    #     # for x in accs:
    #     classes = [x.split(' ')[0] for x in classes]
    #     # print(len(per_class_iou))
    #     # print(len(classes))
    #
    #     class_acc_dict = {}
    #
    #     for i in range(len(classes)):
    #         class_acc_dict[classes[i]] = per_class_iou[i]
    #
    #     print(class_acc_dict)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration