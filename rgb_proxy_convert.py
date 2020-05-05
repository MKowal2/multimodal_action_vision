from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime
import random

import torch.nn as nn
import torch
import torchvision as tv
from torchvision import models
from PIL import Image
import cv2

from utils.utils import *
from data.modality_pred.depth.models_resnet import Resnet50_md, Resnet18_md
from proxy_convert.models.models import ModelBuilder, SegmentationModule
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math


modality = 'is'
frame_root_dir = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/UCF-101_rgbflow/rgb_modalities/jpg/'
dataset = 'ucf'
device = 'cuda:0'
data_list_dir = '/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/UCF-101_rgbflow/rgb_modalities/ucfTrainTestlist/data_list_frontcrawl.txt'
batch_size = 10

totensor = tv.transforms.ToTensor()
# model definition
def get_model_and_data_loader(modality):
    if modality == 'depth':
        model = Resnet18_md()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print('Loading depth estimation model')
    elif modality == 'ss':
        model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print('Loading semantic segmentation model')

    elif modality == 'is_things':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        preprocess = transforms.ToTensor()
        print('Loading instance segmentation model')

    elif modality == 'ss_stuff':

        preprocess = transforms.ToTensor()

        print('Loading instance segmentation model')

    # move to GPU and evaluation mode
    model.eval().cuda(device)
    return model, preprocess


def convert_rgb_to_is(modality):
    # retrieve model and preprocessing
    model, preprocess = get_model_and_data_loader(modality)
    # open directory
    with open(data_list_dir) as f:
        lines = [line.rstrip('\n') for line in f]
        # print(lines[0].split('.')[0]) # /home/m3kowal/Research/vfhlt/PyTorchConv3D/data/UCF-101_rgbflow/rgb_modalities/jpg/YoYo/v_YoYo_g25_c05.avi
        for vid in lines:
            frame_dir = os.path.join(frame_root_dir, vid.split('.')[0])
            print(frame_dir)
            with open(frame_dir+ '/n_frames') as f:
                n_frames = int(f.readlines()[0])
                num_batches = math.floor(n_frames/batch_size)
                last_batch = n_frames % batch_size
                idx_counter = 1
                for b in range(0,num_batches):
                    img_list = []
                    img_name_list = []
                    for i in range(0, batch_size):
                        img_path = frame_dir + '/image_' + str(idx_counter).zfill(5) + '.jpg'
                        img_rgb = Image.open(img_path)
                        img = preprocess(img_rgb).cuda(device)
                        img_list.append(img)
                        img_name_list.append('instseg_'+ str(idx_counter).zfill(5))
                        idx_counter += 1

                    mask = get_mask(img_list, model, threshold=0.7)
                    for k in range(len(mask)):
                        # mask.shape = (240, 320)
                        inst_seg_mask = Image.fromarray(mask[k].astype('uint8'))
                        inst_seg_mask.save(frame_dir + '/' + img_name_list[k] + ".png", "PNG")

                    # fig, ax = plt.subplots(1, 5)
                    # example1 = img_rgb
                    # ax[0].imshow(example1)
                    # ax[1].imshow(mask[0])
                    # ax[2].imshow(mask[1])
                    # ax[3].imshow(mask[2])
                    # ax[4].imshow(mask[3])
                    # plt.show()
                    # sys.exit()

                if last_batch > 0:
                    img_list = []
                    img_name_list = []
                    for j in range(0, last_batch):
                        img_path = frame_dir + '/image_' + str(idx_counter).zfill(5) + '.jpg'
                        img_rgb = Image.open(img_path)
                        img = preprocess(img_rgb).cuda(device)
                        img_list.append(img)
                        img_name_list.append('instseg_' + str(idx_counter).zfill(5))
                        idx_counter += 1

                    mask = get_mask(img_list, model, threshold=0.7)

                    for k in range(len(mask)):
                        inst_seg_mask = Image.fromarray(mask[k].astype('uint8'))
                        inst_seg_mask.save(frame_dir + '/' + img_name_list[k] + ".png", "PNG")


def convert_rgb_to_is_single(modality):
    # retrieve model and preprocessing
    model, preprocess = get_model_and_data_loader(modality)
    # open directory
    with open(data_list_dir) as f:
        lines = [line.rstrip('\n') for line in f]
        # print(lines[0].split('.')[0]) # /home/m3kowal/Research/vfhlt/PyTorchConv3D/data/UCF-101_rgbflow/rgb_modalities/jpg/YoYo/v_YoYo_g25_c05.avi
        for vid in lines:
            frame_dir = os.path.join(frame_root_dir, vid.split('.')[0])
            print(frame_dir)
            with open(frame_dir+ '/n_frames') as f:
                n_frames = int(f.readlines()[0])
                for i in range(n_frames):
                    img_path = frame_dir + '/image_' + str(i+1).zfill(5) + '.jpg'
                    img_rgb = Image.open(img_path)
                    img = preprocess(img_rgb).cuda(device)
                    mask = get_mask_single(img, model, threshold=0.7)
                    inst_seg_mask = Image.fromarray(mask.astype('uint8'))
                    inst_seg_mask.save(frame_dir + '/instseg_' + str(i+1).zfill(5) + ".png", "PNG")


                    # fig, ax = plt.subplots(1, 2)
                    # example1 = img_rgb
                    # ax[0].imshow(example1)
                    # ax[1].imshow(mask)
                    # plt.show()
                    # sys.exit()

                    #
                    # img_list = []
                    # img_name_list = []
                    # for i in range(0, batch_size):
                    #     img_path = frame_dir + '/image_' + str(idx_counter).zfill(5) + '.jpg'
                    #     img_rgb = Image.open(img_path)
                    #     img = preprocess(img_rgb).cuda(device)
                    #     img_list.append(img)
                    #     img_name_list.append('instseg_'+ str(idx_counter).zfill(5))
                    #     idx_counter += 1
                    #
                    # mask = get_mask(img_list, model, threshold=0.7)
                    # for k in range(len(mask)):
                    #     # mask.shape = (240, 320)
                    #     inst_seg_mask = Image.fromarray(mask[k].astype('uint8'))
                    #     inst_seg_mask.save(frame_dir + '/' + img_name_list[k] + ".png", "PNG")

                    # fig, ax = plt.subplots(1, 5)
                    # example1 = img_rgb
                    # ax[0].imshow(example1)
                    # ax[1].imshow(mask[0])
                    # ax[2].imshow(mask[1])
                    # ax[3].imshow(mask[2])
                    # ax[4].imshow(mask[3])
                    # plt.show()
                    # sys.exit()


def convert_rgb_to_depth(modality):
    # retrieve model and preprocessing
    model, preprocess = get_model_and_data_loader(modality)
    # open directory
    i = 0
    for root, _, files in os.walk(frame_dir):
        # print(root) -> /home/m3kowal/Research/vfhlt/PyTorchConv3D/data/UCF-101/jpg/Swing/v_Swing_g22_c05

        for file in files:
            if file[-4:] == '.jpg':
                file_dir = os.path.join(root,file)
                # with open(file_dir, 'rb') as f:
                img_rgb = Image.open(file_dir)
                img = preprocess(img_rgb).unsqueeze(0)
                print(img.shape)
                img = model(img)['out'][0].argmax(0)

                if i > 2:
                    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
                    colors = (colors % 255).numpy().astype("uint8")

                    # plot the semantic segmentation predictions of 21 classes in each color
                    r = Image.fromarray(img.byte().cpu().numpy()).resize(img_rgb.size)
                    r.putpalette(colors)

                    fig, ax = plt.subplots(1, 2)
                    example1 = img_rgb
                    ax[0].imshow(example1)
                    example2 = np.asarray(img.detach())
                    ax[1].imshow(r)
                    plt.show()

                print(file[-9:])
        #     if its a jpg:
        #         output = model(file)
        #         save_as_png(root, str(modality) + file[-5:])
        i += 1
        if i == 25:
            sys.exit()

def convert_rgb_to_ss(modality):
    # retrieve model and preprocessing
    model, preprocess = get_model_and_data_loader(modality)
    # open directory
    for frame_dir_misising in ss_incomplete:
        for root, _, files in os.walk(frame_dir_misising):
            # print(root) -> /home/m3kowal/Research/vfhlt/PyTorchConv3D/data/UCF-101/jpg/Swing/v_Swing_g22_c05
            print(root)
            for file in files:
                if file[-4:] == '.jpg':
                    file_dir = os.path.join(root,file)
                    # with open(file_dir, 'rb') as f:
                    img_rgb = Image.open(file_dir)
                    img = preprocess(img_rgb).unsqueeze(0).cuda(device)
                    # img = model(img)
                    img = model(img)['out'][0].argmax(0)

                    # threshold AND THEN argmax!

                    img = Image.fromarray(np.asarray(img.detach().cpu()).astype('uint8'))
                    img.save(str(root + '/' + 'semseg' + file[5:-4]) + ".png", "PNG")


def get_mask_single(img, model, threshold=0.5):
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold]

    if len(pred_t) == 0:
        mask_class_2d = np.zeros((240,320))

    else:
        pred_t = pred_t[-1]
        masks = (pred[0]['masks']>0.5).squeeze(1).detach().cpu().numpy()
        class_labels = pred[0]['labels'].detach().cpu().numpy()
        masks = masks[:pred_t+1]
        # argmax here to obtain only masks in 1 channel with highest score first?
        class_labels = class_labels[:pred_t+1]
        mask_class_list = []
        for i in range(len(class_labels)):
            mask_class_list.append(np.where(masks[i]==True, class_labels[i], 0))

        mask_class_2d = mask_class_list[0]
        if pred_t > 0:
            for j in range(len(class_labels)-1):
                mask_class_2d = np.where(mask_class_2d == 0, mask_class_list[j+1], mask_class_2d)

    return mask_class_2d

def get_mask(img, model, threshold=0.5):
    # img = Image.open(img_path)
    # img = transform(img).cuda(device)
    mask_final_list = []
    pred_overall = model(img)
    for i in range(len(img)):
        pred = pred_overall[i]
        pred_score = list(pred['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold]

        if len(pred_t) == 0:
            mask_final_list.append(np.zeros((240,320)))

        else:
            pred_t = pred_t[-1]
            masks = (pred['masks']>0.5).squeeze(1).detach().cpu().numpy()
            class_labels = pred['labels'].detach().cpu().numpy()
            masks = masks[:pred_t+1]
            # argmax here to obtain only masks in 1 channel with highest score first?
            class_labels = class_labels[:pred_t+1]
            mask_class_list = []
            for i in range(len(class_labels)):
                mask_class_list.append(np.where(masks[i]==True, class_labels[i], 0))

            mask_class_2d = mask_class_list[0]
            if pred_t > 0:
                for j in range(len(class_labels)-1):
                    mask_class_2d = np.where(mask_class_2d == 0, mask_class_list[j+1], mask_class_2d)

            mask_final_list.append(mask_class_2d)

    return mask_final_list

def get_prediction(img_path, model, transform, threshold=0.5):
    img = Image.open(img_path)
    img = transform(img).cuda(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    class_labels = pred[0]['labels'].detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES2[i] for i in list(class_labels)]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1] # keep the top pred_t masks
    pred_class = pred_class[:pred_t+1] # keep the top pred_t class predictions
    return masks, pred_boxes, pred_class

def random_colour_masks(image):
  colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
  coloured_mask = np.stack([r, g, b], axis=2)
  return coloured_mask

def instance_segmentation_api(img_path, model, transform, rect_th=3, text_size=3, text_th=3):
  masks, boxes, pred_cls = get_prediction(img_path, model, transform)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  for i in range(len(masks)):
    rgb_mask = random_colour_masks(masks[i])
    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
  plt.figure(figsize=(20,30))
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

COCO_INSTANCE_CATEGORY_NAMES2 = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parkmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def convert_rgb_to_modality(modality):
    if modality == 'depth':
        convert_rgb_to_depth(modality)

    elif modality == 'ss':
        convert_rgb_to_ss(modality)

    elif modality == 'is':
        convert_rgb_to_is_single(modality)

convert_rgb_to_modality(modality)
#
# if __name__== "__main__":
#     modality = sys.argv[1]
#     frame_dir = sys.argv[2]
#     dataset = sys.argv[3]  # (ucf | phav | kinetics)
