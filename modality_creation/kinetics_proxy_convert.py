from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from PIL import Image
import cv2
from torch.utils.data import DataLoader

from utils.utils import *
# from data.KINETICS.modality_pred.depth.models_resnet import Resnet50_md, Resnet18_md
# from data.KINETICS.modality_pred.flow.models import FlowNet2SD, FlowNet2CSS, FlowNet2
from torchvision import transforms
import matplotlib.pyplot as plt
import math
import numpy as np
from imageio import imread
from collections import OrderedDict
from checkpoints.deeplabv2 import *


# model definition
def get_model_and_data_loader(modality):
    if modality == 'depth':
        print('Loading depth estimation model')
        model = Resnet18_md(num_in_layers=3)
        checkpoint = torch.load('/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/modality_pred/depth/monodepth_resnet18_001.pth')
        model.load_state_dict(checkpoint)
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    elif modality == 'ss':
        model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print('Loading semantic segmentation model')
    elif modality == 'ss_stuff':
        # model = DeepLabV2(n_classes=67, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24])
        model = DeepLabV2(n_classes=182, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
        checkpoint = torch.load('/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/cocostuff/deeplabv2_resnet101_msc-cocostuff164k-100000.pth')
        checkpoint_new = OrderedDict()
        for key, value in checkpoint.items():
            # print(key,value)
            checkpoint_new[key[5:]] = value
        del checkpoint
        model.load_state_dict(checkpoint_new)
        # BGR_MEAN = (104.008, 116.669, 122.675)
        # preprocess = transforms.Compose([
        #     transforms.Resize((321, 321)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([122.675/255, 116.669/255, 104.008/255], std=[1,1,1])
        # ])
        preprocess = transforms.Compose([
            transforms.Resize((513, 513))])

    elif modality == 'is':
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        preprocess = transforms.ToTensor()

        print('Loading instance segmentation model')

    elif modality == 'flow':
        print('Loading flow model')
    
        model = FlowNet2(args).eval()
        # model = FlowNet2CSS(args).eval()
        # checkpoint = torch.load('/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/modality_pred/flow/FlowNet2-CSS-ft-sd_checkpoint.pth')
        checkpoint = torch.load('/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/modality_pred/flow/FlowNet2_checkpoint.pth')
        # print(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['state_dict'])
        preprocess = transforms.ToTensor()

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

def convert_rgb_to_is_single(modality, data_list_dir, subset):
    # retrieve model and preprocessing
    model, preprocess = get_model_and_data_loader(modality)
    # open directory
    if subset == 'training':
        mode = 'train_frames'
    elif subset == 'validation':
        mode = 'valid_frames'
    with open(data_list_dir) as f:
        lines = [line.rstrip('\n') for line in f]
        for vid in lines:
            frame_dir = frame_root_dir + mode + str(vid.split('.')[0])
            print(frame_dir)
            with open(frame_dir + '/n_frames') as f:
                n_frames = int(f.readlines()[0])
                for i in range(n_frames):
                    img_path = frame_dir + '/frame' + str(i) + '.jpg'
                    img_rgb = Image.open(img_path)
                    img = preprocess(img_rgb).cuda(device)
                    mask = np.transpose(get_mask_single(img, model, threshold=0.5), (1,2,0))

                    # instance_segmentation_api(img_path, model, preprocess)

                    mask = Image.fromarray(mask.astype('uint8'))
                    mask.save(frame_dir + '/instseg_' + str(i) + ".png", "PNG")
                    # sys.exit()
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

def convert_rgb_to_is_dataload(model, dataloader):
    for i, (img, path) in enumerate(dataloader):
        if i % 300 == 0:
            print(path[0])
        # instance_segmentation_api(img_path, model, preprocess)
        mask = np.transpose(get_mask_single(img.squeeze(0).cuda(device), model, threshold=0.5), (1,2,0))
        # img_path = path[0] + '/frame0.jpg'
        # print(img_path)
        # instance_segmentation_api(img_path, model, preprocess)

        mask = Image.fromarray(mask.astype('uint8'))
        # print(path[0].replace('frame', 'instseg_').replace('jpg', 'png'))
        mask.save(path[0].replace('frame', 'instseg_').replace('jpg', 'png'), "PNG")

def convert_rgb_to_depth(model, dataloader):
    for i, (img, path) in enumerate(dataloader):
        if i % 1000 == 0:
            print('{:.3f}% of frames processed!'.format(100 * i * batch_size / len(dataloader.dataset)))
            print(path[0])
        depth = model(img.cuda(device))
        # rgb_img = img[0].detach().cpu().numpy().transpose(1,2,0)
        depth = depth[0][:,0,:,:].detach().cpu()

        for j in range(batch_size):
            single_img = Image.fromarray(np.asarray(depth[j].detach().cpu()).astype('uint8'))
            single_i
            mg.save(path[j].replace('frame', 'depth_').replace('jpg', 'png'), "PNG", compress_level=1)

    print('DONE! Now go get em tiger :)')

def convert_rgb_to_ss(model, dataloader):
    for i, (img, path) in enumerate(dataloader):
        # start_time = time.time()
        if i % 500 == 0:
            print('{:.3f}% of frames processed!'.format(100*i*batch_size/len(dataloader.dataset)))
            print(path[0])
        img = model(img.cuda(device))
        img = img['out'].argmax(1)
        for j in range(batch_size):
            single_img = img[j]
            single_img = Image.fromarray(np.asarray(single_img.detach().cpu()).astype('uint8'))
            single_img.save(path[j].replace('frame', 'semseg_').replace('jpg', 'png'), "PNG", compress_level = 1)
        # print(float(time.time() - start_time))
        # sys.exit()
    print('DONE! Now go get em tiger :)')
# 1.1 vs 1.14 with save

def convert_rgb_to_ss_stuff(model, dataloader):
    with torch.no_grad():
        for i, (img, path) in enumerate(dataloader):
            # start_time = time.time()
            if i % 50 == 0:
                print('{:.3f}% of frames processed!'.format(100*i*batch_size/len(dataloader.dataset)))
                print(path[0])
            output = model(img.cuda(device))
            output = F.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
            output = output.argmax(1)
            for j in range(batch_size):
                single_output = output[j]
                single_output = Image.fromarray(np.asarray(single_output.detach().cpu()).astype('uint8'))

                # if i % 50 == 0:
                #     fig, ax = plt.subplots(1, 2)
                #     ax[0].imshow(np.array(img[j]/255).transpose(1,2,0))
                #     ax[1].imshow(single_output)
                #     plt.show()
                new_path = path[j].replace('/frame', '/semseg_stuff_').replace('jpg', 'png')
                single_output.save(new_path, "PNG", compress_level = 1)


        # print(float(time.time() - start_time))
        # sys.exit()
    print('DONE! Now go get em tiger :)')

def convert_rgb_to_flow(model, dataloader):
    data_length = len(dataloader.dataset)
    for step, (img, path) in enumerate(dataloader):
        # start_time = time.time()
        if step % 1000 == 0:
            print(path[0])
            print('{:.3f}% of frames processed!'.format(100*batch_size*step/data_length))
        flow = model(img.cuda(device))
        flow = flow + abs(flow.min())
        multiplier = 255/flow.max()
        flow = flow * multiplier
        # flow.shape = [b, 2, 256, 256]
        # flow_color = flow_vis.flow_to_color(np.array(flow[0].detach().cpu()).transpose(2,1,0), convert_to_bgr=False)
        # rgb_img = img[0,:,0,:,:].detach().cpu().numpy().transpose(1, 2, 0)
        # rgb_img = rgb_img.astype(np.int)
        # # Display the image
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(rgb_img)
        # ax[1].imshow(flow_color.transpose(1,0,2))
        # plt.show()

        for j in range(batch_size):
            flow_x = flow[j][0]
            flow_y = flow[j][1]
            flow_x = Image.fromarray(np.asarray(flow_x.detach().cpu()).astype(dtype='uint8'))
            flow_y = Image.fromarray(np.asarray(flow_y.detach().cpu()).astype(dtype='uint8'))
            flow_x.save(path[j].replace('frame', 'flow_x_').replace('jpg', 'png'), "PNG", compress_level = 1)
            flow_y.save(path[j].replace('frame', 'flow_y_').replace('jpg', 'png'), "PNG", compress_level = 1)

def get_mask_single(img, model, threshold=0.5):
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold]

    if len(pred_t) == 0:
        return np.zeros((3, img.size(1),img.size(2)))
    else:
        pred_t = pred_t[-1]
        masks = ""(pred[0]['masks']>0.5).squeeze(1).detach().cpu().numpy()
        class_labels = pred[0]['labels'].detach().cpu().numpy()
        masks = masks[:pred_t+1]
        # argmax here to obtain only masks in 1 channel with highest score first?
        class_labels = class_labels[:pred_t+1]
        unique = np.unique(class_labels)
        inst_class_label = {}
        for x in unique:
            inst_class_label[x] = 0

        mask_class_list = []
        for i in range(len(class_labels)):
            mask_class_list.append(np.where(masks[i]==True, class_labels[i], 0))

        mask_class_2d = mask_class_list[0]
        instance_mask = np.where(mask_class_2d != 0, 1, 0)
        inst_class_label[class_labels[0]] += 1
        for j in range(len(class_labels)-1):
            if inst_class_label[class_labels[j+1]] > 0:
                # class_label = [1 1 1 1 1 1 1 1 1 9 1 9 1 9 1 1]
                inst_class_label[class_labels[j+1]] += 1
                mask_class_2d = np.where(mask_class_2d == 0, mask_class_list[j+1], mask_class_2d)
                mask_class_list[j+1] = np.where(mask_class_list[j+1] != 0, inst_class_label[class_labels[j+1]], 0)
                instance_mask = np.where(instance_mask == 0, mask_class_list[j + 1], instance_mask)
            else:
                mask_class_2d = np.where(mask_class_2d == 0, mask_class_list[j + 1], mask_class_2d)
                mask_class_list[j+1] = np.where(mask_class_list[j+1] != 0, 1 , 0)
                instance_mask = np.where(instance_mask == 0, mask_class_list[j + 1], instance_mask)
                inst_class_label[class_labels[j + 1]] += 1

    mask_class_3ch = np.stack([mask_class_2d, instance_mask, np.zeros((img.size(1),img.size(2)))], axis=0)
    return mask_class_3ch

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

def instance_segmentation_api(img_path, model, transform, rect_th=1, text_size=1, text_th=1):
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

def convert_rgb_to_modality(modality, dataloader):
    if modality == 'depth':
        convert_rgb_to_depth(modality)

    elif modality == 'ss':
        convert_rgb_to_ss(modality)

    elif modality == 'ss_stuff':
        convert_rgb_to_ss(modality)

    elif modality == 'flow':
        print('Converting flow training files!')
        convert_rgb_to_flow(modality, train_list_dir, dataloader)
        print('Converting flow validation files!')
        convert_rgb_to_flow(modality, val_list_dir, dataloader)
    elif modality == 'is':
        # print('Converting training files!')
        # convert_rgb_to_is_single(modality, train_list_dir, subset = 'training')
        print('Converting validation files!')
        # convert_rgb_to_is_single(modality, val_list_dir, subset = 'validation')

def make_flow_dataset(root_path, data_list):
    dataset = []
    for i in range(len(data_list)):
        n_frames_file_path = os.path.join(root_path + data_list[i], 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if i % 100 == 0:
            print('Loading {} videos: '.format(i))

        for j in range(n_frames-1):
            rgb1 = os.path.join(root_path + data_list[i], "frame{}.jpg".format(j))
            rgb2 = os.path.join(root_path + data_list[i], "frame{}.jpg".format(j+1))
            rgb_frames = [rgb1, rgb2]
            dataset.append(rgb_frames)
    return dataset

def make_dataset(root_path, data_list):
    dataset = []
    for i in range(len(data_list)):
        n_frames_file_path = os.path.join(root_path + data_list[i], 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if i % 100 == 0:
            print('Loading {} videos: '.format(i))
        for j in range(n_frames):
            frame_path = os.path.join(root_path + data_list[i], "frame{}.jpg".format(j))
            dataset.append(frame_path)
    return dataset


class dataset():
    def __init__(self, data_list, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_list) as f:
            lines = [line.rstrip('\n') for line in f]
        self.data_list = lines
        self.transform = transform
        self.root_dir = root_dir

        self.dataset = make_dataset(root_dir, self.data_list)
        print('total frames: {}'.format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        image = transforms.ToTensor()(image)*255
        image = transforms.Normalize([122.675, 116.669, 104.008], std=[1,1,1])(image)

        image = image
        return image, image_path

class flow_dataset():
    def __init__(self, data_list, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(data_list) as f:
            lines = [line.rstrip('\n') for line in f]
        self.data_list = lines
        self.transform = transform
        self.root_dir = root_dir

        self.dataset = make_flow_dataset(root_dir, self.data_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path1 = self.dataset[idx][0]
        image_path2 = self.dataset[idx][1]
        image1 = imread(image_path1)
        image2 = imread(image_path2)
        images = [image1, image2]
        images = np.array(images).transpose(3,0,1,2)
        images = torch.from_numpy(images.astype(np.float32))
        images = nn.functional.interpolate(images, (256,256), mode='bilinear')
        # images = self.transform(images)

        return images, image_path1



modality = 'ss_stuff'
frame_root_dir = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/'
device = 'cuda:1'
data_list_dir =  '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/data_list_full.txt'
# train_list_dir = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/train_list_full_V2.txt'
train_list_dir = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/train_list_full_V2_copy.txt'
val_list_dir = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/val_list_full_V2.txt'
test_list_dir = '/home/m3kowal/Research/Datasets/KINETICS/kinetics100/test_list_full_V2.txt'
batch_size = 32
mode = 'train'
print('starting poop stuff')
# if modality =='flow':
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--start_epoch', type=int, default=1)
#     # parser.add_argument('--total_epochs', type=int, default=10000)
#     # parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
#     # parser.add_argument('--train_n_batches', type=int, default = -1, help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
#     parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
#     # parser.add_argument('--gradient_clip', type=float, default=None)
#     # parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
#     # parser.add_argument('--schedule_lr_fraction', type=float, default=10)
#     parser.add_argument("--rgb_max", type=float, default = 255.)
#
#     parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=4)
#     parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
#     parser.add_argument('--no_cuda', action='store_true')
#
#     parser.add_argument('--seed', type=int, default=1)
#     # parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
#     # parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')
#
#     # parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
#     # parser.add_argument('--validation_n_batches', type=int, default=-1)
#     # parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')
#
#     # parser.add_argument('--inference', action='store_true')
#     parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
#     parser.add_argument('--inference_batch_size', type=int, default=1)
#     # parser.add_argument('--inference_n_batches', type=int, default=-1)
#     # parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')
#
#     parser.add_argument('--resume', default='/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/modality_pred/flow/FlowNet2-CSS-ft-sd_checkpoint.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#     parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")
#
#     parser.add_argument('--skip_training', action='store_true')
#     parser.add_argument('--skip_validation', action='store_true')
#
#     parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
#     parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
#     parser.add_argument('--mode', type=str, default='')


    # args = parser.parse_args()
# convert_rgb_to_modality(modality, dataloader)

model, preprocess = get_model_and_data_loader(modality)

if mode == 'train':
    transformed_dataset = dataset(data_list=train_list_dir,
                                           root_dir='/home/m3kowal/Research/Datasets/KINETICS/kinetics400/train_frames',
                                           transform=preprocess)
elif mode == 'valid':
    transformed_dataset = dataset(data_list=val_list_dir,
                                           root_dir='/home/m3kowal/Research/Datasets/KINETICS/kinetics400/valid_frames',
                                           transform=preprocess)
elif mode == 'test':
    transformed_dataset = dataset(data_list=test_list_dir,
                                           root_dir='/home/m3kowal/Research/Datasets/KINETICS/kinetics400/test_frames',
                                           transform=preprocess)

# elif mode == 'full':
#     transformed_dataset = dataset(data_list=data_list_dir,
#                                            root_dir='/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/kinetics100/valid',
#                                            transform=preprocess)

# else:
#     if mode == 'train':
#         transformed_dataset = dataset(data_list=train_list_dir,
#                                                root_dir='/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/kinetics100/train',
#                                                transform=preprocess)
#     elif mode == 'valid':
#         transformed_dataset = dataset(data_list=val_list_dir,
#                                                root_dir='/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/KINETICS/kinetics100/valid',
#                                                transform=preprocess)



dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

convert_rgb_to_ss_stuff(model, dataloader)
# convert_rgb_to_flow(model, dataloader)

# if __name__== "__main__":
#     modality = sys.argv[1]
#     frame_dir = sys.argv[2]
#     dataset = sys.argv[3]  # (ucf | phav | kinetics)
