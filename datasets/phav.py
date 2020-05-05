# Source: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/mean.py

import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import math
import functools
import json
import copy
import random

from utils.utils import load_value_file
import sys
import numpy as np
#config = parse_opts()
##########################################################################################
##########################################################################################
class_dict_ss={
0:'Terrain',
1:'Sky',
2:'Tree',
3:'Vegetation',
4:'Building',
5:'Road',
6:'TrafficSign',
7:'TrafficLight',
8:'Pole',
9:'Misc',
10:'Truck',
11:'Car',
12:'Bus',
13:'Human',
14:'Window',
15:'Door',
16:'Sofa',
17:'Ceiling',
18:'Chair',
19:'Floor',
20:'Table',
21:'Curtain',
22:'Bed',
23:'Fireplace',
24:'Shelf',
25:'Lamp',
26:'Stair',
27:'Bench',
28:'Screen',
29:'Fridge',
30:'Ball',
31:'BaseballBat',
32:'Bow',
33:'Gun',
34:'GolfClub',
35:'HairBrush',
36:'Head',
37:'RightUpperArm',
38:'RightLowerArm',
39:'RightHand',
40:'LeftUpperArm',
41:'LeftLowerArm',
42:'LeftHand',
43:'Chest',
44:'RightUpperLeg',
45:'RightLowerLeg',
46:'RightFoot',
47:'LeftUpperLeg',
48:'LeftLowerLeg',
49:'LeftFoot',
50:'Neck',
51:'LeftShoulder',
52:'RightShoulder',
53:'LeftElbow',
54:'RightElbow',
55:'LeftWrist',
56:'RightWrist',
57:'LeftHip',
58:'RightHip',
59:'LeftKnee',
60:'RightKnee',
61:'LeftAnkle',
62:'RightAnkle',
}

def pil_loader(path, one_channel=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if one_channel:
                return img.convert('L')
            else:
                return img.convert('RGB')

def pil_proxy_loader(path, one_channel=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if one_channel:
                return img.convert('I;16')
            else:
                return img.convert('RGB')

def pil_rgb_loader(path, one_channel=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_proxy_loader

def get_default_image_loader2():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader()

def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path, False))
        else:
            return video
    return video

def modality_loader(video_dir_path, frame_indices, modality, image_loader):
    video = []
    if modality == 'rgb':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, False))
            else:
                return video
    elif modality == 'depth':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'depth_{:05d}.png'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, True))
            else:
                return video
    elif modality == 'classgt':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'classgt_{:05d}.png'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, True))
            else:
                return video
    elif modality == 'flow_x':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'flow_x_{:05d}.jpg'.format(i))
            if os.path.exists(image_path) and os.path.exists(image_path):
                video.append(pil_loader(image_path, True))
            else:
                return video
    elif modality == 'flow_y':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'flow_y_{:05d}.jpg'.format(i))
            if os.path.exists(image_path) and os.path.exists(image_path):
                video.append(pil_loader(image_path, True))
            else:
                return video
    elif modality == 'instgt':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'instancegt_{:05d}.png'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, True))
            else:
                return video
    return video


def proxy_loader(video_dir_path, index, modality, image_loader):
    if modality == 'rgb':
        image_path = os.path.join(video_dir_path, 'img_{:05d}.jpg'.format(index))
        if os.path.exists(image_path):
            image = image_loader(image_path)
            return image
        else:
            print('Doesnt exist!')
    if modality == 'depth':
        image_path = os.path.join(video_dir_path, 'depth_{:05d}.png'.format(index))
        if os.path.exists(image_path):
            image = image_loader(image_path, True)
            return image
        else:
            print('Doesnt exist!')
    elif modality == 'classgt':
        image_path = os.path.join(video_dir_path, 'proxy_classgt{:05d}.png'.format(index))
        if os.path.exists(image_path):
            image = image_loader(image_path, True)
            return image
        else:
            print('Doesnt exist!')
    elif modality == 'flow':
        flow = []
        image_pathx = os.path.join(video_dir_path, 'flow_x_{:05d}.jpg'.format(index))
        image_pathy = os.path.join(video_dir_path, 'flow_y_{:05d}.jpg'.format(index))
        if os.path.exists(image_pathx) and os.path.exists(image_pathy):
            imagex = pil_loader(image_pathx, True)
            imagey = pil_loader(image_pathy, True)
            flow.append(imagex)
            flow.append(imagey)
            return flow
        else:
            print('doesnt exist!!!')
    elif modality == 'instgt':
        image_path = os.path.join(video_dir_path, 'instancegt_{:05d}.png'.format(index))
        if os.path.exists(image_path):
            image = image_loader(image_path, False)
            return image
        else:
            print('Doesnt exist!')


def proxy_modality_loader(video_dir_path, frame_indices, modality, image_loader):
    video = []
    if modality == 'depth':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'depth_{:05d}.png'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, True))
            else:
                return video
    elif modality == 'classgt':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'classgt_pred_{:04d}.png'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, True))
            else:
                return video
    elif modality == 'flow_x':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'flow_x_pred_{:05d}.png'.format(i))
            if os.path.exists(image_path) and os.path.exists(image_path):
                video.append(pil_loader(image_path, True))
            else:
                return video
    elif modality == 'flow_y':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'flow_y_pred_{:05d}.png'.format(i))
            
            if os.path.exists(image_path) and os.path.exists(image_path):
                video.append(pil_loader(image_path, True))
            else:
                return video
    elif modality == 'instgt':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'instancegt_{:05d}.png'.format(i))
            if os.path.exists(image_path):
                video.append(image_loader(image_path, True))
            else:
                return video
    return video

def get_proxy_loader():
    image_loader = get_default_image_loader()
    return functools.partial(proxy_loader, image_loader=image_loader)

def get_proxy_loader2():
    image_loader = get_default_image_loader2()
    return functools.partial(proxy_loader, image_loader=image_loader)

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(modality_loader, image_loader=image_loader)

def get_default_proxy_video_loader():
    image_loader = get_default_image_loader()
    #return functools.partial(video_loader, image_loader=image_loader)
    return functools.partial(proxy_modality_loader, image_loader=image_loader)

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations

def get_modality_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations

##########################################################################################
##########################################################################################

def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):

    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)

    if not annotations:
        raise ValueError('Unable to load annotations...')

    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('Loading PHAV videos [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        if not os.path.exists(n_frames_file_path):
           raise FileNotFoundError('n_frames_file_path does not exist: {}'.format(n_frames_file_path))

        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class

def make_modality_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):

    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('Loading PHAV modality videos [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])

        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        if not os.path.exists(n_frames_file_path):
           raise FileNotFoundError('n_frames_file_path does not exist: {}'.format(n_frames_file_path))

        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        sample = {
            'video': video_path,
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset


##########################################################################################
##########################################################################################

class PHAV(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 config,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        self.config = config
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if len(frame_indices) < self.config.sample_duration+1:
            del frame_indices[-1]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if not self.config.multi_modal:
            clips = []
            if self.config.rgb:
                clip = self.loader(path, frame_indices, modality='rgb')
                clips.append(clip)
            if self.config.depth:
                clip = self.loader(path, frame_indices, modality='depth')
                clips.append(clip)
            if self.config.classgt:
                clip = self.loader(path, frame_indices, modality='classgt')
                clips.append(clip)
            if self.config.inst_seg:
                clip = self.loader(path, frame_indices, modality='instgt')
                clips.append(clip)
            if self.config.flow:
                clip = self.loader(path, frame_indices, modality='flow_x')
                clips.append(clip)
                clip = self.loader(path, frame_indices, modality='flow_y')
                clips.append(clip)
            # if self.config.joints:
            #     clip = self.loader(path, frame_indices, modality = 'joints')

            transformed_clips = []
            # clips [ [rgb_clip], [depth clip]... [flow clip] ]
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                for clip in clips:
                    transformed_clip = [self.spatial_transform(img) for img in clip]
                    transformed_clips.append(transformed_clip)

            concat_clip = []
            for clip in transformed_clips:
                clips = torch.stack(clip, 0).permute(1, 0, 2, 3)
                concat_clip.append(clips)

            concat_clip = torch.cat(concat_clip, 0)
            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return concat_clip, target

        else:
            clip = self.loader(path, frame_indices, modality='rgb')

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            target = self.data[index]

            if self.target_transform is not None:
                target = self.target_transform(target)

            return clip, target


    def __len__(self):
        return len(self.data)

class PHAV_modalities2(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 config,
                 root_path,
                 annotation_path,
                 subset,
                 modality,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_proxy_loader):

        self.data = make_modality_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.config = config
        self.spatial_transform = spatial_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.modality = modality


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """


        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']

        rand_index = frame_indices[random.randrange(1, len(frame_indices)-1)]

        # grab input image
        input = self.loader(path, rand_index, modality='rgb')
        target = self.loader(path, rand_index, self.modality)

        print(target.mode)
        fig, ax = plt.subplots(1, 2)
        example1 = input
        ax[0].imshow(example1)

        example2 = target
        ax[1].imshow(example2)

        plt.show()
        sys.exit()
        # print('pre', np.unique(np.asarray(target1)))

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            input = self.spatial_transform(input)
            if self.modality == 'flow':
                input_prev = self.loader(path, rand_index-1, modality='rgb')
                input_prev = self.spatial_transform(input_prev)
                input = torch.cat((input_prev, input), 0)

                target = [self.spatial_transform(img, norm=False) for img in target]
                target = torch.cat((target[0], target[1]), 0)
            elif self.modality == 'depth':
                target = Image.fromarray((np.array(target) / 16).astype('uint32')).convert('L')
                target = self.spatial_transform(target, norm=False)
                target = target / 255
            else:
                target = self.spatial_transform(target, norm=False)

        #
        # fig, ax = plt.subplots(1, 2)
        # example1 = np.asarray(input.permute(1,2,0))
        # ax[0].imshow(example1)
        #
        # example2 = np.asarray(target.detach().cpu().squeeze(0))
        # ax[1].imshow(example2)
        #
        # plt.show()

        return input, target

    def __len__(self):
        return len(self.data)


class PHAV_modalities(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 config,
                 root_path,
                 annotation_path,
                 subset,
                 modality,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_proxy_loader):

        self.data = make_modality_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.config = config
        self.spatial_transform = spatial_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.modality = modality


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """


        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']

        rand_index = frame_indices[random.randrange(1, len(frame_indices)-1)]

        # grab input image
        input = self.loader(path, rand_index, modality='rgb')



        target = self.loader(path, rand_index, self.modality)
        # fig, ax = plt.subplots(1, 2)
        # example1 = input
        # ax[0].imshow(example1)
        #
        # example2 = target
        # ax[1].imshow(example2)
        #
        # plt.show()

        # print('pre', np.unique(np.asarray(target1)))

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            input = self.spatial_transform(input)
            if self.modality == 'flow':
                input_prev = self.loader(path, rand_index-1, modality='rgb')
                input_prev = self.spatial_transform(input_prev)
                input = torch.cat((input_prev, input), 0)
                target = [self.spatial_transform(img, norm=False) for img in target]
                target = torch.cat((target[0], target[1]), 0)
            elif self.modality == 'depth':
                target = Image.fromarray((np.array(target) / 16).astype('uint32')).convert('L')
                target = self.spatial_transform(target, norm=False)
                target = target / 255
            else:
                target = self.spatial_transform(target, norm=False)

        return input, target

    def __len__(self):
        return len(self.data)

class PHAV_proxy(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 config,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_proxy_video_loader):

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        self.config = config
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if len(frame_indices) < self.config.sample_duration + 1:
            del frame_indices[-1]

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if self.config.proxy:
            clips = []
            if self.config.depth:
                clip = self.loader(path, frame_indices, modality='depth')
                clips.append(clip)
            if self.config.classgt:
                clip = self.loader(path, frame_indices, modality='classgt')
                clips.append(clip)
            if self.config.inst_seg:
                clip = self.loader(path, frame_indices, modality='instgt')
                clips.append(clip)
            if self.config.flow:
                clip = self.loader(path, frame_indices, modality='flow_x')
                clips.append(clip)
                clip = self.loader(path, frame_indices, modality='flow_y')
                clips.append(clip)
            # if self.config.joints:
            #     clip = self.loader(path, frame_indices, modality = 'joints')

            transformed_clips = []
            # clips [ [rgb_clip], [depth clip]... [flow clip] ]
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                for clip in clips:
                    transformed_clip = [self.spatial_transform(img) for img in clip]
                    transformed_clips.append(transformed_clip)

            concat_clip = []
            for clip in transformed_clips:
                clips = torch.stack(clip, 0).permute(1, 0, 2, 3)
                concat_clip.append(clips)

            concat_clip = torch.cat(concat_clip, 0)
            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return concat_clip, target

        else:
            clip = self.loader(path, frame_indices, modality='rgb')

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            target = self.data[index]

            if self.target_transform is not None:
                target = self.target_transform(target)


            return clip, target

    def __len__(self):
        return len(self.data)
