import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import math
import functools
import json
import copy
import sys
from utils.utils import load_value_file


def rgb_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def png_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def rgb_vid_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'frame{}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def flow_vid_loader(video_dir_path, frame_indices, image_loader, axis):
    video = []
    for i in frame_indices:
        image_path= os.path.join(video_dir_path, 'flow_' + axis + '_{}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def depth_vid_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'depth_{}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def ss_vid_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'semseg_{}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def ss_stuff_vid_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'semseg_stuff_{}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print('Doesnt exist!', image_path)
            return video
    return video

def is_vid_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'instseg_{}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def inst_person_vid_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'instseg_{}.png'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video

def get_default_video_loader(config):
    if not config.multi_modal:
        image_loader = rgb_loader
        return functools.partial(rgb_vid_loader, image_loader=image_loader)
    else:
        if config.inst_person:
            image_loader = rgb_loader
            return functools.partial(inst_person_vid_loader, image_loader=image_loader)
        image_loader = png_loader
        if config.flow:
            return functools.partial(flow_vid_loader, image_loader=image_loader)
        elif config.depth:
            return functools.partial(depth_vid_loader, image_loader=image_loader)
        elif config.classgt:
            return functools.partial(ss_vid_loader, image_loader=image_loader)
        elif config.classgt_stuff:
            return functools.partial(ss_stuff_vid_loader, image_loader=image_loader)
        elif config.inst_seg:
            return functools.partial(is_vid_loader, image_loader=rgb_loader)

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_label = class_label.replace(" ", "_")
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map

def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []
    for key, value in data['database'].items():
        # value['annotations']['label'] = value['annotations']['label'].replace(" ", "_")
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'test':
                video_names.append('test_frames/{}/{}'.format(value['annotation']['label'], key))
                annotations.append(value['annotation'])
            else:
                label = value['annotation']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotation'])
                annotations = annotations
    return video_names, annotations

def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    if subset == 'training':
        print('Loading Training Set!')
        root_path = root_path + '/train_frames'
    elif subset == 'validation':
        print('Loading Validation Set!')
        root_path = root_path + '/valid_frames'

    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('Found [{}/{}] videos'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            print('VIDEO NOT FOUND!!!')
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i][:-14].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(0, n_frames))
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

class Kinetics(data.Dataset):
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
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader(config)
        self.config = config

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        # need to add flow indices handling

        frame_indices = self.data[index]['frame_indices']
        if len(frame_indices) < self.config.sample_duration+1:
            del frame_indices[-1]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        if self.config.flow:
            self.spatial_transform.randomize_parameters()
            clip_x = self.loader(path, frame_indices, axis='x')
            clip_y = self.loader(path, frame_indices, axis='y')
            clip_x = [self.spatial_transform(img) for img in clip_x]
            clip_y = [self.spatial_transform(img) for img in clip_y]
            clip_x = torch.stack(clip_x, 0).permute(1, 0, 2, 3)
            clip_y = torch.stack(clip_y, 0).permute(1, 0, 2, 3)
            clip = torch.cat([clip_x, clip_y], dim= 0)

            target = self.data[index]
            target = self.target_transform(target)
            return clip, target
        else:
            clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        if self.config.inst_person:
            human_mask = torch.where(clip[0]*255 == 1, torch.tensor(1), torch.tensor(0)) # bxhxw
            clip = torch.where(human_mask == 1, clip[1], torch.tensor(0.0)).unsqueeze(0)

        if self.config.only_person:
            person = torch.tensor(0.0588)
            background = torch.tensor(0.0)
            clip = torch.where(clip > person, person, background)
            clip = torch.where(clip < person, person, background)

        if self.config.inst_seg:
            clip = clip[:2,:,:,:]
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)

class Kinetics_fusion(data.Dataset):
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
                 sample_duration=16):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.config = config

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        # need to add flow indices handling
        frame_indices = self.data[index]['frame_indices']
        if len(frame_indices) < self.config.sample_duration+1:
            del frame_indices[-1]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()

        # order = [RGB, Flow, Depth, SS, IS]
        clips = []
        if self.config.rgb:
            # load RGB
            rgb_clip = rgb_vid_loader(path, frame_indices, rgb_loader)
            rgb_clip = [self.spatial_transform(img) for img in rgb_clip]
            rgb_clip = torch.stack(rgb_clip, 0).permute(1, 0, 2, 3)
            clips.append(rgb_clip)

        if self.config.flow:
            # Load flow
            self.spatial_transform.randomize_parameters()
            clip_x = flow_vid_loader(path, frame_indices, png_loader, axis='x')
            clip_y = flow_vid_loader(path, frame_indices, png_loader, axis='y')
            clip_x = [self.spatial_transform(img) for img in clip_x]
            clip_y = [self.spatial_transform(img) for img in clip_y]
            clip_x = torch.stack(clip_x, 0).permute(1, 0, 2, 3)
            clip_y = torch.stack(clip_y, 0).permute(1, 0, 2, 3)
            flow_clip = torch.cat([clip_x, clip_y], dim= 0)
            clips.append(flow_clip)

        if self.config.depth:
            #load depth
            depth_clip = depth_vid_loader(path, frame_indices, png_loader)
            depth_clip = [self.spatial_transform(img) for img in depth_clip]
            depth_clip = torch.stack(depth_clip, 0).permute(1, 0, 2, 3)
            clips.append(depth_clip)

        if self.config.classgt:
            # load classgt
            classgt_clip = ss_vid_loader(path, frame_indices, png_loader)
            classgt_clip = [self.spatial_transform(img) for img in classgt_clip]
            classgt_clip = torch.stack(classgt_clip, 0).permute(1, 0, 2, 3)
            clips.append(classgt_clip)

        if self.config.classgt_stuff:
            # load classgt_stuff
            classgt_clip = ss_stuff_vid_loader(path, frame_indices, png_loader)
            classgt_clip = [self.spatial_transform(img) for img in classgt_clip]
            classgt_clip = torch.stack(classgt_clip, 0).permute(1, 0, 2, 3)
            clips.append(classgt_clip)

        if self.config.inst_seg:
            # load inst seg
            is_clip = is_vid_loader(path, frame_indices, rgb_loader)
            is_clip = [self.spatial_transform(img) for img in is_clip]
            is_clip = torch.stack(is_clip, 0).permute(1, 0, 2, 3)
            is_clip = is_clip[:2, :, :, :]
            clips.append(is_clip)

        # load class label
        target = self.data[index]
        target = self.target_transform(target)

        return clips, target

    def __len__(self):
        return len(self.data)