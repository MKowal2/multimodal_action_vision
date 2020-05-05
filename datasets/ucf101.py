# Source: https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/mean.py

import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import string
import matplotlib.pyplot as plt

from utils.utils import load_value_file

##########################################################################################
##########################################################################################

def pil_loader(path):
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
        return pil_loader

def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print('RGB FRAME NOT FOUND!')
            return video
    return video

def flow_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path.replace('HandStandPushups','HandstandPushups'), 'frame{:06d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            print(image_path)
            print('FLOW FRAME NOT FOUND!')
            return video
    return video

def get_default_video_loader(config):
    image_loader = get_default_image_loader()
    if config.multi_modal:
        return functools.partial(video_loader, image_loader=image_loader)
    if config.flow:
        return functools.partial(flow_loader, image_loader=image_loader)

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

##########################################################################################
##########################################################################################

def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, config):

    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)

    if not annotations:
        raise ValueError('Unable to load annotations...')

    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    if config.multi_modal:
        root_path = root_path + '/rgb_modalities/jpg/'
        for i in range(len(video_names)):
            if i % 1000 == 0:
                print('Loading UCF-101 videos [{}/{}]'.format(i, len(video_names)))

            video_path = os.path.join(root_path, video_names[i])
            if not os.path.exists(video_path):
                print('Video path does not exist!')
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

    elif config.flow:
        for i in range(len(video_names)):
            if i % 1000 == 0:
                print('Loading UCF-101 videos [{}/{}]'.format(i, len(video_names)))

            video_u_path = os.path.join(root_path + '/flow/u/', video_names[i].split('/')[1].replace('HandStandPushups','HandstandPushups'))
            if not os.path.exists(video_u_path):
                print('u video not found')
                print(video_u_path)
                continue

            video_v_path = os.path.join(root_path + '/flow/v/', video_names[i].split('/')[1].replace('HandStandPushups','HandstandPushups'))
            if not os.path.exists(video_v_path):
                print('v video not found')
                print(video_v_path)
                continue

            n_frames_file_path = os.path.join(os.path.join(root_path + '/rgb_modalities/jpg/', video_names[i], 'n_frames'))
            if not os.path.exists(n_frames_file_path):
               raise FileNotFoundError('n_frames_file_path does not exist: {}'.format(n_frames_file_path))

            n_frames = int(load_value_file(n_frames_file_path))
            if n_frames <= 0:
                continue

            begin_t = 1
            end_t = n_frames
            sample = {
                'video': os.path.join(root_path + '/flow/', video_names[i].split('/')[1]),
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

##########################################################################################
##########################################################################################

class UCF101(data.Dataset):
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
                 root_path,
                 annotation_path,
                 subset,
                 config,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, config)


        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.config = config
        self.loader = get_loader(config)

    def __getitem__(self, index):
        """
        Args:[
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']

        if self.config.multi_modal:
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            clip = self.loader(path, frame_indices)

            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)

            return clip, target

        elif self.config.flow:
            path = path.replace('HandStandPushups','HandstandPushups')
            file_count = len([name for name in os.listdir(path.replace('/flow/', '/flow/u/'))])
            if file_count < len(frame_indices):
                del frame_indices[file_count:]

            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            # print(path.replace('/flow/', '/flow/u/'))
            # print('2', frame_indices)
            clip_u = self.loader(path.replace('/flow/', '/flow/u/'), frame_indices)
            clip_v = self.loader(path.replace('/flow/', '/flow/v/'), frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                # don't normalize as it is input
                clip_u = [self.spatial_transform(img) for img in clip_u]
                clip_v = [self.spatial_transform(img) for img in clip_v]

                clip_u = torch.stack(clip_u, 0).permute(1, 0, 2, 3)
                clip_v = torch.stack(clip_v, 0).permute(1, 0, 2, 3)

                ## view flow images
                # fig, ax = plt.subplots(1, 2)
                # example1 = clip_u[:, 0, :, :].cpu().squeeze(0).permute(1, 2, 0)
                # ax[0].imshow(example1)
                # example2 = clip_v[:, 0, :, :].cpu().squeeze(0).permute(1, 2, 0)
                # ax[1].imshow(example2)
                # plt.show()

                clip = torch.cat([clip_u, clip_v], dim=0)
                # print(clip.shape)  # torch.Size([6, 64, 112, 112])

                target = self.data[index]
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return clip, target

        # shape = [b, 2, 64, h, w]
    def __len__(self):
        return len(self.data)