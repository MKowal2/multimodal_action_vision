from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import torch

from transforms.spatial_transforms import Normalize
from torch.utils.data import DataLoader

from datasets.kinetics import Kinetics, Kinetics_fusion
from datasets.phav import PHAV, PHAV_modalities, PHAV_proxy
from datasets.ucf101 import UCF101

##########################################################################################
##########################################################################################

def get_training_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['phav', 'kinetics', 'activitynet', 'ucf101', 'blender']

    if config.dataset == 'kinetics':
        if config.multi_modal:
            training_data = Kinetics(
                config,
                config.video_path,
                config.annotation_path,
                'training',
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform)
        else:
            training_data = Kinetics(
                config,
                config.video_path,
                config.annotation_path,
                'training',
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform)

    elif config.dataset == 'phav':
        if not config.proxy:
            training_data = PHAV(
            config,
            config.video_path,
            config.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
        else:
            training_data = PHAV_proxy(
                config,
                config.video_path,
                config.annotation_path,
                'training',
                spatial_transform=spatial_transform,
                temporal_transform=temporal_transform,
                target_transform=target_transform)


    elif config.dataset == 'activitynet':

        training_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif config.dataset == 'ucf101':

        training_data = UCF101(
            config.video_path,
            config.annotation_path,
            'training',
            config,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif config.dataset == 'blender':

        training_data = BlenderSyntheticDataset(
            root_path=config.video_path,
            subset='train',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


##########################################################################################
##########################################################################################

def get_validation_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['phav','kinetics', 'activitynet', 'ucf101', 'blender']

    # Disable evaluation
    if config.no_eval:
        return None

    if config.dataset == 'kinetics':

        validation_data = Kinetics(
            config,
            config.video_path,
            config.annotation_path,
            'validation',
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)


    elif config.dataset == 'phav':
        if not config.proxy:
            validation_data = PHAV(
                config,
                config.video_path,
                config.annotation_path,
                'validation',
                config.num_val_samples,
                spatial_transform,
                temporal_transform,
                target_transform,
                sample_duration=config.sample_duration)
        else:
            validation_data = PHAV_proxy(
                config,
                config.video_path,
                config.annotation_path,
                'validation',
                config.num_val_samples,
                spatial_transform,
                temporal_transform,
                target_transform,
                sample_duration=config.sample_duration)

    elif config.dataset == 'activitynet':

        validation_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            'validation',
            False,
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'ucf101':

        validation_data = UCF101(
            config.video_path,
            config.annotation_path,
            'validation',
            config,
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'blender':

        validation_data = BlenderSyntheticDataset(
            root_path=config.video_path,
            subset='validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return validation_data


##########################################################################################
##########################################################################################

def get_test_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender', 'phav']
    # assert config.test_subset in ['val', 'test']
    #
    # if config.test_subset == 'val':
    #     subset = 'validation'
    # elif config.test_subset == 'test':
    #     subset = 'testing'

    if config.dataset == 'kinetics':
        if config.val_fusion:
            test_data = Kinetics_fusion(
                config,
                config.video_path,
                config.annotation_path,
                'test',
                1,
                spatial_transform,
                temporal_transform,
                target_transform,
                sample_duration=config.sample_duration)
        else:
            test_data = Kinetics(
                config,
                config.video_path,
                config.annotation_path,
                'test',
                1,
                spatial_transform,
                temporal_transform,
                target_transform,
                sample_duration=config.sample_duration)

    elif config.dataset == 'phav':
        test_data = PHAV(
        config,
        config.video_path,
        config.annotation_path,
        'test',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform)

    elif config.dataset == 'activitynet':

        test_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'ucf101':

        test_data = UCF101(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    return test_data


##########################################################################################
##########################################################################################

def get_normalization_method(config):
    if config.no_mean_norm and not config.std_norm:
        return Normalize([0, 0, 0], [1, 1, 1])
    elif not config.std_norm:
        return Normalize(config.mean, [1, 1, 1])
    else:
        return Normalize(config.mean, config.std)


def get_modality_training_set(config, input_trainsform,modality):
    if config.dataset == 'phav':
        training_data = PHAV_modalities(
            config,
            config.video_path,
            config.annotation_path,
            'training',
            modality,
            spatial_transform=input_trainsform
        )
    return training_data

def get_modality_validation_set(config, input_trainsform ,modality):
    if config.dataset == 'phav':
        validation_data = PHAV_modalities(
            config,
            config.video_path,
            config.annotation_path,
            'validation',
            modality,
            spatial_transform=input_trainsform
        )
    return validation_data

def get_modality_test_set(config, spatial_transform, modality):
    if config.dataset == 'phav':
        test_data = PHAV_modalities(
            config,
            config.video_path,
            config.annotation_path,
            'test',
            modality,
            spatial_transform=spatial_transform
        )
    return test_data


##########################################################################################
##########################################################################################

def get_data_loaders(config, train_transforms, validation_transforms=None):

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    data_loaders = dict()

    if config.only_eval:
        if not config.test_set_eval:
            dataset_validation = get_validation_set(
                config, validation_transforms['spatial'],
                validation_transforms['temporal'], validation_transforms['target'])

            print('Found {} validation examples'.format(len(dataset_validation)))

            data_loaders['validation'] = DataLoader(
                dataset_validation, config.batch_size, shuffle=True,
                num_workers=config.num_workers, pin_memory=True)
        else:
            dataset_validation = get_test_set(
                config, validation_transforms['spatial'],
                validation_transforms['temporal'], validation_transforms['target'])

            print('Found {} validation examples'.format(len(dataset_validation)))

            data_loaders['validation'] = DataLoader(
                dataset_validation, config.batch_size, shuffle=True,
                num_workers=config.num_workers, pin_memory=True)

        return data_loaders

    # Define the data pipeline
    dataset_train = get_training_set(
        config, train_transforms['spatial'],
        train_transforms['temporal'], train_transforms['target'])

    data_loaders['train'] = DataLoader(
        dataset_train, config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)

    print('Found {} training examples'.format(len(dataset_train)))

    if not config.no_eval and validation_transforms:

        dataset_validation = get_validation_set(
            config, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])

        print('Found {} validation examples'.format(len(dataset_validation)))

        data_loaders['validation'] = DataLoader(
            dataset_validation, config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True)

    return data_loaders

def get_test_data_loader(config, validation_transforms=None):

    data_loaders = dict()

    if config.only_eval:
        if not config.test_set_eval:
            dataset_validation = get_validation_set(
                config, validation_transforms['spatial'],
                validation_transforms['temporal'], validation_transforms['target'])

            data_loaders['validation'] = DataLoader(
                dataset_validation, config.batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True)
        else:
            dataset_validation = get_test_set(
                config, validation_transforms['spatial'],
                validation_transforms['temporal'], validation_transforms['target'])

            data_loaders['validation'] = DataLoader(
                dataset_validation, config.batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True)

        return data_loaders

def get_fusion_data_loaders(config, validation_transforms=None):

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    if config.val_fusion:
        dataset_test = get_validation_set(
            config, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])
    else:
        dataset_test = get_test_set(
            config, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])

    print('Found {} test examples'.format(len(dataset_test)))

    data_loaders = DataLoader(
        dataset_test, config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=False)

    return data_loaders


def get_modality_data_loaders(config, train_transforms, validation_transforms=None):

    if config.depth:
        modality = 'depth'
    if config.classgt:
        modality='classgt'
    if config.inst_seg:
        modality='instgt'
    if config.flow:
        modality = 'flow'

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    data_loaders = dict()
    # Define the data pipeline
    dataset_train = get_modality_training_set(
        config, train_transforms['spatial'], modality)


    data_loaders['train'] = DataLoader(
        dataset_train, config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)

    print('Found {} training examples'.format(len(dataset_train)))

    if not config.no_eval and validation_transforms:

        dataset_validation = get_modality_validation_set(
            config, validation_transforms['spatial'], modality)

        print('Found {} validation examples'.format(len(dataset_validation)))

        data_loaders['validation'] = DataLoader(
            dataset_validation, config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True)

    return data_loaders
