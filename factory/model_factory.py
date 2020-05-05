from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn

from models import resnet, wide_resnet, resnext, densenet, ResNet2D
from models.i3d import InceptionI3D


def get_model(config):

    assert config.model in ['i3d', 'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']
    # print('Initializing {} model (num_classes={})...'.format(config.model, config.num_classes))

    if config.model == 'i3d':

        from models.i3d import get_fine_tuning_parameters
        if not config.only_eval:
            if config.flow:
                model = InceptionI3D(
                    num_classes=config.num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=2,
                    dropout_keep_prob=config.dropout_keep_prob
                )
            else:
                model = InceptionI3D(
                    num_classes=config.num_classes,
                    spatial_squeeze=True,
                    final_endpoint='logits',
                    in_channels=3,
                    dropout_keep_prob=config.dropout_keep_prob
                )
        else:
            if config.flow:
                input_ch = 2
            elif config.depth:
                input_ch = 1
            elif config.classgt:
                input_ch = 1
            elif config.classgt_stuff:
                input_ch = 1
            elif config.inst_seg:
                input_ch = 1
            else:
                input_ch = 3

            model = InceptionI3D(
                num_classes=config.num_classes,
                spatial_squeeze=True,
                final_endpoint='logits',
                in_channels=input_ch,
                dropout_keep_prob=config.dropout_keep_prob
            )

    elif config.model == 'resnet':

        assert config.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from models.resnet import get_fine_tuning_parameters

        if config.flow:
            input_ch = 2
        elif config.depth:
            input_ch = 1
        elif config.classgt:
            input_ch = 1
        elif config.classgt_stuff:
            input_ch = 1
        elif config.inst_seg:
            input_ch = 1
        else:
            input_ch = 3

        if config.model_depth == 10:

            model = resnet.resnet10(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 18:

            if config.model_dimension == 2:
                print('2d yaaaaay')
                model = ResNet2D.resnet18(
                    num_classes=config.finetune_num_classes,
                    input_ch=input_ch
                )
            else:
                print('uh oh 3D resnet time!')
                model = resnet.resnet18(
                    num_classes=config.num_classes,
                    shortcut_type=config.resnet_shortcut,
                    spatial_size=config.spatial_size,
                    input_ch=input_ch,
                    sample_duration=config.sample_duration)

        elif config.model_depth == 34:

            model = resnet.resnet34(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 50:

            model = resnet.resnet50(
                num_classes=config.finetune_num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                input_ch=input_ch,
                sample_duration=config.sample_duration)

        elif config.model_depth == 101:

            model = resnet.resnet101(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 152:

            model = resnet.resnet152(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 200:

            model = resnet.resnet200(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'wideresnet':

        assert config.model_depth in [50]
        from models.wide_resnet import get_fine_tuning_parameters

        if config.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                k=config.wide_resnet_k,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'resnext':

        assert config.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters

        if config.model_depth == 50:
            model = resnext.resnet50(
                num_classes=config.finetune_num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 101:
            model = resnext.resnet101(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 152:
            model = resnext.resnet152(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'densenet':

        assert config.model_depth in [121, 169, 201, 264]
        from models.densenet import get_fine_tuning_parameters

        if config.model_depth == 121:
            model = densenet.densenet121(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 169:
            model = densenet.densenet169(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 201:
            model = densenet.densenet201(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 264:
            model = densenet.densenet264(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)



    if 'cuda' in config.device:

        # print('Moving model to CUDA device...')
        # Move model to the GPU
        model = model.cuda(config.device)

        if config.checkpoint_path:

            print('Loading pretrained model {}'.format(config.checkpoint_path))
            assert os.path.isfile(config.checkpoint_path)

            checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
            if config.model == 'i3d':
                if config.val_mode:
                    pretrained_weights = checkpoint['state_dict']
                else:
                    # pretrained_weights = checkpoint['state_dict']
                    pretrained_weights = checkpoint
            else:
                pretrained_weights = checkpoint['state_dict']

        # if config.checkpoint_path:
        #     if not config.flow:
        #         model.load_state_dict(pretrained_weights)

        if not config.val_mode:
            # Setup finetuning layer for different number of classes
            # Note: the DataParallel adds 'module' dict to complicate things...
            print('Replacing model logits with {} output classes.'.format(config.finetune_num_classes))

            # Replace input layer to receive image modalities
            if config.model == 'i3d':
                if not config.flow:
                    if config.multi_modal:
                        input_channel_dimension = 0
                        if config.rgb:
                            input_channel_dimension += 3
                        if config.depth:
                            input_channel_dimension += 1
                        # if config.flow:
                        #     input_channel_dimension += 2
                        if config.classgt:
                            input_channel_dimension += 1
                        if config.classgt_stuff:
                            input_channel_dimension += 1
                        if config.inst_seg:
                            input_channel_dimension += 2
                            if config.inst_person:
                                input_channel_dimension = input_channel_dimension - 1
                        print('Replacing model input channel dimension with {} channels.'.format(input_channel_dimension))
                        model.replace_input_layer(input_channel_dimension, config.device)

                # if config.flow:
                #     model.load_state_dict(pretrained_weights)
                #     if config.dataset == 'ucf101':
                #         print('Replacing model input channel dimension with {} channels.'.format(6))
                #         model.replace_input_layer(6, config.device)

            if config.model == 'i3d':
                model.replace_logits(config.finetune_num_classes, config.device)

            elif config.model == 'densenet':
                model.module.classifier = nn.Linear(model.module.classifier.in_features, config.finetune_num_classes)
                model.module.classifier = model.module.classifier.cuda(config.device)
            # else:
            #     model.module.fc = nn.Linear(model.module.fc.in_features, config.finetune_num_classes)
            #     model.module.fc = model.module.fc.cuda(config.device)
        # model.load_state_dict(pretrained_weights)
        # Setup which layers to train
        assert config.model in ('i3d', 'resnet'), 'finetune params not implemented...'
        finetune_criterion = config.finetune_prefixes if config.model in ('i3d', 'resnet') else config.finetune_begin_index
        if config.model == 'resnet':
            parameters_to_train = model.parameters()
        else:
            parameters_to_train = get_fine_tuning_parameters(model, finetune_criterion)

        return model, parameters_to_train
    else:
        raise ValueError('CPU training not supported.')

    return model, model.parameters()
