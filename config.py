import argparse

def parse_opts():

    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--video_path', type=str, default='/home/m3kowal/Research/Datasets/KINETICS/kinetics400', help='Path to location of dataset videos')
    parser.add_argument('--annotation_path', type=str, default='/home/m3kowal/Research/Datasets/KINETICS/kinetics100/kinetics100_V2.json', help='Path to location of dataset annotation file')
    parser.add_argument('--save_dir', default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics', type=str, help='Where to save training outputs.')
    parser.add_argument('--fusion_save_dir', default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/fusion_outputs', type=str, help='Where to save training outputs.')
    parser.add_argument('--fusion_layer_save_dir', default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/fusion_outputs/fusion_layer', type=str, help='Where to save training outputs.')

    # Dataset
    parser.add_argument('--dataset', type=str, default='kinetics', help='Dataset string (phav | kinetics | activitynet | ucf101 | blender)')
    parser.add_argument('--num_val_samples', type=int, default=1, help='Number of validation samples for each activity')
    parser.add_argument('--norm_value', default=255, type=int, help='Divide inputs by 255 or 1')
    parser.add_argument('--no_dataset_mean', action='store_true', default=True, help='Dont use the dataset mean but normalize to zero mean')
    parser.add_argument('--no_dataset_std', action='store_true',  default=True, help='Dont use the dataset std but normalize to unity std')
    parser.add_argument('--num_classes', default=400, type=int, help= 'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')

    # Preprocessing pipeline
    parser.add_argument('--spatial_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--num_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')

    # Modalities for input
    parser.add_argument('--rgb_checkpoint_path',default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191103_1426_kinetics_i3d_lr0.010_RGB_FINAL/checkpoints/save_best.pth', type=str, help='Checkpoint file (.pth) of previous rgb training')
    parser.add_argument('--depth_checkpoint_path', default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191101_2025_kinetics_i3d_lr0.010_DEPTH_FINAL/checkpoints/save_best.pth', type=str, help='Checkpoint file (.pth) of previous depth training')
    parser.add_argument('--classgt_checkpoint_path', default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191028_1945_kinetics_i3d_lr0.010_SS_FINAL/checkpoints/save_best.pth', type=str, help='Checkpoint file (.pth) of previous semantic segmentation training')
    parser.add_argument('--inst_seg_checkpoint_path', default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191101_2015_kinetics_i3d_lr0.010/checkpoints/save_best.pth', type=str, help='Checkpoint file (.pth) of previous instance segmentation training')
    parser.add_argument('--flow_checkpoint_path', default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20191027_1702_kinetics_i3d_lr0.010_FLOW_FINAL/checkpoints/save_best.pth', type=str, help='Checkpoint file (.pth) of previous flow training')
    parser.add_argument('--val_fusion', action='store_true', help='Whether validation set is used instead of test set for fusion')
    parser.add_argument('--fusion_layer', action='store_true', help='Use a fusion layer')
    parser.add_argument('--fusion_layer_type', type=str, help='Type of fusion layer to use: (uniform | class)')
    parser.add_argument('--fusion_layer_checkpoint_path',default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/fusion_outputs/fusion_layer/20190918_1828_phav_i3d_RDCIF/save_007.pth', type=str, help='Checkpoint file (.pth) of previous fusion layer training')


    # Models (general)
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float, help='Dropout keep probability')

    # Models (3D-ResNet specific)
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')

    # Finetuning
    parser.add_argument('--val_mode', action='store_true', help='Whether validation mode is used')
    parser.add_argument('--checkpoint_path', default='', type=str, help='Checkpoint file (.pth) of previous training') #/mnt/zeta_share_1/m3kowal/vfhlt_Kinetics/checkpoints/I3D/rgb_imagenet.pth
    parser.add_argument('--finetune_num_classes', default=100, type=int, help='Number of classes for fine-tuning. num_classes is set to the number when pretraining.')
    parser.add_argument('--finetune_prefixes', default='logits,Dropout,Mixed_5,Mixed_4,Mixed_3,Conv3d_2,Conv3d_1', type=str, help='Prefixes of layers to finetune, comma seperated (only used by I3D).')
    parser.add_argument('--finetune_begin_index', default=4, type=int, help='Begin block index of fine-tuning (not used by I3D).')
    parser.add_argument('--finetune_restore_optimizer', action='store_true', help='Whether to restore optimizer state')

    # Optimization
    parser.add_argument('--optimizer', default='SGD', type=str, help='Which optimizer to use (SGD | adam | rmsprop)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate (divided by 10 while training by lr-scheduler)')
    parser.add_argument('--lr_plateau_patience', default='8', type=int, help='Decay learning rate on plateauing of the validation loss (set -1 to disable)')
    parser.add_argument('--lr_scheduler_milestones', default='30,50', type=str, help='Learning rate scheduling, when to multiply learning rate by gamma when not Plateau')
    parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float, help='Learning rate decay factor')
    parser.add_argument('--early_stopping_patience', default=15, type=int, help='Early stopping patience (number of epochs)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument('--batch_size', default=48, type=int, help='Batch Size')
    parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch, only relevant for finetuning')
    parser.add_argument('--num_epochs', default=150, type=int, help='Number of epochs to train for')

    # Logging
    parser.add_argument('--print_frequency', type=int, default=100, help='Print frequency in number of train steps')
    parser.add_argument('--checkpoint_frequency', type=int, default=1, help='Save checkpoint after this number of epochs')
    parser.add_argument('--checkpoints_num_keep', type=int, default=5, help='Number of checkpoints to keep')
    parser.add_argument('--log_frequency', type=int, default=5, help='Logging frequency in number of steps')
    parser.add_argument('--log_image_frequency', type=int, default=5000, help='Logging images frequency in number of steps')
    parser.add_argument('--no_tensorboard', action='store_true', default=False, help='Disable the use of TensorboardX')

    # Misc
    parser.add_argument('--device', default='cuda:0', help='Device string cpu | cuda:0')
    parser.add_argument('--history_steps', default=25, type=int, help='History of running average meters')
    parser.add_argument('--num_workers', default=6, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--no_eval', action='store_true', default=False, help='Disable evaluation')
    parser.add_argument('--only_eval', action='store_true', default=True, help='Only perform evaluation')
    parser.add_argument('--test_set_eval', action='store_true', default=True, help='if true, validation set = test set')
    parser.add_argument('--eval_checkpoint_path',
                        default='/mnt/zeta_share_1/m3kowal/outputs/vfhlt_Kinetics/20200416_1614_kinetics_resnet_lr0.010_PERSON_1/checkpoints/save_best.pth', type=str, help='Checkpoint file (.pth) of previous training') #/home/m3kowal/Research/vfhlt/PyTorchConv3D/checkpoints/I3D/rgb_imagenet.pth
    parser.add_argument('--sample_duration', default=1, type=int, help='Temporal duration of inputs')
    parser.add_argument('--model_dimension', default=2, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--model', default='resnet', type=str, help='(i3d | resnet | preresnet | wideresnet | resnext | densenet)')

    # modalities
    parser.add_argument('--multi_modal', action='store_false', default=True,help='Use ONLY rgb images as input when unspecified')
    parser.add_argument('--rgb', action='store_true', default=False, help='Use rgb images as input')
    parser.add_argument('--depth', action='store_true', default=False, help='Use rgb images as input')
    parser.add_argument('--classgt', action='store_true', default=True, help='Use rgb images as input')
    parser.add_argument('--only_person', action='store_true', default=True, help='Use with --classgt for only_person in modality')
    parser.add_argument('--inst_seg', action='store_true', default=False, help='Use rgb images as input')
    parser.add_argument('--inst_person', action='store_true', default=False, help='Use with inst_seg to have only persons with instances')
    parser.add_argument('--classgt_stuff', action='store_true', default=False, help='Use rgb images as input')
    parser.add_argument('--flow', action='store_true', default=False, help='Use rgb images as input')
    return parser.parse_args()
