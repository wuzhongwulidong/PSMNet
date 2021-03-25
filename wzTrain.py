import torch
from torch.cuda import synchronize
from torch.utils.data import DataLoader
from torch import distributed

import argparse
import numpy as np
import os

import dataloader
from dataloader.StereoDataset import getDataLoader
from utils import utils
from models import *
import model

# python main.py --maxdisp 192 \
#                --model PSMNet_stackhourglass \
#                --data_dir data/SceneFlow \
#                --max_epoch 10 \
#                --loadmodel ./trained/checkpoint_10.tar \
#                --savemodel ./trained/

parser = argparse.ArgumentParser()
parser.add_argument('--isDebug', default=1, type=int, help='For code debug only!')  # 0: False; 1: True
parser.add_argument('--print_freq', default=50, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')

parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
parser.add_argument('--model', default='PSMNet_stackhourglass', help='select model')
parser.add_argument('--mode', default='train', type=str, help='Validation mode on small subset or test mode on full test data')
parser.add_argument('--max_epoch', default=10, type=int, help='Maximum epoch number for training')

# Training data
parser.add_argument('--data_dir', default='./dataset/SceneFlow', type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')
parser.add_argument('--do_validate', action='store_true', help='Do validation')
parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')

parser.add_argument('--batch_size', default=3, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=3, type=int, help='Batch size for validation')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--lr_scheduler_type', default=None, help='Type of learning rate scheduler')  # 'MultiStepLR'

parser.add_argument('--accumulation_steps', default=1, type=int, help='Batch size for training')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--load_pseudo_gt', default=False, type=bool, help='For KITTI2015 KITTI2012')

parser.add_argument('--freeze_bn', action='store_true', help='Switch BN to eval mode to fix running statistics')
parser.add_argument('--highest_loss_only', action='store_true', help='Only use loss on highest scale for finetuning')

# 【训练时】裁剪的图片尺寸
parser.add_argument('--img_height', default=288, type=int, help='Image height for training')
parser.add_argument('--img_width', default=512, type=int, help='Image width for training')
# 【验证时】裁剪的图片尺寸
# 1.KITTI：For KITTI, using 384x1248 for validation
# 2.SceneFlow
parser.add_argument('--val_img_height', default=576, type=int, help='Image height for validation')
parser.add_argument('--val_img_width', default=960, type=int, help='Image width for validation')

parser.add_argument('--checkpoint_dir', default="./checkpoint/", type=str, help='Directory to save model checkpoints and logs')
parser.add_argument('--pretrained_net', default=None, type=str, help='Pretrained network, Absolute Full Path!')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
#
# # Model
# parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')

# parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')


#
# # AANet
# parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor')
# parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
# parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature')
# parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
# parser.add_argument('--feature_similarity', default='correlation', type=str,
#                     help='Similarity measure for matching cost')
# parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
# parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
# parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
# parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
#                                                                'aggragetion')
# parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
# parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
# parser.add_argument('--no_intermediate_supervision', action='store_true',
#                     help='Whether to add intermediate supervision')
# parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
# parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
# parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')
#


#
# # Learning rate
# parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Decay gamma')

# parser.add_argument('--milestones', default=None, type=str, help='Milestones for MultiStepLR')
#
# # Loss

# parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')
#
# # Log


# parser.add_argument('--no_build_summary', action='store_true', help='Dont save sammary when training to save space')
# parser.add_argument('--save_ckpt_freq', default=5, type=int, help='Save checkpoint frequency (epochs)')

# parser.add_argument('--evaluate_only', action='store_true', help='Evaluate pretrained models')

#  尝试分布式训练:
parser.add_argument("--local_rank", type=int)  # 必须有这一句，但是local_rank是torch.distributed.launch自动分配和传入的。
parser.add_argument("--distributed", action='store_true', help="use DistributedDataParallel")
parser.add_argument('--seed', type=int, default=326, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
# 确保./checkpoint/目录已经存在
utils.check_path(args.checkpoint_dir)
logger = utils.get_logger(os.path.join(args.checkpoint_dir, "trainLog.txt"))

if args.isDebug:
    args.max_epoch = 8
    args.do_validate = True

if args.distributed:
    #  尝试分布式训练
    # local_rank = torch.distributed.get_rank()
    # local_rank表示本台机器上的进程序号,是由torch.distributed.launch自动分配和传入的。
    local_rank = args.local_rank
    # 根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # 初始化DDP，使用默认backend(nccl)就行
    torch.distributed.init_process_group(backend="nccl")
    logger.info("args.local_rank={}".format(args.local_rank))
else:
    device = torch.device("cuda")

# 尝试分布式训练
local_master = True if not args.distributed else args.local_rank == 0
torch.backends.cudnn.benchmark = True  # https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317
# 打印所用的参数
if local_master:
    logger.info('[Info] used parameters: {}'.format(vars(args)))
    logger.info('=> Use %d GPUs' % torch.cuda.device_count()) if local_master else None


def setInitLR(net, args):
    # Learning rate for offset learning is set 0.1 times those of existing layers
    specific_params = list(filter(utils.filter_specific_params,
                                  net.named_parameters()))
    base_params = list(filter(utils.filter_base_params,
                              net.named_parameters()))

    specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
    base_params = [kv[1] for kv in base_params]

    specific_lr = args.learning_rate * 0.1
    params_group = [
        {'params': base_params, 'lr': args.learning_rate},
        {'params': specific_params, 'lr': specific_lr},]

    return params_group


def selectModel(model_name):
    net = None
    if model_name == 'PSMNet_stackhourglass':
        net = PSMNet_stackhourglass(args.max_disp)
    elif model_name == 'PSMNet_basic':
        net = PSMNet_basic(args.max_disp)
    else:
        print('no model')

    assert net is not None, "net must not be None!!!"
    return net


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader = getDataLoader(args, logger)

    net = selectModel(args.model)

    # logger.info('%s' % net) if local_master else None

    # if args.pretrained_net is not None:
    #     logger.info('=> Loading pretrained Net: %s' % args.pretrained_net)
    #     # Enable training from a partially pretrained model
    #     utils.load_pretrained_net(net, args.pretrained_net, strict=args.strict, logger=logger)

    net.to(device)
    # if torch.cuda.device_count() > 1:
    if args.distributed:
        # aanet = torch.nn.DataParallel(aanet)
        #  尝试分布式训练
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        synchronize()

    # Save parameters
    num_params = utils.count_parameters(net)
    logger.info('=> Number of trainable parameters: %d' % num_params)

    # 网络的特殊部分，设置特殊的学习率：specific_lr = args.learning_rate * 0.1
    params_group = setInitLR(net, args)

    # Optimizer
    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)

    # Resume training
    if args.resume:
        # 1. resume Net
        start_epoch, start_iter, best_epe, best_epoch = utils.resume_latest_ckpt(
            args.checkpoint_dir, net, 'net_latest', False, logger)
        # 2. resume Optimizer
        utils.resume_latest_ckpt(args.checkpoint_dir, optimizer, 'optimizer_latest', True, logger)
    else:
        start_epoch = 0
        start_iter = 0
        best_epe = None
        best_epoch = None

    # LR scheduler
    if args.lr_scheduler_type is not None:
        last_epoch = start_epoch if args.resume else start_epoch - 1
        if args.lr_scheduler_type == 'MultiStepLR':
            milestones = [int(step) for step in args.milestones.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones,
                                                                gamma=args.lr_decay_gamma,
                                                                last_epoch=last_epoch)  # 最后这个last_epoch参数很重要：如果是resume的话，则会自动调整学习率适去应last_epoch。
        else:
            raise NotImplementedError
    # model.Model(net)对net做了进一步封装。
    train_model = model.Model(args, logger, optimizer, net, device, start_iter, start_epoch,
                              best_epe=best_epe, best_epoch=best_epoch)
    logger.info('=> Start training...')

    for epoch in range(start_epoch, args.max_epoch):  # 训练主循环（Epochs）！！！
        # ensure distribute worker sample different data,
        # set different random seed by passing epoch to sampler
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            logger.info('train_loader.sampler.set_epoch({})'.format(epoch))

        train_model.train(train_loader, local_master)

        if args.do_validate:
            train_model.validate(val_loader, local_master)  # 训练模式下：边训练边验证。

        if args.lr_scheduler_type is not None:
            lr_scheduler.step()  # 调整Learning Rate

    logger.info('=> End training\n\n')


if __name__ == '__main__':
    main()
