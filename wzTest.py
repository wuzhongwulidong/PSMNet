from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader.StereoDataset import StereoDataset
from utils import utils
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
from utils.visualization import disp_error_img, save_images
from utils.metric import d1_metric, thres_metric
from dataloader import transforms as myTransforms

parser = argparse.ArgumentParser(description='PSMNet')
# parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
# parser.add_argument('--model', default='stackhourglass', help='select model')
# parser.add_argument('--dataSet', default="SceneFlow", help='data set name')
# parser.add_argument('--datapath', default="./dataset/SceneFlow", help='datapath')
# parser.add_argument('--max_epoch', type=int, default=10, help='number of epochs to train')
# parser.add_argument('--loadmodel', default=None, help='load model')  # 需设置为model的保存路径和名称: ./checkpoint/+文件名称
# parser.add_argument('--checkpoint_dir', default='./checkpoint', help='save model')  # model的保存路径
# parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=234, metavar='S', help='random seed (default: 1)')
parser.add_argument('--isDebug', default=1, type=int, help='For code debug only!')  # 0: False; 1: True
parser.add_argument('--model', default='PSMNet_stackhourglass', help='select model')
parser.add_argument('--mode', default='test', type=str, help='Validation mode on small subset or test mode on full test data')

# For KITTI, using 384x1248 for validation
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')
parser.add_argument('--data_dir', default='dataset/SceneFlow', type=str, help='Training dataset')
parser.add_argument('--checkpoint_dir', default="./checkpoint/", type=str, help='Directory to save model checkpoints and logs')

parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for test')
parser.add_argument('--test_img_height', default=544, type=int, help='Image height for test')
parser.add_argument('--test_img_width', default=960, type=int, help='Image width for test')

parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')

# Training data
# parser.add_argument('--data_dir', default='data/SceneFlow', type=str, help='Training dataset')

# parser.add_argument('--do_validate', action='store_true', help='Do validation')
# parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')
#
# parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
# parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
# parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
# parser.add_argument('--accumulation_steps', default=4, type=int, help='Batch size for training')


# parser.add_argument('--img_height', default=288, type=int, help='Image height for training')
# parser.add_argument('--img_width', default=512, type=int, help='Image width for training')

parser.add_argument('--pretrained_net', default='./checkpoint/net_best.pth', type=str, help='Pretrained network, Absolute Full Path!')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')

args = parser.parse_args()
logger = utils.get_logger(os.path.join(args.checkpoint_dir, "testLog.txt"))


# torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
# if args.cuda: torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子；
# 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。

# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#
# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
#
# TrainImgLoader = torch.utils.data.DataLoader(
#     DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
#     batch_size=12, shuffle=True, num_workers=8, drop_last=False)
#
# TestImgLoader = torch.utils.data.DataLoader(
#     DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
#     batch_size=8, shuffle=False, num_workers=4, drop_last=False)

# if args.model == 'stackhourglass':
#     model = stackhourglass(args.maxdisp)
# elif args.model == 'basic':
#     model = basic(args.maxdisp)
# else:
#     logger.info('No model found!!!')
#
# if args.cuda:
#     model = nn.DataParallel(model)
#     model.cuda()
#
#
# def loadTrainedModel(emptyModel, path):
#     logger.info('=> Loading pretrained model: {}'.format(path))
#     pretrain_dict = torch.load(path)
#     missing_keys, unexpected_keys = emptyModel.load_state_dict(pretrain_dict['state_dict'])
#     if len(missing_keys) == 0 and len(unexpected_keys) == 0:
#         logger.info("=> Pre-trained model is loaded Successfully!")
#     else:
#         logger.info("=> Fail to load pre-trained model!!!")


# if args.loadmodel is not None:
#     loadTrainedModel(model, args.loadmodel)
#
# logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#
#
# def test(imgL, imgR, disp_true):
#     model.eval()
#
#     if args.cuda:
#         imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
#     # ---------
#     mask = disp_true < 192
#     # ----
#     # 确保测试图像的[H,W]能被16整除。
#     if imgL.shape[2] % 16 != 0:
#         times = imgL.shape[2] // 16
#         top_pad = (times + 1) * 16 - imgL.shape[2]
#     else:
#         top_pad = 0
#
#     if imgL.shape[3] % 16 != 0:
#         times = imgL.shape[3] // 16
#         right_pad = (times + 1) * 16 - imgL.shape[3]
#     else:
#         right_pad = 0
#
#     imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
#     imgR = F.pad(imgR, (0, right_pad, top_pad, 0))
#
#     with torch.no_grad():
#         output3 = model(imgL, imgR)
#         output3 = torch.squeeze(output3)
#
#     if top_pad != 0:
#         img = output3[:, top_pad:, :]
#     else:
#         img = output3
#
#     # 确保测试图像有效标签像素个数不为0
#     if len(disp_true[mask]) == 0:
#         loss = 0
#     else:
#         loss = F.l1_loss(img[mask],
#                          disp_true[mask])  # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error
#
#     return loss.data.cpu()


def showTimeInfo(freq, epochs, epoch, batchNum, batch_idx, usedTime, batch_time, batch_avg_loss):
    """
    输出程序执行时间的信息。
    :param freq: 打印频率，即训练多少个batch打一次
    :param epochs: 总的epoach数
    :param epoch: 当前epoach数
    :param batchNum: batch的总个数
    :param batch_idx: 当前是当前epoach的第几个batch
    :param usedTime: 已经训练了多长时间（秒）
    :param batch_time: 训练一个batch所花的时间（秒）
    :param batch_avg_loss: 实际是每个像素的loss，原因是loss函数设置了元素平均（具体参见torch.nn.SmoothL1Loss的文档）
    """
    if batch_idx % freq == 0:
        time_to_finish = (epochs - epoch) * batchNum * batch_time / 3600.0  # 还有多久才能完成训练。单位：小时
        logger.info(
            'Epoch:[%3d/%3d] [%6d/%6d] oneBatchTime:%4.2fs usedTime: %4.2fh RemainT: %4.2fh Batch_Avg_loss: %.3f'
            .format(epoch + 1, epochs,
                    batch_idx + 1, batchNum,
                    batch_time,
                    usedTime / 3600.0,
                    time_to_finish,
                    batch_avg_loss))


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    board_writer = SummaryWriter(os.path.join(args.checkpoint_dir, "tensorBoard"))

    if args.model == 'PSMNet_stackhourglass':
        net = PSMNet_stackhourglass(args.max_disp)
    elif args.model == 'PSMNet_basic':
        net = PSMNet_basic(args.max_disp)
    else:
        print('no model')

    # Validation loader
    test_transform_list = [myTransforms.RandomCrop(args.test_img_height, args.test_img_width, validate=True),
                           myTransforms.ToTensor(),
                           myTransforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    test_transform = myTransforms.Compose(test_transform_list)
    test_data = StereoDataset(data_dir=args.data_dir,
                              isDebug=args.isDebug,
                              dataset_name=args.dataset_name,
                              mode='test',
                              transform=test_transform)

    logger.info('=> {} test samples found in the test set'.format(len(test_data)))

    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    net.cuda()

    if args.pretrained_net is not None:
        logger.info('=> Loading pretrained Net: %s' % args.pretrained_net)
        # Enable training from a partially pretrained model
        utils.load_pretrained_net(net, args.pretrained_net, strict=args.strict, logger=logger)
    else:
        logger.info('=>  args.pretrained_net is None! Please specify it!!!')
        return

    assert args.test_batch_size == 1, "test_batch_size must be 1."

    logger.info('=> Start testing...')
    testOnTestSet(net, test_loader, args.dataset_name, board_writer, mode="test", epoch=0)


# assert args.val_batch_size == 1
def testOnTestSet(model, data_loader, dataset_name, board_writer, mode="test", epoch=0):
    """
    如果是要看在测试集上的效果，请确保val_loader的BatchSize=1
    :param data_loader:
    :param dataset_name:
    :param mode:
    :param epoch:
    :return:
    """
    model.eval()
    num_samples = len(data_loader)
    logger.info('=> {} samples found in the {} set'.format(num_samples, mode))
    val_epe = 0
    val_d1 = 0
    val_thres1 = 0
    val_thres2 = 0
    val_thres3 = 0
    val_count = 0
    val_file = os.path.join(args.checkpoint_dir, '{}_results.txt'.format(mode))

    num_imgs = 0
    valid_samples = 0
    # 遍历验证样本或测试样本
    for i, sample in enumerate(data_loader):
        if (i + 1) % 100 == 0 or args.isDebug:
            logger.info('=> Testing batches：[%d/%d], batch_size: %d' % (i + 1, num_samples, args.test_batch_size))

        left = sample['left'].cuda()  # [B, 3, H, W]
        right = sample['right'].cuda()
        gt_disp = sample['disp'].cuda()   # [B, H, W]
        mask = (gt_disp > 0) & (gt_disp < args.max_disp)

        # 这里的Mask计算非常关键：
        # 1.在预处理时，如果图像实际尺寸小于所需尺寸时，会做padding（左右图和真实视差图都会做padding，补常数0），
        # 因此，这里的Mask操作，就会自然地屏蔽掉padding出来的部分。【注意】要留心padding对网络内部处理的影响。
        # 2.KITTI数据集约定：视差为0，表示无效视差。当然KITTI也会按照上述1进行预处理。
        if not mask.any():
            # 全图无有效视差
            continue
        else:
            valid_samples += 1

        num_imgs += gt_disp.size(0)

        with torch.no_grad():
            pred_disp = model(left, right)  # 请确保pred_disp的维度为[B, H, W]

        # 1. 如果预测的pred_disp尺寸大于gt_disp的尺寸，是OK的，因为mask是和gt_disp同尺寸的，后续的mask操作会裁掉大于gt_disp的部分。
        # 2. 如果预测的pred_disp尺寸小于gt_disp的尺寸，是不OK的，需要对pred_disp进行上采样。
        if pred_disp.size(-1) < gt_disp.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
                                      mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
        d1 = d1_metric(pred_disp, gt_disp, mask)
        thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
        thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
        thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)

        val_epe += epe.item()
        val_d1 += d1.item()
        val_thres1 += thres1.item()
        val_thres2 += thres2.item()
        val_thres3 += thres3.item()

        # Save 3 images for visualization
        if i in [num_samples // 4, num_samples // 2, num_samples // 4 * 3]:
            img_summary = dict()
            img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
            img_summary['left'] = left
            img_summary['right'] = right
            img_summary['gt_disp'] = gt_disp
            img_summary['pred_disp'] = pred_disp
            save_images(board_writer, 'test' + str(val_count), img_summary, epoch)
            val_count += 1

    mean_epe = val_epe / valid_samples
    mean_d1 = val_d1 / valid_samples
    mean_thres1 = val_thres1 / valid_samples
    mean_thres2 = val_thres2 / valid_samples
    mean_thres3 = val_thres3 / valid_samples

    # Save test results
    with open(val_file, 'a') as f:
        f.write('dataset_name= %s\t mode=%s\n' % (dataset_name, mode))
        f.write('epoch: %03d\t' % epoch)
        f.write('epe: %.3f\t' % mean_epe)
        f.write('d1: %.4f\t' % mean_d1)
        f.write('thres1: %.4f\t' % mean_thres1)
        f.write('thres2: %.4f\t' % mean_thres2)
        f.write('thres3: %.4f\n' % mean_thres3)

    logger.info(
        '=> [%s] %s mean results: epoch: %03d\t epe: %.3f\t d1: %.4f\t thres1: %.4f\t thres2: %.4f\t thres3: %.4f\n'
        % (mode, dataset_name, epoch, mean_epe, mean_d1, mean_thres1, mean_thres2, mean_thres3))

    logger.info('=> {} Done!'.format(mode))


if __name__ == '__main__':
    main()
