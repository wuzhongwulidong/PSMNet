from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.utils.data import Dataset, DataLoader
import os

from dataloader import transforms as myTransforms
from utils import utils
from utils.file_io import read_img, read_disp


class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 isDebug,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform

        nameFileDir = "filenames_debug" if isDebug else "filenames"
        sceneflow_finalpass_dict = {
            'train': '{}/SceneFlow_finalpass_train.txt'.format(nameFileDir),
            'val': '{}/SceneFlow_finalpass_val.txt'.format(nameFileDir),
            'test': '{}/SceneFlow_finalpass_test.txt'.format(nameFileDir)
        }

        kitti_2012_dict = {
            'train': '{}/KITTI_2012_train.txt'.format(nameFileDir),
            'train_all': '{}/KITTI_2012_train_all.txt'.format(nameFileDir),
            'val': '{}/KITTI_2012_val.txt'.format(nameFileDir),
            'test': '{}/KITTI_2012_test.txt'.format(nameFileDir)
        }

        kitti_2015_dict = {
            'train': '{}/KITTI_2015_train.txt'.format(nameFileDir),
            'train_all': '{}/KITTI_2015_train_all.txt'.format(nameFileDir),
            'val': '{}/KITTI_2015_val.txt'.format(nameFileDir),
            'test': '{}/KITTI_2015_test.txt'.format(nameFileDir)
        }

        kitti_mix_dict = {
            'train': '{}/KITTI_mix.txt'.format(nameFileDir),
            'test': '{}/KITTI_2015_test.txt'.format(nameFileDir)
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]

            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]  # left_img.split('/', 1)即以'/'为分隔符，分割一次（即分割成两部分）

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None

            if load_pseudo_gt and sample['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                                     'disp_occ_0_pseudo_gt')
                # KITTI 2012
                elif 'disp_occ' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ',
                                                                     'disp_occ_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample['pseudo_disp'] = None

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def getDataLoader(args, logger):
    # Train loader
    train_transform_list = [myTransforms.RandomCrop(args.img_height, args.img_width),
                            myTransforms.RandomColor(),
                            myTransforms.RandomVerticalFlip(),
                            myTransforms.ToTensor(),
                            # 将图像数据转化为Tensor并除以255.0，将像素数值范围归一化到[0,1]之间且[H, W, C=3]->[C=3, H, W]
                            myTransforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # 使用ImageNet数据集的均值和方差再做归一化
                            ]
    train_transform = myTransforms.Compose(train_transform_list)

    train_data = StereoDataset(data_dir=args.data_dir,
                               isDebug=args.isDebug,
                               dataset_name=args.dataset_name,
                               mode='train' if args.mode != 'train_all' else 'train_all',
                               load_pseudo_gt=args.load_pseudo_gt,
                               transform=train_transform)

    logger.info('=> {} training samples found in the training set'.format(len(train_data)))
    #  尝试分布式训练
    # 注意DistributedSampler默认参数就进行了shuffle
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if args.distributed else None

    # train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.num_workers, pin_memory=True, drop_last=True,
    #                           sampler=train_sampler)
    #  尝试分布式训练
    is_shuffle = False if args.distributed else True
    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=is_shuffle,
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler)
    # Validation loader
    val_transform_list = [myTransforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
                          myTransforms.ToTensor(),
                          myTransforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

    val_transform = myTransforms.Compose(val_transform_list)
    val_data = StereoDataset(data_dir=args.data_dir,
                             isDebug=args.isDebug,
                             dataset_name=args.dataset_name,
                             mode='val',
                             transform=val_transform)
    logger.info('=> {} val samples found in the val set'.format(len(val_data)))
    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.val_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    return train_loader, val_loader
