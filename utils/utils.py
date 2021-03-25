import os
import sys
import json
import torch
from glob import glob
import logging


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def save_command(save_path, filename='command_train.txt'):
    # # 用于DistributedDataParallel分布式训练: 仅在rank为0的进行
    # local_rank = torch.distributed.get_rank()
    # if local_rank != 0:
    #     return

    check_path(save_path)
    command = sys.argv   # 一个list：启动程序的命令行（类似于C++中main函数的第二个参数argv）
    save_file = os.path.join(save_path, filename)
    with open(save_file, 'w') as f:
        f.write(' '.join(command))


def save_args(args, filename='args.json'):
    # # 用于DistributedDataParallel分布式训练: 仅在rank为0的进行
    # local_rank = torch.distributed.get_rank()
    # if local_rank != 0:
    #     return

    args_dict = vars(args)
    check_path(args.checkpoint_dir)
    save_path = os.path.join(args.checkpoint_dir, filename)

    with open(save_path, 'w') as f:
        json.dump(args_dict, f, indent=4, sort_keys=False)


def int_list(s):
    """Convert string to int list"""
    return [int(x) for x in s.split(',')]


def fix_net_parameters(net):
    for param in net.parameters():
        param.requires_grad = False


def count_parameters(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num


def filter_specific_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return True
    return False


def filter_base_params(kv):
    specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
    for name in specific_layer_name:
        if name in kv[0]:
            return False
    return True


def get_logger(logFilePath):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    streamHandler = logging.StreamHandler()
    FileHandler = logging.FileHandler(logFilePath)

    # fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    fmt = "[%(asctime)s] %(message)s"
    streamHandler.setFormatter(logging.Formatter(fmt))
    FileHandler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(streamHandler)
    logger.addHandler(FileHandler)
    return logger


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
        print('Epoch:[%3d/%3d] [%6d/%6d] oneBatchTime:%4.2fs usedTime: %4.2fh RemainT: %4.2fh Batch_Avg_loss: %.3f'
              .format(epoch + 1, epochs,
                      batch_idx + 1, batchNum,
                      batch_time,
                      usedTime / 3600.0,
                      time_to_finish,
                      batch_avg_loss))


def save_checkpoint(save_path, optimizer, net, epoch, num_iter,
                    epe, best_epe, best_epoch, filename=None, save_optimizer=True):
    net_state = {
        'epoch': epoch,
        'num_iter': num_iter,
        'epe': epe,
        'best_epe': best_epe,
        'best_epoch': best_epoch,
        'state_dict': net.state_dict()
    }
    net_filename = 'net_epoch_{:0>3d}.pth'.format(epoch) if filename is None else filename
    net_save_path = os.path.join(save_path, net_filename)
    torch.save(net_state, net_save_path)

    # Optimizer
    if save_optimizer:
        optimizer_state = {
            'epoch': epoch,
            'num_iter': num_iter,
            'epe': epe,
            'best_epe': best_epe,
            'best_epoch': best_epoch,
            'state_dict': optimizer.state_dict()
        }
        optimizer_name = net_filename.replace('net', 'optimizer')
        optimizer_save_path = os.path.join(save_path, optimizer_name)
        torch.save(optimizer_state, optimizer_save_path)


def load_pretrained_net(net, pretrained_path, strict, return_epoch_iter=False, logger=None):
    if pretrained_path is not None:
        if torch.cuda.is_available():
            state = torch.load(pretrained_path, map_location='cuda')
            # state = torch.load(pretrained_path, map_location=map_location)
        else:
            state = torch.load(pretrained_path, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()

        weights = state['state_dict'] if 'state_dict' in state.keys() else state
        # torch.save() 和 torch.load() 函数本质将模型参数转存为一个 OrderDict 字典。
        # 使用torch.nn.parallel.DistributedDataParallel或torch.nn.parallel.DistributedDataParallel包装的模型的net.state_dict()
        # 返回的网络状态（权重）字典dict[(key,value),(key,value)]中参数名字key中都带有module字样，即形如：module.feature_extractor.conv1.0.weight
        # 但是，当不使用DistributedDataParallel和DistributedDataParallel包装时，key不带有module字样。
        # 所以建议net.load_state_dict使用strict=True，以立即发现网络load错误的情形。
        # 需要注意，是否需要删除module字符串。如下代码时用于删除module字符串的。
        # for k, v in weights.items():
        #     name = k[7:] if 'module' in k and not resume else k
        #     new_state_dict[name] = v
        new_state_dict = weights

        if not strict:
            missing_keys, unexpected_keys = net.load_state_dict(new_state_dict, strict=False)  # ignore intermediate output
            # missing_keys: net模型需要但是未找到的参数key; unexpected_keys: new_state_dict提供的但是net模型不需要的字典key
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                logger.info("load pre-trained model Successfully!!!")
            else:
                logger.info("fail to load pre-trained model!!!")
        else:
            net.load_state_dict(new_state_dict)  # optimizer has no argument `strict` and does not return anything!

        if return_epoch_iter:
            epoch = state['epoch'] if 'epoch' in state.keys() else None
            num_iter = state['num_iter'] if 'num_iter' in state.keys() else None
            best_epe = state['best_epe'] if 'best_epe' in state.keys() else None
            best_epoch = state['best_epoch'] if 'best_epoch' in state.keys() else None
            return epoch, num_iter, best_epe, best_epoch


def resume_latest_ckpt(checkpoint_dir, net, net_name, strict, logger):
    ckpts = sorted(glob(checkpoint_dir + '/' + net_name + '*.pth'))

    if len(ckpts) == 0:
        raise RuntimeError('=> No checkpoint found while resuming training')

    latest_ckpt = ckpts[-1]
    logger.info('=> Resume latest %s checkpoint: %s' % (net_name, os.path.basename(latest_ckpt)))
    epoch, num_iter, best_epe, best_epoch = load_pretrained_net(net, latest_ckpt, strict, return_epoch_iter=True, logger=logger)

    return epoch, num_iter, best_epe, best_epoch