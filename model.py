import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
from contextlib import nullcontext

from utils import utils
from utils.visualization import disp_error_img, save_images
from utils.metric import d1_metric, thres_metric


class Model(object):
    def __init__(self, args, logger, optimizer, net, device, start_iter=0, start_epoch=0,
                 best_epe=None, best_epoch=None):
        self.args = args
        self.logger = logger
        self.optimizer = optimizer
        self.net = net
        self.device = device
        self.num_iter = start_iter
        self.epoch = start_epoch

        self.best_epe = 999. if best_epe is None else best_epe
        self.best_epoch = -1 if best_epoch is None else best_epoch

        self.train_writer = SummaryWriter(os.path.join(self.args.checkpoint_dir, "tensorBoard"))

    def train(self, train_loader, local_master):
        args = self.args
        logger = self.logger
        device = self.device
        self.net.train()  # 设置模型为训练模式！

        iter_per_epoch = len(train_loader) / args.accumulation_steps # len(train_loader)返回的是Batch的个数
        iterInEpoch = 0

        if args.freeze_bn:
            def set_bn_eval(m):
                classname = m.__class__.__name__  # 实例调用__class__属性时会指向该实例对应的类。.__class__将实例变量指向类，然后再去调用__name__类属性
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.net.apply(set_bn_eval)  # apply(fn: Callable[Module, None])：Applies fn recursively to every submodule (as returned by .children()) as well as self. Typical use includes initializing the parameters of a model (see also torch.nn.init).

        # Learning rate summary
        base_lr = self.optimizer.param_groups[0]['lr']
        offset_lr = self.optimizer.param_groups[1]['lr']
        self.train_writer.add_scalar('learningRate/base_lr', base_lr, self.epoch + 1)
        self.train_writer.add_scalar('learningRate/offset_lr', offset_lr, self.epoch + 1)

        last_print_time = time.time()

        for i, sample in enumerate(train_loader):
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]

            # 这里的Mask计算非常关键：
            # 1.在预处理时，如果图像实际尺寸小于所需尺寸时，会做padding（左右图和真实视差图都会做padding，补常数0），
            # 因此，这里的Mask操作，就会自然地屏蔽掉padding出来的部分。【注意】要留心padding对网络内部处理的影响。
            # 2.KITTI数据集约定：视差为0，表示无效视差。当然KITTI也会按照上述1进行预处理
            mask = (gt_disp > 0) & (gt_disp < args.max_disp)

            if args.load_pseudo_gt:
                pseudo_gt_disp = sample['pseudo_disp'].to(device)
                pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < args.max_disp) & (~mask)  # inverse mask # 需要修补的像素位置的mask

            if not mask.any():  # np.array.any()是或操作，任意一个元素为True，输出为True。
                continue

            # 尝试分布式训练
            # 只在DDP模式下，轮数不是args.accumulation_steps整数倍的时候使用no_sync。
            # 博客：https://blog.csdn.net/a40850273/article/details/111829836
            my_context = self.net.no_sync if args.distributed and (i + 1) % args.accumulation_steps != 0 else nullcontext
            with my_context():
                pred_disp_pyramid = self.net(left, right)  # list of H/12, H/6, H/3, H/2, H

                if args.highest_loss_only:
                    pred_disp_pyramid = [pred_disp_pyramid[-1]]  # only the last highest resolution output

                disp_loss = 0
                pseudo_disp_loss = 0
                pyramid_loss = []
                pseudo_pyramid_loss = []

                # 不同尺度loss的权重
                pyramid_weight = [0.5, 0.7, 1.0]  # PSMNet

                # if len(pred_disp_pyramid) == 5:
                #     pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
                # elif len(pred_disp_pyramid) == 4:
                #     pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
                # elif len(pred_disp_pyramid) == 3:
                #     pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
                # elif len(pred_disp_pyramid) == 1:
                #     pyramid_weight = [1.0]  # highest loss only
                # else:
                #     raise NotImplementedError

                # 将不同尺度的loss相加
                assert len(pyramid_weight) == len(pred_disp_pyramid)
                for k in range(len(pred_disp_pyramid)):
                    pred_disp = pred_disp_pyramid[k]
                    weight = pyramid_weight[k]

                    if pred_disp.size(-1) != gt_disp.size(-1):
                        pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                        pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                                  mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))  # 最后乘上这一项是必须的。因为图像放大，视差要相应增大。
                        pred_disp = pred_disp.squeeze(1)  # [B, H, W]

                    curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask], reduction='mean')
                    disp_loss += weight * curr_loss
                    pyramid_loss.append(curr_loss)

                    # Pseudo gt loss
                    if args.load_pseudo_gt:
                        pseudo_curr_loss = F.smooth_l1_loss(pred_disp[pseudo_mask], pseudo_gt_disp[pseudo_mask],
                                                            reduction='mean')
                        pseudo_disp_loss += weight * pseudo_curr_loss

                        pseudo_pyramid_loss.append(pseudo_curr_loss)

                total_loss = disp_loss + pseudo_disp_loss

                total_loss /= args.accumulation_steps
                total_loss.backward()

            isTimeToUpdate = (i + 1) % args.accumulation_steps == 0
            if isTimeToUpdate:
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.num_iter += 1
                iterInEpoch += 1

                if self.num_iter % args.print_freq == 0 or args.isDebug:
                    this_cycle = time.time() - last_print_time
                    last_print_time += this_cycle

                    time_to_finish = (args.max_epoch - self.epoch) * (1.0 * iter_per_epoch / args.print_freq) * \
                                     this_cycle / 3600.0  # 还有多久才能完成训练。单位：小时
                    logger.info('Epoch: [%3d/%3d] [%5d/%5d] Iterated:[%7d] time: %4.2fs remainT: %4.2fh disp_loss: %.3f' %
                                (self.epoch + 1, args.max_epoch, iterInEpoch, iter_per_epoch, self.num_iter, this_cycle,
                                 time_to_finish, disp_loss.item()))
                    # self.num_iter：表示当前一共进行了多少次迭代，一次参数更新表示一次迭代。
                    # steps_per_epoch：表示一个epoch中有多少次迭代，一次参数更新表示一次迭代。

                if self.num_iter % args.summary_freq == 0 or args.isDebug:
                    img_summary = dict()
                    img_summary['left'] = left    # [B, C=3, H, W]
                    img_summary['right'] = right  # [B, C=3, H, W]
                    img_summary['gt_disp'] = gt_disp  # [B, H, W]

                    if args.load_pseudo_gt:
                        img_summary['pseudo_gt_disp'] = pseudo_gt_disp

                    # Save pyramid disparity prediction
                    for s in range(len(pred_disp_pyramid)):
                        # Scale from low to high, reverse
                        save_name = 'pred_disp' + str(len(pred_disp_pyramid) - s - 1)  # pred_disp0-->pred_disp4：高分辨率->低分辨率
                        save_value = pred_disp_pyramid[s]
                        img_summary[save_name] = save_value

                    pred_disp = pred_disp_pyramid[-1]

                    if pred_disp.size(-1) != gt_disp.size(-1):
                        pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                        pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                                  mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                        pred_disp = pred_disp.squeeze(1)  # [B, H, W]
                    img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)  # [B, C=3, H, W]

                    save_images(self.train_writer, 'train', img_summary, self.num_iter)

                    epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')

                    self.train_writer.add_scalar('train/epe', epe.item(), self.num_iter)
                    self.train_writer.add_scalar('train/disp_loss', disp_loss.item(), self.num_iter)
                    self.train_writer.add_scalar('train/total_loss', total_loss.item(), self.num_iter)

                    # Save loss of different scale
                    for s in range(len(pyramid_loss)):
                        save_name = 'train/loss' + str(len(pyramid_loss) - s - 1)  # loss0-->loss4：低分辨率~高分辨率
                        save_value = pyramid_loss[s]
                        self.train_writer.add_scalar(save_name, save_value, self.num_iter)

                    d1 = d1_metric(pred_disp, gt_disp, mask)   # pred_disp.shape=[B, H, W], gt_disp.shape=[B, H, W]
                    self.train_writer.add_scalar('train/d1', d1.item(), self.num_iter)
                    thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)  # mask.shape=[B, H, W]
                    thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
                    thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
                    self.train_writer.add_scalar('train/thres1', thres1.item(), self.num_iter)
                    self.train_writer.add_scalar('train/thres2', thres2.item(), self.num_iter)
                    self.train_writer.add_scalar('train/thres3', thres3.item(), self.num_iter)

        self.epoch += 1

        # 一个epoch结束：
        # args.no_validate=False,则后面不会做self.validate()，故需要在此处记录如下信息。
        # args.no_validate=True,则后面会做self.validate()，会在elf.validate()中记录如下信息，此处不必记录。
        # 需记录的信息包括：
        # 1.最新的训练的模型和状态(写入aanet_latest.pth、optimizer_latest.pth文件) for resuming training;
        # 2.Save checkpoint of specific epoch.
        # Always save the latest model for resuming training
        utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.net,
                              epoch=self.epoch, num_iter=self.num_iter,
                              epe=-1, best_epe=self.best_epe,
                              best_epoch=self.best_epoch,
                              filename='net_latest.pth') if local_master else None
        #
        #     # Save checkpoint of specific epoch
        #     if self.epoch % args.save_ckpt_freq == 0:
        #         model_dir = os.path.join(args.checkpoint_dir, 'models')
        #         utils.check_path(model_dir)
        #         utils.save_checkpoint(model_dir, self.optimizer, self.net,
        #                               epoch=self.epoch, num_iter=self.num_iter,
        #                               epe=-1, best_epe=self.best_epe,
        #                               best_epoch=self.best_epoch,
        #                               save_optimizer=False) if local_master else None

    def validate(self, val_loader, local_master):
        args = self.args
        logger = self.logger
        logger.info('=> Start validation...')

        self.net.eval()

        num_samples = len(val_loader)
        logger.info('=> %d batches of samples found in the validation set, batchSize is %d' % (num_samples, args.val_batch_size))

        val_epe = 0
        val_d1 = 0
        val_thres1 = 0
        val_thres2 = 0
        val_thres3 = 0

        val_count = 0

        val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')

        num_imgs = 0
        valid_samples = 0

        # 遍历验证样本或测试样本
        for i, sample in enumerate(val_loader):
            if (i+1) % 100 == 0 or args.isDebug:
                logger.info('=> Validating batches：[%d/%d], batch_size: %d' % (i+1, num_samples, args.val_batch_size))

            left = sample['left'].to(self.device)  # [B, 3, H, W]
            right = sample['right'].to(self.device)
            gt_disp = sample['disp'].to(self.device)  # [B, H, W]
            mask = (gt_disp > 0) & (gt_disp < args.max_disp)

            if not mask.any():
                continue

            valid_samples += 1

            num_imgs += gt_disp.size(0)

            with torch.no_grad():
                pred_disp = self.net(left, right)  # [B, H, W], 注意，如果net输出的是多尺度的预测结果，则这里需要取出最高分辨率的

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
                save_images(self.train_writer, 'val' + str(val_count), img_summary, self.epoch)
                val_count += 1
        # 遍历验证样本或测试样本完成

        logger.info('=> Validation done!')

        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples
        mean_thres1 = val_thres1 / valid_samples
        mean_thres2 = val_thres2 / valid_samples
        mean_thres3 = val_thres3 / valid_samples

        # Save validation results
        with open(val_file, 'a') as f:
            f.write('epoch: %03d\t' % self.epoch)
            f.write('epe: %.3f\t' % mean_epe)
            f.write('d1: %.4f\t' % mean_d1)
            f.write('thres1: %.4f\t' % mean_thres1)
            f.write('thres2: %.4f\t' % mean_thres2)
            f.write('thres3: %.4f\n' % mean_thres3)
            f.write('dataset_name= %s\t mode=%s\n' % (args.dataset_name, args.mode))

        logger.info('=> Mean validation epe of epoch %d: %.3f' % (self.epoch, mean_epe))

        self.train_writer.add_scalar('val/epe', mean_epe, self.epoch)
        self.train_writer.add_scalar('val/d1', mean_d1, self.epoch)
        self.train_writer.add_scalar('val/thres1', mean_thres1, self.epoch)
        self.train_writer.add_scalar('val/thres2', mean_thres2, self.epoch)
        self.train_writer.add_scalar('val/thres3', mean_thres3, self.epoch)

        # 记录最好的效果，并保存对应的模型
        if args.val_metric == 'd1':
            if mean_d1 < self.best_epe:
                # Actually best_epe here is d1
                self.best_epe = mean_d1
                self.best_epoch = self.epoch

                utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.net,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=mean_d1, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      filename='net_best.pth') if local_master else None
        elif args.val_metric == 'epe':
            if mean_epe < self.best_epe:
                self.best_epe = mean_epe
                self.best_epoch = self.epoch

                utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.net,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=mean_epe, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      filename='net_best.pth') if local_master else None
        else:
            raise NotImplementedError

        if self.epoch == args.max_epoch:
            # Save best validation results
            with open(val_file, 'a') as f:
                f.write('\nbest epoch: %03d \t best %s: %.3f\n\n' % (self.best_epoch,
                                                                     args.val_metric,
                                                                     self.best_epe))

            logger.info('=> best epoch: %03d \t best %s: %.3f\n' % (self.best_epoch,
                                                                    args.val_metric,
                                                                    self.best_epe))

        # # Always save the latest model for resuming training
        # if not args.evaluate_only:
        #     utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.net,
        #                           epoch=self.epoch, num_iter=self.num_iter,
        #                           epe=mean_epe, best_epe=self.best_epe,
        #                           best_epoch=self.best_epoch,
        #                           filename='aanet_latest.pth') if local_master else None
        #
        #     # Save checkpoint of specific epochs
        #     if self.epoch % args.save_ckpt_freq == 0:
        #         model_dir = os.path.join(args.checkpoint_dir, 'models')
        #         utils.check_path(model_dir)
        #         utils.save_checkpoint(model_dir, self.optimizer, self.net,
        #                               epoch=self.epoch, num_iter=self.num_iter,
        #                               epe=mean_epe, best_epe=self.best_epe,
        #                               best_epoch=self.best_epoch,
        #                               save_optimizer=False) if local_master else None
