import argparse
import logging
import os
import pprint
import random
import sys

import torch
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.mydataset import  MyDataset, RandomGenerator
import time
from torchvision import transforms

from networks.net_factory import net_factory
import shutil
import yaml
from util.utils import (count_params, cal_metric_pixel_2D, compute_confidence_interval)
from util import losses
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str,
                    default="configs/isic2018.yaml",
                    help='isic2018/KvasirSEG')
# model name and save path
parser.add_argument('--method', default="FS", type=str,
                    help='')
parser.add_argument('--model', type=str, default='unet',
                    help='')
# label rat
parser.add_argument('--label-rat', default=0.05,
                    type=float, help='label_rat')
# seed
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
def getLog(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/logging.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

def main(args):
    # save model and logging
    exp = "exp/{}/{}".format(
        cfg['data']['dataset'], args.method)

    lb_rat = str(int(args.label_rat * 100)) + '%'

    snapshot_path = "{}/{}/{}_labeled".format(
        exp, args.model, lb_rat)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    current_file = __file__
    shutil.copyfile(current_file, os.path.join(snapshot_path, os.path.basename(current_file)))

    getLog(args, snapshot_path)
    logging.info("Configuration settings:\n%s", yaml.dump(cfg))

    evens_path = snapshot_path + '/log'
    if not os.path.exists(evens_path):
        os.makedirs(evens_path)
    writer = SummaryWriter(evens_path)

    dataset = cfg['data']['dataset']
    root_path = cfg['data']['root_path']
    crop_size = cfg['data']['crop_size']
    in_ch = cfg['data']['in_chns']

    # train params
    base_lr = cfg['train']['base_lr']
    num_classes = cfg['train']['num_classes']
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']

    labeled_id_path = "{}/ratio_{}/labeled.txt".format(
        root_path, lb_rat)
    unlabeled_id_path = "{}/ratio_{}/unlabeled.txt".format(
        root_path, lb_rat)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    model = net_factory(net_type=args.model, in_chns=in_ch,
                        class_num=num_classes)
    optimizer = SGD(model.parameters(), lr=base_lr,
                    momentum=0.9, weight_decay=0.0001)

    logging.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
    dice_loss = losses.DiceLoss(num_classes)

    trainset = MyDataset(dataset, root_path, 'train_l',
              crop_size, labeled_id_path,
              transform=transforms.Compose([RandomGenerator(crop_size)])
              )

    valset = MyDataset(dataset, root_path, 'val')

    if "ISIC2018" in dataset:
        total_num = 2075
    elif "KvasirSEG" in dataset:
        total_num = 880

    logging.info('Total samples is: {}, labeled samples is: {}, labeled rate is: {}\n'.format(
        total_num, len(trainset.lb_num), lb_rat)
    )
    
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True, num_workers=1,
                             drop_last=True,
                             worker_init_fn=worker_init_fn,
                             )
    valloader = DataLoader(valset, batch_size=1, shuffle=False,
                        num_workers=1)

    total_iters = len(trainloader) * epochs
    logging.info('Total iters is: {}\n'.format(total_iters))
    previous_best_dice, previous_best_acc = 0.0, 0.0
    epoch = -1
    iters = 0

    if os.path.exists(os.path.join(snapshot_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(snapshot_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_dice = checkpoint['previous_best_dice']
        previous_best_acc = checkpoint['previous_best_acc']

        logging.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, epochs):
        logging.info(
            '===========> Epoch: {:}, LR: {:.5f}, \033[31m Previous best {} dice: {:.2f}, Overall Accuracy: {:.2f}\033[0m'.format(
                epoch, optimizer.param_groups[0]['lr'], dataset, previous_best_dice, previous_best_acc))

        for i, sample_l in enumerate(trainloader):

            img, mask = sample_l["image"], sample_l["label"]
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)
            pred_soft = torch.softmax(pred, dim=1)

            loss_ce = criterion(pred, mask.long())
            loss_dice = dice_loss(pred_soft, mask.unsqueeze(1).float(), ignore=255)
            loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = base_lr * (1 - iters / total_iters) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            iters = epoch * len(trainloader) + i

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logging.info('Iters: %d, Total loss: %f, loss ce: %f, loss dice: %f '%(
                    iters, loss.item(), loss_ce.item(), loss_dice.item()
                ))

        model.eval()
        metric_list = 0.0
        m_dice_list = []
        m_jaccard_list = []
        pa_list = []
        m_hd95_list = []
        m_asd_list = []
        for i, (image, target) in enumerate(valloader):
            metric_i, PA = cal_metric_pixel_2D(
                image, target, model,
                classes=cfg['train']['num_classes'],
                patch_size=cfg['data']['crop_size'])

            dice = [row[0] for row in metric_i]
            jaccard = [row[1] for row in metric_i]
            hd95 = [row[2] for row in metric_i]
            asd = [row[3] for row in metric_i]

            metric_list += np.array(metric_i)
            m_dice_list.append(np.mean(dice))
            m_jaccard_list.append(np.mean(jaccard))
            m_hd95_list.append(np.mean(hd95))
            m_asd_list.append(np.mean(asd))
            pa_list.append(PA)

        metric_list = metric_list / len(valset)

        dice_list = [row[0] for row in metric_list]
        jaccard_list = [row[1] for row in metric_list]
        hd95_list = [row[2] for row in metric_list]
        asd_list = [row[3] for row in metric_list]

        m_dice = np.nanmean(dice_list)
        m_jaccard = np.nanmean(jaccard_list)
        m_hd95 = np.nanmean(hd95_list)
        m_asd = np.nanmean(asd_list)
        m_pa = np.nanmean(pa_list)

        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                              metric_list[class_i, 0], iters)
            writer.add_scalar('info/val_{}_jac'.format(class_i + 1),
                              metric_list[class_i, 1], iters)
            writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                              metric_list[class_i, 2], iters)
            writer.add_scalar('info/val_{}_asd'.format(class_i + 1),
                              metric_list[class_i, 3], iters)

        is_best = m_dice * 100.0 > previous_best_dice
        previous_best_dice = max(m_dice * 100.0, previous_best_dice)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_dice': previous_best_dice,
            'previous_best_acc': previous_best_acc * 100.0,
        }
        if is_best:
            logging.info('***** \033[31m best eval! \033[0m *****')

            previous_best_acc = m_pa * 100.0
            save_mode_path = os.path.join(snapshot_path,
                                          'epoch_{}_dice_{}_jac_{}_pa_{}.pth'.format(
                                              epoch, round(previous_best_dice, 2),
                                              round(m_jaccard * 100.0, 2), round(previous_best_acc, 2)))
            torch.save(checkpoint, save_mode_path)
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        logging.info('***** \033[33m eval! \033[0m *****')

        dsc, std_dsc, ci_lower_dsc, ci_upper_dsc = compute_confidence_interval(m_dice_list)
        jac, std_jac, ci_lower_jac, ci_upper_jac = compute_confidence_interval(m_jaccard_list)
        hd95, std_hd95, ci_lower_hd95, ci_upper_hd95 = compute_confidence_interval(m_hd95_list)
        asd, std_asd, ci_lower_asd, ci_upper_asd = compute_confidence_interval(m_asd_list)
        pa, std_pa, ci_lower_pa, ci_upper_pa = compute_confidence_interval(pa_list)

        logging.info(
            'iteration %d : DSC : %.2f Jac : %.2f HD95 : %.2f ASD : %.2f PA : %.2f' % (
                iters, dsc * 100.0, jac * 100.0, hd95, asd, pa * 100.0))

        logging.info(f"dsc: {dsc * 100.0:.2f}, dsc_std: ({std_dsc:.2f}), "
                     f"95% CI: ({ci_lower_dsc * 100.0:.2f}, {ci_upper_dsc * 100.0:.2f})")
        logging.info(f"jac: {jac * 100.0:.2f}, jac_std: ({std_jac:.2f}), "
                     f"95% CI: ({ci_lower_jac * 100.0:.2f}, {ci_upper_jac * 100.0:.2f})")
        logging.info(f"hd95: {hd95:.2f}, hd95_std: ({std_hd95:.2f}), "
                     f"95% CI: ({ci_lower_hd95:.2f}, {ci_upper_hd95:.2f})")
        logging.info(f"asd: {asd:.2f}, asd_std: ({std_asd:.2f}), "
                     f"95% CI: ({ci_lower_asd:.2f}, {ci_upper_asd:.2f})")
        logging.info(f"pa: {pa * 100.0:.2f}, PA_std: ({std_pa:.2f}), "
                     f"95% CI: ({ci_lower_pa * 100.0:.2f}, {ci_upper_pa * 100.0:.2f})")


if __name__ == '__main__':
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args)
