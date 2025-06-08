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
from dataset.acdc import  ACDCDataSets, RandomGenerator
from util import losses

import time
from torchvision import transforms

from networks.net_factory import net_factory
import shutil
import yaml
from util.utils import count_params, test_single_volume, compute_confidence_interval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str,
                    default="configs/ACDC.yaml")
# model name and save path
parser.add_argument('--method', default="FS", type=str,
                    help='')
parser.add_argument('--model', type=str, default='unet',
                    help='')
# label rat
parser.add_argument('--labeled-num', default='3', type=str,
                    help='3 : 5%, 7 : 10%, 14 : 20%')
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

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[patiens_num]

def main(args):
    # save model and logging
    exp = "exp/{}/{}".format(
        cfg['data']['dataset'], args.method)

    lb_rat = args.labeled_num
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

    labeled_id_path = "splits/acdc/{}/labeled.txt".format(
        args.labeled_num)
    unlabeled_id_path = "splits/acdc/{}/unlabeled.txt".format(
        args.labeled_num)

    if args.labeled_num == "140":
        labeled_id_path = "splits/acdc/train_slices.txt"

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    model = net_factory(net_type=args.model, in_chns=in_ch,
                        class_num=num_classes)
    optimizer = SGD(model.parameters(), lr=base_lr,
                    momentum=0.9, weight_decay=0.0001)

    logging.info('Total params: {:.1f}M\n'.format(count_params(model)))

    model.cuda()

    ce_loss = nn.CrossEntropyLoss().cuda()
    dice_loss = losses.DiceLoss(num_classes)

    labeled_slice = patients_to_slices(root_path, args.labeled_num)

    db_train = ACDCDataSets(base_dir=root_path,
                              split="train_l",
                              id_path=labeled_id_path,
                              transform=transforms.Compose([
                                  RandomGenerator(crop_size)
                              ]
                              ))
    
    db_val = ACDCDataSets(base_dir=root_path, split="val")

    total_slices = len(db_train)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))

    trainloader = DataLoader(db_train, batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,num_workers=1,
                             drop_last=True,
                             worker_init_fn=worker_init_fn
                             )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    total_iters = len(trainloader) * epochs
    logging.info('Total iters is: {}\n'.format(total_iters))
    previous_best_mdice, previous_best_mhd95, previous_best_acc = 0.0, 0.0, 0.0
    epoch = -1
    iters = 0

    if os.path.exists(os.path.join(snapshot_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(snapshot_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_mdice = checkpoint['previous_best_mdice']
        previous_best_mhd95 = checkpoint['previous_best_mhd95']
        previous_best_acc = checkpoint['previous_best_acc']

        logging.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, epochs):
        logging.info(
            '===========> Epoch: {:}, LR: {:.5f}, \033[31m Previous best {} mdice: {:.2f}, mhd95: {:.2f}, Overall Accuracy: {:.2f}\033[0m'.format(
                epoch, optimizer.param_groups[0]['lr'], dataset, previous_best_mdice, previous_best_mhd95, previous_best_acc))

        for i, (sampled_batch) in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # supervised_loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = epoch * len(trainloader) + i

            lr = base_lr * (1 - iters / total_iters) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iters)

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logging.info('Iters: %d, Total loss: %f, loss ce: %f, loss dice: %f'%(
                    iters, loss.item(), loss_ce.item(), loss_dice.item()
                ))

        model.eval()
        metric_list = 0.0
        pa_list = []
        mdice_list = []
        mhd95_list = []
        mjaccard_list = []
        masd_list = []

        for i, sampled_batch in enumerate(valloader):
            metric_i, PA = test_single_volume(
                sampled_batch["image"], sampled_batch["label"],
                model,
                classes=cfg['train']['num_classes'],
                patch_size=cfg['data']['crop_size']
            )

            mdice = [row[0] for row in metric_i]
            mhd95 = [row[1] for row in metric_i]
            mjaccard = [row[2] for row in metric_i]
            masd = [row[3] for row in metric_i]

            metric_list += np.array(metric_i)
            mdice_list.append(np.mean(mdice))
            mhd95_list.append(np.mean(mhd95))
            mjaccard_list.append(np.mean(mjaccard))
            masd_list.append(np.mean(masd))
            pa_list.append(PA)

        metric_list = metric_list / len(db_val)

        dice_list = [row[0] for row in metric_list]
        hd95_list = [row[1] for row in metric_list]
        jaccard_list = [row[2] for row in metric_list]
        asd_list = [row[3] for row in metric_list]

        m_dice = np.nanmean(dice_list)
        m_jaccard = np.nanmean(jaccard_list)
        m_hd95 = np.nanmean(hd95_list)
        m_asd = np.nanmean(asd_list)
        m_pa = np.nanmean(pa_list)

        for class_i in range(num_classes - 1):
            writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                              metric_list[class_i, 0], iters)
            writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                              metric_list[class_i, 1], iters)
            writer.add_scalar('info/val_{}_jaccard'.format(class_i + 1),
                              metric_list[class_i, 2], iters)
            writer.add_scalar('info/val_{}_asd'.format(class_i + 1),
                              metric_list[class_i, 3], iters)

        writer.add_scalar('info/val_mean_dice', m_dice * 100.0, iters)
        writer.add_scalar('info/val_mean_hd95', m_hd95, iters)
        writer.add_scalar('info/val_mean_jaccard', m_jaccard * 100.0, iters)
        writer.add_scalar('info/val_mean_asd', m_asd, iters)
        writer.add_scalar('info/val_pa', m_pa * 100.0, iters)

        is_best = m_dice * 100.0 > previous_best_mdice
        previous_best_mdice = max(m_dice * 100.0, previous_best_mdice)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_mdice': previous_best_mdice,
            'previous_best_mhd95': previous_best_mhd95,
            'previous_best_acc': previous_best_acc * 100.0,
        }

        if is_best:
            logging.info('***** \033[31m best eval! \033[0m *****')
            previous_best_mhd95 = m_hd95
            previous_best_acc = m_pa * 100.0
            save_mode_path = os.path.join(snapshot_path,
                                          'epoch_{}_mDSC_{}_mJac_{}_mHD95_{}.pth'.format(
                                              epoch, round(previous_best_mdice, 2),
                                              round(m_jaccard * 100.0, 2),
                                              round(previous_best_mhd95, 2)))
            torch.save(checkpoint, save_mode_path)
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))

        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))

        logging.info('***** \033[33m eval! \033[0m *****')

        for (cls_idx, dice) in enumerate(dice_list):
            logging.info('***** Evaluation ***** >>>> Class [{:}] Dice: {:.2f}'.format(
                cls_idx, dice * 100.0))
        logging.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(m_dice * 100.0))

        for (cls_idx, jaccard) in enumerate(jaccard_list):  # 新增
            logging.info('***** Evaluation ***** >>>> Class [{:}] Jaccard: {:.2f}'.format(
                cls_idx, jaccard * 100.0))
        logging.info('***** Evaluation ***** >>>> MeanJaccard: {:.2f}\n'.format(m_jaccard * 100.0))

        for (cls_idx, hd95) in enumerate(hd95_list):
            logging.info('***** Evaluation ***** >>>> Class [{:}] hd95: {:.2f}'.format(
                cls_idx, hd95))
        logging.info('***** Evaluation ***** >>>> MeanHd95: {:.2f}\n'.format(m_hd95))

        for (cls_idx, asd) in enumerate(asd_list):  # 新增
            logging.info('***** Evaluation ***** >>>> Class [{:}] ASD: {:.2f}'.format(
                cls_idx, asd))
        logging.info('***** Evaluation ***** >>>> MeanASD: {:.2f}\n'.format(m_asd))

        logging.info('***** Evaluation ***** >>>> mPA: {:.2f}\n'.format(m_pa * 100.0))

        dsc, std_dsc, ci_lower_dsc, ci_upper_dsc = compute_confidence_interval(mdice_list)
        jac, std_jac, ci_lower_jac, ci_upper_jac = compute_confidence_interval(mjaccard_list)
        hd95, std_hd95, ci_lower_hd95, ci_upper_hd95 = compute_confidence_interval(mhd95_list)
        asd, std_asd, ci_lower_asd, ci_upper_asd = compute_confidence_interval(masd_list)
        pa, std_pa, ci_lower_pa, ci_upper_pa = compute_confidence_interval(pa_list)

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
