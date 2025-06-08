import glob
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.net_factory_3d import net_factory_3d
from util.losses import dice_loss
from util import losses
from dataset.LA_3D import (
    LAHeart,
    RandomCrop,
    RandomRotFlip,
    ToTensor
)
from util.utils import test_all_case
from util import ramps
import time

import yaml
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str,
                    default="configs/LA.yaml")  # LA.yaml
# model name and save path
parser.add_argument('--method', default="FS", type=str,
                    help='method name')
parser.add_argument('--model', type=str, default='vnet',
                    help='net')
parser.add_argument("--label-num", type=int, default=4,
                    help="4(5%), 8(10%), 16(20%), 80(100%)")
# default sets
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")

args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

def getLog(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/logging.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    consistency = cfg['semi']['consistency']
    consistency_rampup = cfg['semi']['consistency_rampup']
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def main(args):
    root_path = cfg['data']['root_path']
    train_data_path = cfg['data']['root_path']
    exp = "exp/{}/{}".format(
        cfg['data']['dataset'], args.method)

    lb_rat = str(args.label_num)
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

    patch_size = cfg['data']['patch_size']
    in_ch = cfg['data']['in_chns']
    # train params
    base_lr = cfg['train']['base_lr']
    num_classes = cfg['train']['num_classes']
    batch_size = cfg['train']['batch_size']
    epochs = cfg['train']['epochs']
    max_iterations = cfg['train']['max_iterations']
    opt = cfg['train']['optimizer']
    # semi params
    ema_decay = cfg['semi']['ema_decay']
    conf_thresh = cfg['semi']['conf_thresh']

    # crossmatch
    LABELED_ID_NUM = args.label_num  # 8 or 16

    # label and unlabel
    unlabel_id = train_data_path + f"/train_{LABELED_ID_NUM}_unlabel.list"
    if args.label_num == 80:
        label_id = train_data_path + f"/train.list"
    else:
        label_id = train_data_path + f"/train_{LABELED_ID_NUM}_label.list"

    model = net_factory_3d(net_type=args.model, in_chns=in_ch, class_num=num_classes).cuda()
    model = model.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainset_l = LAHeart(
        base_dir=train_data_path,
        mode="train_l",
        # num=80 - LABELED_ID_NUM,
        transform=transforms.Compose(
            [
                RandomRotFlip(),
                RandomCrop(patch_size),
                ToTensor(),
            ]
        ),
        id_path=label_id,
    )
    trainsampler_l = torch.utils.data.sampler.RandomSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_l,
        worker_init_fn=worker_init_fn
    )

    model.train()

    if opt == "SGD":
        optimizer = optim.SGD(
            model.parameters(), lr=base_lr, momentum=0.9, weight_decay= 0.0001
        )
    elif opt == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=base_lr, weight_decay=0.0001
        )
    elif opt == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=0.0001
        )
    else:
        raise NotImplementedError

    pervious_bset_dice = 0.0
    iter_num = 0
    max_epoch = max_iterations // len(trainloader_l) + 1
    logging.info(f"Total Epochs: {max_epoch}, resuming from iteration: {iter_num}")
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(n_classes=num_classes)

    # 自动恢复训练：检查是否已有模型文件
    if os.path.exists(snapshot_path):
        list_of_files = glob.glob(f'{snapshot_path}/iter_*.pth')
        if len(list_of_files) > 0:
            latest_file = max(list_of_files, key=os.path.getctime)
            logging.info(f"Found existing model: {latest_file}. Loading...")
            checkpoint = torch.load(latest_file)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            iter_num = checkpoint['iter_num']
            pervious_bset_dice = checkpoint['pervious_bset_dice']
            logging.info(f"Resuming from iteration {iter_num}, best dice: {pervious_bset_dice}")
        else:
            logging.info("No saved model found. Starting from scratch.")
    else:
        os.makedirs(snapshot_path)
        logging.info("Snapshot path not found. Creating new directory.")


    for epoch_num in tqdm(range(max_epoch), ncols=70):

        # time1 = time.time()
        model.train()

        for i_batch, sampled_batch in enumerate(trainloader_l):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = 0.5 * (loss_dice + loss_ce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                # evals
                model.eval()
                with torch.no_grad():
                    with open(root_path + "/test.list", "r") as f:
                        image_list = f.readlines()
                    image_list = [
                        root_path+ '/' + item.replace("\n", "") + "/mri_norm2.h5"
                        for item in image_list
                    ]

                    dice, jc, hd, asd = test_all_case(
                        model,
                        image_list,
                        num_classes=num_classes,
                        patch_size=patch_size,
                        stride_xy=18,
                        stride_z=4,
                        save_result=False,
                        test_save_path=None,
                    )

                    dice = dice * 100
                    jc = jc * 100

                    logging.info('***** \033[33m pre best DSC: {:.2f} \033[0m *****'.format(pervious_bset_dice))

                    if dice > pervious_bset_dice:
                        logging.info('***** \033[31m best eval! DSC: {:.2f} \033[0m *****'.format(dice))
                        pervious_bset_dice = dice
                        save_mode_path = os.path.join(snapshot_path,
                                                      'iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(pervious_bset_dice, 2)))
                        save_best = os.path.join(snapshot_path, 'best_model.pth')
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'pervious_bset_dice': pervious_bset_dice,
                        }, save_mode_path)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'pervious_bset_dice': pervious_bset_dice,
                        }, save_best)

                        logging.info("save model to {}".format(save_mode_path))

                    logging.info(
                        'iteration %d : DSC : %.2f Jac : %.2f HD95 : %.2f ASD : %.2f' % (
                            iter_num, dice, jc, hd, asd))

                    model.train()

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "iter_" + str(iter_num) + ".pth"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'pervious_bset_dice': pervious_bset_dice,
                }, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                print("finish training, iter_num > max_iterations")
                break
            # time1 = time.time()
        if iter_num > max_iterations:
            print("finish training")
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":

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

