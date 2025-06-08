import argparse
import logging
import os
import pprint
import random
import shutil
import sys

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.acdc import ACDCDataSets
from networks.net_factory import net_factory
from util.utils import test_single_volume, compute_confidence_interval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str,
                   default="configs/ACDC.yaml")
# model name and save path
parser.add_argument('--method', default="CRKD", type=str,
                   help='method name')
parser.add_argument('--model', type=str, default='unet_KD',
                   help='net')
# label rat
parser.add_argument('--labeled-num', default='3', type=str,
                   help='3 : 5%, 7 : 10%, 14 : 20%')
# seed
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument('--deterministic', type=int, default=1,
                   help='whether use deterministic training')
parser.add_argument('--model_path', type=str, required=True,
                   help='path to the trained model checkpoint')
args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

def getLog(args, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/logging.txt", level=logging.INFO,
                       format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

def validate(args):
    # Setup logging
    exp = "exp/{}/{}".format(
        cfg['data']['dataset'], args.method)
    lb_rat = args.labeled_num
    snapshot_path = "{}/{}/{}_labeled".format(
        exp, args.model, lb_rat)
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    getLog(args, snapshot_path)
    logging.info("Configuration settings:\n%s", yaml.dump(cfg))

    # Initialize model
    in_ch = cfg['data']['in_chns']
    num_classes = cfg['train']['num_classes']
    model = net_factory(net_type=args.model, in_chns=in_ch,
                       class_num=num_classes).cuda()
    
    # Load trained model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    logging.info('Model loaded from %s' % args.model_path)

    # Prepare validation dataset
    root_path = cfg['data']['root_path']
    valset = ACDCDataSets(base_dir=root_path, split="val")
    valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    # Start evaluation
    metric_list = 0.0
    pa_list = []
    mdice_list = []
    mhd95_list = []
    mjaccard_list = []
    masd_list = []

    with torch.no_grad():
        for i, sampled_batch in enumerate(valloader):
            metric_i, PA = test_single_volume(
                sampled_batch["image"], sampled_batch["label"],
                model,
                classes=num_classes,
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

    metric_list = metric_list / len(valset)

    dice_list = [row[0] for row in metric_list]
    hd95_list = [row[1] for row in metric_list]
    jaccard_list = [row[2] for row in metric_list]
    asd_list = [row[3] for row in metric_list]

    m_dice = np.nanmean(dice_list)
    m_jaccard = np.nanmean(jaccard_list)
    m_hd95 = np.nanmean(hd95_list)
    m_asd = np.nanmean(asd_list)
    m_pa = np.nanmean(pa_list)

    # Log results
    logging.info('***** Evaluation Results *****')
    for (cls_idx, dice) in enumerate(dice_list):
        logging.info('Class [{:}] Dice: {:.2f}'.format(cls_idx, dice * 100.0))
    logging.info('MeanDice: {:.2f}\n'.format(m_dice * 100.0))

    for (cls_idx, jaccard) in enumerate(jaccard_list):
        logging.info('Class [{:}] Jaccard: {:.2f}'.format(cls_idx, jaccard * 100.0))
    logging.info('MeanJaccard: {:.2f}\n'.format(m_jaccard * 100.0))

    for (cls_idx, hd95) in enumerate(hd95_list):
        logging.info('Class [{:}] HD95: {:.2f}'.format(cls_idx, hd95))
    logging.info('MeanHD95: {:.2f}\n'.format(m_hd95))

    for (cls_idx, asd) in enumerate(asd_list):
        logging.info('Class [{:}] ASD: {:.2f}'.format(cls_idx, asd))
    logging.info('MeanASD: {:.2f}\n'.format(m_asd))

    logging.info('Overall Accuracy: {:.2f}\n'.format(m_pa * 100.0))

    # Compute confidence intervals
    dsc, std_dsc, ci_lower_dsc, ci_upper_dsc = compute_confidence_interval(mdice_list)
    jac, std_jac, ci_lower_jac, ci_upper_jac = compute_confidence_interval(mjaccard_list)
    hd95, std_hd95, ci_lower_hd95, ci_upper_hd95 = compute_confidence_interval(mhd95_list)
    asd, std_asd, ci_lower_asd, ci_upper_asd = compute_confidence_interval(masd_list)
    pa, std_pa, ci_lower_pa, ci_upper_pa = compute_confidence_interval(pa_list)

    logging.info(f"Mean DSC: {dsc * 100.0:.2f} ± {std_dsc * 100.0:.2f}, "
                 f"95% CI: ({ci_lower_dsc * 100.0:.2f}, {ci_upper_dsc * 100.0:.2f})")
    logging.info(f"Mean Jaccard: {jac * 100.0:.2f} ± {std_jac * 100.0:.2f}, "
                 f"95% CI: ({ci_lower_jac * 100.0:.2f}, {ci_upper_jac * 100.0:.2f})")
    logging.info(f"Mean HD95: {hd95:.2f} ± {std_hd95:.2f}, "
                 f"95% CI: ({ci_lower_hd95:.2f}, {ci_upper_hd95:.2f})")
    logging.info(f"Mean ASD: {asd:.2f} ± {std_asd:.2f}, "
                 f"95% CI: ({ci_lower_asd:.2f}, {ci_upper_asd:.2f})")
    logging.info(f"Overall Accuracy: {pa * 100.0:.2f} ± {std_pa * 100.0:.2f}, "
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

    validate(args)