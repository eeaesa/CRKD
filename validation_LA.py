import argparse
import logging
import os
import random

import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.LA_3D import LAHeart_CRKD, ToTensor
from networks.net_factory_3d import net_factory_3d
from util.utils import test_all_case

def validate(args):
    # Load configuration
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    # Setup paths and logging
    root_path = cfg['data']['root_path']
    exp = "exp/{}/{}".format(cfg['data']['dataset'], args.method)
    snapshot_path = "{}/{}/{}_labeled".format(exp, args.model, str(args.label_num))
    
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    logging.basicConfig(filename=os.path.join(snapshot_path, "validation_log.txt"), 
                       level=logging.INFO,
                       format='[%(asctime)s.%(msecs)03d] %(message)s', 
                       datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Validation configuration:\n{yaml.dump(cfg)}")
    
    # Model parameters
    patch_size = cfg['data']['patch_size']
    in_ch = cfg['data']['in_chns']
    num_classes = cfg['train']['num_classes']
    
    # Initialize model
    model = net_factory_3d(net_type=args.model, in_chns=in_ch, class_num=num_classes).cuda()
    
    # Load trained model
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded model from {args.model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    
    model.eval()
    
    # Prepare test dataset
    test_list_path = os.path.join(root_path, "test.list")
    with open(test_list_path, "r") as f:
        image_list = f.readlines()
    image_list = [os.path.join(root_path, item.strip(), "mri_norm2.h5") for item in image_list]
    
    # Run validation
    logging.info("Starting validation...")
    with torch.no_grad():
        dice, jaccard, hd95, asd = test_all_case(
            model,
            image_list,
            num_classes=num_classes,
            patch_size=patch_size,
            stride_xy=18,
            stride_z=4,
            save_result=args.save_results,
            test_save_path=args.save_dir if args.save_results else None,
        )
    
    # Convert to percentages
    dice *= 100
    jaccard *= 100
    
    # Log results
    logging.info("\n***** Validation Results *****")
    logging.info(f"Dice Coefficient: {dice:.2f}%")
    logging.info(f"Jaccard Index: {jaccard:.2f}%")
    logging.info(f"HD95: {hd95:.2f}")
    logging.info(f"ASD: {asd:.2f}")
    
    return {
        'dice': dice,
        'jaccard': jaccard,
        'hd95': hd95,
        'asd': asd
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/LA.yaml")
    parser.add_argument('--method', default="CRKD", type=str, help='method name')
    parser.add_argument('--model', type=str, default='vnet_crkd', help='network model')
    parser.add_argument("--label-num", type=int, default=4, help="label number used in training")
    parser.add_argument("--model-path", type=str, required=True, help="path to trained model checkpoint")
    parser.add_argument("--save-results", action="store_true", help="save prediction results")
    parser.add_argument("--save-dir", type=str, default="./predictions", help="directory to save predictions")
    parser.add_argument("--deterministic", type=int, default=1, help="whether use deterministic training")
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    
    args = parser.parse_args()
    
    # Set up deterministic/reproducible settings
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
    
    # Run validation
    results = validate(args)
    print("\nValidation completed with results:")
    print(f"Dice: {results['dice']:.2f}%")
    print(f"Jaccard: {results['jaccard']:.2f}%")
    print(f"HD95: {results['hd95']:.2f}")
    print(f"ASD: {results['asd']:.2f}")