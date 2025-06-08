#!/bin/bash

config_LA=configs/LA.yaml
lb_4=4
lb_8=8
lb_all=80

python train_3d_FS.py --config $config_LA --label-num $lb_4
python train_3d_CRKD.py --config $config_LA --label-num $lb_4
