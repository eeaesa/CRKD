#!/bin/bash

config_isic2018=configs/isic2018.yaml
config_KvasirSEG=configs/KvasirSEG.yaml

i=0.05
python train_2d_CRKD.py --config $config_isic2018 --label-rat $i

config_ACDC=configs/ACDC.yaml

i=3
python train_2d_CRKD.py --config $config_ACDC --labeled-num $i
