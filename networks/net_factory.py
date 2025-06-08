try:
    from unet import (UNet, UNet_KD)
except:
    from .unet import (UNet, UNet_KD)

def net_factory(net_type="unet", in_chns=3, class_num=2,
                has_dropout=True, ema=False):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_KD":
        net = UNet_KD(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
