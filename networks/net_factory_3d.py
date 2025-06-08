try:
    from unet_3D import unet_3D
    from vnet import (VNet, VNet_CRKD)
except:
    from .unet_3D import unet_3D
    from .vnet import (VNet, VNet_CRKD)

def net_factory_3d(net_type="vnet", in_chns=1, class_num=2,
                   has_dropout=True, mode="train", pert_gap=0.5, pert_type='dropout'):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=class_num, in_channels=in_chns).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=has_dropout).cuda()
    elif net_type == "vnet_crkd":
        net = VNet_CRKD(n_channels=in_chns, n_classes=class_num,
                   normalization='batchnorm', has_dropout=has_dropout).cuda()
    else:
        net = None
    return net
