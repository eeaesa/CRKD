# from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image, ImageOps, ImageFilter

class MyDataset(Dataset):
    def __init__(self,
                 name,
                 root,
                 mode,
                 size=None,
                 id_path=None,
                 nsample=None,
                 transform=None,
                 ops_weak=None,
                 ops_strong=None
    ):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            print("Train samples is {}".format(len(self.ids)))
            self.lb_num = deepcopy(self.ids)
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            # with open('splits/%s/val.txt' % name, 'r') as f:
            with open(self.root+'/val.txt', 'r') as f:
            # with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            print("Val samples is {}".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert("L")
        mask = np.array(mask)
        mask = mask / 255
        mask = Image.fromarray(mask.astype(np.uint8))
        sample = {"image": img, "label": mask}

        if self.mode == 'val':
            img, mask = toTensor(img, mask)
            return img, mask

        if self.mode == 'train_l' or self.mode == 'train_u':
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        return sample

class WeakStrongAugment_mocoV2(object):

    def __init__(self, output_size):

        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample["image"], sample["label"]
        image, label = resize_pil(image, label, self.output_size)

        # weak augmentation is random_crop、rotation / flip
        image, label = random_crop(image, label, self.output_size)
        image, label = random_rot_flip_pil(image, label)

        image_weak = deepcopy(image)
        image, label = toTensor(image, label)

        ########## weak augmentation ##########
        img_strong = deepcopy(image_weak)
        image_weak = toTensor(image_weak)

        ########## strong augmentation ##########
        img_strong = mocoV2_aug(img_strong)

        sample = {
            "image": image,
            "label": label,
            "image_weak": image_weak,
            "image_strong": img_strong,
        }
        return sample

def random_rot_flip_pil(image, label=None):
    k = random.randint(0, 3)
    image = image.rotate(k * 90)
    axis = random.randint(0, 1)
    if axis == 0:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if label is not None:
        label = label.rotate(k * 90)
        if axis == 0:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        return image, label
    return image

def random_crop(image, mask=None, crop_size=256):

    width, height = image.size
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    right = left + crop_size
    bottom = top + crop_size

    # Crop the image and mask
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image = cropped_image.resize((width, height), Image.BILINEAR)

    if mask is not None:
        cropped_mask = mask.crop((left, top, right, bottom))
        cropped_mask = cropped_mask.resize((width, height), Image.NEAREST)
        return cropped_image, cropped_mask

    return cropped_image


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # label = Image.fromarray(np.array(label))

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        # Resize both image and label
        image, label = resize_pil(image, label, self.output_size)
        image, label = toTensor(image, label)

        return {"image": image, "label": label}


class WeakStrongAugment(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # label = Image.fromarray(np.array(label))

        # Resize both image and label
        image, label = resize_pil(image, label, self.output_size)

        # Weak augmentation
        image_weak, label = random_rot_flip(image, label)

        # Strong augmentation
        image_strong = transforms.ColorJitter(
            0.8, 0.8, 0.8, 0.2)(image_weak)
        image_strong = transforms.ToTensor()(image_strong)

        # Convert to tensor
        image, label = toTensor(image, label)
        image_weak = transforms.ToTensor()(image_weak)

        return {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }


def random_rot_flip(image, label=None):
    k = random.randint(0, 3)
    image = image.rotate(k * 90)  # PIL 自带旋转功能
    axis = random.randint(0, 1)
    if axis == 0:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if label is not None:
        label = label.rotate(k * 90)
        if axis == 0:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        return image, label
    return image


def random_rotate(image, label):
    angle = random.randint(-20, 20)
    image = image.rotate(angle)
    label = label.rotate(angle) if label else None
    return image, label


def resize_pil(image, label=None, output_size=256):
    size = (output_size, output_size)
    image_resized = image.resize(size, Image.BILINEAR if image.mode == "RGB" else Image.NEAREST)

    if label is not None:
        # Resize label
        label_resized = label.resize(size, Image.NEAREST)
        return image_resized, label_resized

    return image_resized

def toTensor(image, label=None):
    image = transforms.Compose([
        transforms.ToTensor(),
    ])(image)
    if label is not None:
        label = torch.from_numpy(np.array(label)).long()
        return image, label
    return image


def mocoV2_aug(image, size=256):
    '''
    https://arxiv.org/pdf/2003.04297v1
    '''
    image = transforms.Compose([
        # transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco_GaussianBlur([0.1, 2.0])], p=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize_moco,
    ])(image)

    return image

class moco_GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x