from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import itertools
from torch.utils.data.sampler import Sampler


class ACDCDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train_l",
        num=None,
        id_path=None,
        nsample=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train_l" or self.split == "train_u":
            with open(id_path, "r") as f:
                self.ids = f.read().splitlines()
            if self.split == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]

        elif self.split == "val":
            with open('splits/acdc/valtest.txt', 'r') as f:
            # with open('splits/acdc/val.txt', 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self._base_dir, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        sample = {"image": img, "label": mask}

        if self.split == "train_l" or self.split == "train_u":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = item
        return sample

    def __len__(self):
        return len(self.ids)


class WeakStrongAugment_CRKD(object):
    '''
    image: ndarray of shape (H, W)
    label: ndarray of shape (H, W)
    '''

    def __init__(self, output_size):

        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        image = self.resize(image)
        label = self.resize(label)
        image_weak = deepcopy(image)
        image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.from_numpy(np.array(label)).long()

        ########## weak augmentation is rotation / flip ##########
        image_weak = Image.fromarray((image_weak * 255).astype(np.uint8))
        img_strong = deepcopy(image_weak)
        image_weak = torch.from_numpy(np.array(image_weak)).unsqueeze(0).float() / 255.0

        ########## strong augmentation is color jitter ##########
        if random.random() < 0.8:
            img_strong = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_strong)
        img_strong = blur(img_strong, p=0.5)
        img_strong = torch.from_numpy(np.array(img_strong)).unsqueeze(0).float() / 255.0

        sample = {
            "image": image,
            "label": label,
            "image_weak": image_weak,
            "image_strong": img_strong,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size / x, self.output_size / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size / x, self.output_size / y), order=0)
        label = zoom(label, (self.output_size / x, self.output_size / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # image = torch.from_numpy(image).unsqueeze(0).float()
        # label = torch.from_numpy(np.array(label)).long()
        sample = {"image": image, "label": label}
        return sample

def random_rot_flip1(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rot_flip(img, mask):
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    img = np.flip(img, axis=axis).copy()
    mask = np.flip(mask, axis=axis).copy()
    return img, mask


def random_rotate(img, mask):
    angle = np.random.randint(-20, 20)
    img = ndimage.rotate(img, angle, order=0, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return img, mask


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size / x, self.output_size / y), order=0)

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img
