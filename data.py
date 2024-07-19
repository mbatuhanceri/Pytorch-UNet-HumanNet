import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import numpy as np
import cv2
import glob
from pathlib import Path
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import random


class HumanDataset(Dataset):

    def sameRandomTrans(self, image, mask):
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __init__(self, source_path, dtype, input_channels):
        self.dtype = dtype
        self.input_channels = input_channels
        self.source_path = source_path

        self.source_images = sorted(glob.glob(f"{source_path}\\{dtype}\\images\\*"))
        self.source_targets = sorted(glob.glob(f"{source_path}\\{dtype}\\masks\\*"))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        image_path = self.source_images[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 480))
        if self.input_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)

        if self.dtype == "test":
            image = self.to_tensor(image)
            return image

        mask_path = self.source_targets[idx]
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (640, 480))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask / 255).astype('float32')
        mask = np.expand_dims(mask, axis=-1)

        if self.dtype == "train":
            image, mask = self.sameRandomTrans(image, mask)

            return image, mask

        else:
            return image, mask


if __name__ == "__main__":
    dataset = HumanDataset(r"C:\Users\bahad\intern\pytorch_unet_humannet\Datasets", "train", 1)
    image, mask = dataset.__getitem__(5)
    print(image.shape, mask.shape)
