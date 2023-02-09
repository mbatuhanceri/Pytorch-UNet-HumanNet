from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import numpy as np
import cv2
import glob
from pathlib import Path



class HumanDataset(Dataset):
        def __init__(self,source_path, dtype, input_channels):
            self.input_channels = input_channels
            self.source_path = source_path

            self.source_images = sorted(glob.glob(f"{source_path}/{dtype}/A/*"))
            self.source_targets = sorted(glob.glob(f"{source_path}/{dtype}/OUT/*"))
            
            self.to_tensor = transforms.ToTensor()
        
        def __len__(self):
            return len(self.source_images)
              
        
        def __getitem__(self,idx):
            image_path = self.source_images[idx]
            mask_path = self.source_targets[idx]

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path)
            
            image = cv2.resize(image, (1280, 720))
            mask = cv2.resize(mask, (1280, 720))

            if self.input_channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[mask > 0] = 255

            # cv2.imshow("image", image)
            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)

            image = self.to_tensor(image)
            mask = self.to_tensor(mask)
            
            return (image,mask)


if __name__ == "__main__":
    dataset = HumanDataset(r"D:\Projects\unet2\carvana_dataset", "train",1)
    dataset.__getitem__(5)