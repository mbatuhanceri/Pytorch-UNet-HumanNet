import cv2
import glob
from model import UNet
import torch
from data import HumanDataset
import numpy as np
from torch.utils.data import DataLoader

test_path = r"D:\Projects\Datasets\WHU"
dataset = HumanDataset(test_path, "test", 3)


def test():
    device = torch.device('cuda:0')
    model = UNet().to(device)

    model.load_state_dict(torch.load(r"D:\Projects\Projects\eartqueke\models\best.pt"))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, (input, target) in enumerate(dataloader):        
        input = input.to(device)
        output = model(input)

        output_sample = cv2.resize(output[0].cpu().detach().numpy().clip(0,255).transpose(1, 2, 0), dsize=(640, 720))
        input_sample = cv2.resize(input[0].cpu().numpy().transpose(1, 2, 0), dsize=(640, 720))
        target_sample = cv2.resize(target[0].cpu().numpy().transpose(1, 2, 0), dsize=(640, 720))

        output_sample = cv2.cvtColor(output_sample, cv2.COLOR_GRAY2BGR)
        target_sample = cv2.cvtColor(target_sample, cv2.COLOR_GRAY2BGR)

        grid_image = np.append(arr=output_sample, values=target_sample, axis=1)
        grid_image = np.append(arr=grid_image, values=input_sample, axis=1)

        cv2.imshow("frame", grid_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    test()