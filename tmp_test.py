import cv2
import glob
from model import UNet
import torch
from data import HumanDataset
import numpy as np
from torch.utils.data import DataLoader

test_path = r"C:\Users\bahad\Desktop\dance_dataset"
dataset = HumanDataset(test_path, "test", 3)

def test():
    device = torch.device('cuda:0')
    model = UNet().to(device)

    model.load_state_dict(torch.load(r"C:\Users\bahad\intern\pytorch_unet_humannet\venv\models\best_4-64.pt"))
    model.eval()

    alpha = 0.5
    beta = (1.0 - alpha)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, (input) in enumerate(dataloader):

        input = input.to(device)
        output = model(input)

        output_sample = cv2.resize(output[0].cpu().detach().numpy().transpose(1, 2, 0).astype('uint8'), dsize=(640, 720))
        output3ch = cv2.cvtColor(~output_sample, cv2.COLOR_GRAY2BGR)
        input_sample = cv2.resize((input[0].cpu().numpy().transpose(1, 2, 0)*255).astype('uint8'), dsize=(640, 720))
        transparent_sample = cv2.addWeighted(input_sample, alpha, output3ch, beta, 0.0)
        grid_image = np.hstack((input_sample, output3ch, transparent_sample))

        cv2.imwrite(f"C:\\Users\\bahad\intern\pytorch_unet_humannet\Datasets\collages\collage_{i}.jpeg", grid_image)
        cv2.imshow("frame", grid_image)
        cv2.waitKey(0)



if __name__ == "__main__":
    test()