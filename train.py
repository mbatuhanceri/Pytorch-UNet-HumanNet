import cv2
import math
from statistics import mode
from data import HumanDataset
from utils import DiceBCELoss
import torch
from torch.utils.data import DataLoader
from model import UNet
import os

epochs = 50
batch_size = 2
dataset_path = r"human_ir_dataset"

def train():
    device = torch.device('cuda:0')
    dataset = HumanDataset(dataset_path, "train", 1)
    model = UNet().to(device)

    if not os.path.exists('models'):
        os.mkdir('models')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=0.0000001, verbose=True)
    best_loss = math.inf

    for epoch in range(epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
        
        epoch_loss = 0
        iterations = 0

        model.train()

        pos_weight = 0
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        
        criterion = DiceBCELoss()
        for i, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            loss = criterion(output, target)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            input_sample = cv2.resize(input[0].cpu().numpy().transpose(1, 2, 0), dsize=(640, 720))
            target_sample = cv2.resize(target[0].cpu().numpy().transpose(1, 2, 0), dsize=(640, 720))
            output_sample = cv2.resize(output[0].cpu().detach().numpy().clip(0,255).transpose(1, 2, 0), dsize=(640, 720))
            output_sample[output_sample > 0] = 255
            

            cv2.imshow("input", input_sample)
            cv2.imshow("target", target_sample)
            cv2.imshow("output", output_sample)
            cv2.waitKey(1)

            iterations += batch_size
        
        print("train_loss" + str(loss))
        scheduler.step(epoch_loss)
        torch.save(model.state_dict(), './models/last.pt')

        if epoch_loss < best_loss:
            torch.save(model.state_dict(), './models/best_4-64.pt')


if __name__ == "__main__":
    train()