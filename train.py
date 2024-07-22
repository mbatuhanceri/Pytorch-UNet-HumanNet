import cv2
import math
from statistics import mode
from data import HumanDataset
from utils import DiceLoss
import torch
from torch.utils.data import DataLoader
from model import UNet
import os
import matplotlib.pyplot as plt
import numpy as np

epochs = 50
batch_size = 3
dataset_path = r"C:\Users\bahad\Desktop\dance_dataset"
test_data_path = r"C:\Users\bahad\Desktop\dance_dataset"


def train():
    device = torch.device('cuda:0')
    dataset = HumanDataset(dataset_path, "train", 3)

    testset = HumanDataset(test_data_path, "val", 3)
    model = UNet().to(device)

    if not os.path.exists('models'):
        os.mkdir('models')

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=0.0000001,
                                                           verbose=True)
    best_loss = math.inf
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=batch_size)

    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        print(f">> CURRENT EPOCH: {epoch + 1} <<<")

        epoch_loss = 0
        iterations = 0

        model.train()

        pos_weight = 0
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        for i, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, target)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            input_sample = cv2.resize(input[0].cpu().numpy().transpose(1, 2, 0), dsize=(640, 720))
            target_sample = cv2.resize(target[0].cpu().numpy().transpose(1, 2, 0) * 255, dsize=(640, 720))
            output_sample = cv2.resize(output[0].cpu().detach().numpy().clip(0, 255).transpose(1, 2, 0) * 255,
                                       dsize=(640, 720))
            output_sample[output_sample > 0] = 255

            cv2.imshow("input", input_sample)
            cv2.imshow("target", target_sample)
            cv2.imshow("output", output_sample)
            cv2.waitKey(1)

            iterations += batch_size

        print("train_loss " + str(epoch_loss / iterations))
        train_loss.append(epoch_loss / len(dataloader))
        scheduler.step(epoch_loss)

        tem_val_loss = 0
        for i, (input, target) in enumerate(testloader):
            with torch.no_grad():
                input = input.to(device)
                target = target.to(device)
                output = model(input)
                loss = criterion(output, target)
                tem_val_loss += loss.item()
        test_loss.append(tem_val_loss / len(testloader))

        torch.save(model.state_dict(), '.\\models\\last.pt')

        if epoch_loss < best_loss:
            torch.save(model.state_dict(), '.\\models\\best_4-64.pt')

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.show()


if __name__ == "__main__":
    train()
