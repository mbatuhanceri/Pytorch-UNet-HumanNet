import argparse
import collections
from pickletools import uint8
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from model import UNet


device = torch.device('cuda:0')
model = UNet().to(device)

model.load_state_dict(torch.load(r"models\best_4-64.pt"))
model.eval()

to_tensor = transforms.ToTensor()

cap = cv2.VideoCapture(r"klip2.mp4")

while True:
    ret, frame = cap.read()

    input_image = frame.copy()
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    input_image = cv2.resize(input_image, dsize=(1280, 720))
    input_tensor = to_tensor(input_image).unsqueeze(0).to(device)

    output = model(input_tensor)

    input_image = cv2.resize(input_image, dsize=(1920, 1080))
    output_image = cv2.resize(output[0].cpu().detach().numpy().clip(0,255).transpose(1, 2, 0), dsize=(1920, 1080))

    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    
    input_sample = cv2.resize(input_image.copy(), dsize=(1280,720))
    input_image[output_image > 0] = (150,100,0)
    output_image = cv2.resize(output_image, dsize=(640,480))
    

    
    cv2.imshow("input_window", input_image)
    cv2.imshow("input_sample_window", input_sample)
    cv2.imshow("output_window", output_image)
    cv2.waitKey(1)