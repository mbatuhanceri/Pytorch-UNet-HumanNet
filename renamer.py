import os 
import cv2

source_path = r"carvana_dataset\train\masks"
target_path = r"carvana_dataset\train\masks"
for file in os.listdir(source_path):
    print(file)
    splitted_name = file.split('.')
    name = splitted_name[0]

    image_folder = os.path.join(source_path, file)

    target_folder = os.path.join(target_path, name + ".jpg")

    os.rename(image_folder, target_folder)