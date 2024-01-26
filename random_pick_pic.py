import os
import shutil
import random
import csv
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def copy_or_select_images(source_folder, target_folder, num_images=150):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for folder in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder)
        
        if os.path.isdir(folder_path):
            images = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            selected_images = images

            if len(images) > num_images:
                selected_images = random.sample(images, num_images)

            for image in selected_images:
                src_path = os.path.join(folder_path, image)
                dst_path = os.path.join(target_folder, image)
                shutil.copy2(src_path, dst_path)

def save_to_csv(target_folder, csv_file):
    headers = ["Id", "State", "black", "green", "arrow", "red", "yellow", "right"]
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers) 
        images = [img for img in os.listdir(target_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        for image in sorted(images, key=natural_sort_key):
            state = re.match(r"([a-z]+)_", image, re.I)
            state_formatted = "['{}']".format(state.group(1)) if state else ""

            if state.group(1) == 'black':
                writer.writerow([image[:-4], state_formatted, 1,0,0,0,0,0]) 
            elif state.group(1) == 'red':
                writer.writerow([image[:-4], state_formatted, 0,0,0,1,0,0]) 
            elif state.group(1) == 'yellow':
                writer.writerow([image[:-4], state_formatted, 0,0,0,0,1,0]) 
            elif state.group(1) == 'green':
                writer.writerow([image[:-4], state_formatted, 0,1,0,0,0,0]) 
            else:
                writer.writerow([image[:-4], state_formatted]) 

source_folder = 'croped'
target_folder = 'pick_150'
copy_or_select_images(source_folder, target_folder)
csv_file = 'selected_images.csv'
save_to_csv(target_folder, csv_file)
