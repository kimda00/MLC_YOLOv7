import os
import shutil
import random
import csv
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def copy_or_select_images(source_folder, target_folder, num_images=1000):
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

def extract_states(filename):
    parts = re.split(r'_', filename)
    states = [part for part in parts if not any(char.isdigit() for char in part)]
    return states

def save_to_csv(target_folder, csv_file):
    headers = ["Id", "State", "black", "green", "arrow", "red", "yellow", "right"]
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers) 
        images = [img for img in os.listdir(target_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        for image in sorted(images, key=natural_sort_key):
            states = extract_states(image)
            state_formatted = "['{}']".format("', '".join(states))

            row = [image[:-4], state_formatted] + [1 if state in states else 0 for state in headers[2:]]
            writer.writerow(row) 

source_folder = 'inference/croped'
target_folder = 'inference/pick_1000'
copy_or_select_images(source_folder, target_folder)
csv_file = 'inference/1000_selected_images.csv'
save_to_csv(target_folder, csv_file)
