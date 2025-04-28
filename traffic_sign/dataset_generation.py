import os
import shutil
import pandas as pd
import re
import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files('meowmeowmeowmeowmeow/gtsrb-german-traffic-sign', path="dataset", unzip=True)
kaggle.api.dataset_download_files('ahemateja19bec1025/traffic-sign-dataset-classification', path="dataset", unzip=True)

base_dir = os.path.abspath("dataset")

dataset1_root = os.path.join(base_dir, "Train")
labels1_path = os.path.join("sign_dic.csv")

dataset2_root = os.path.join(base_dir, "traffic_Data", "DATA")
labels2_path = os.path.join(base_dir, "labels.csv")

output_dir = os.path.join(base_dir, "merged_data")
os.makedirs(output_dir, exist_ok=True)

print(f"Dataset 1 Root: {dataset1_root}")
print(f"Dataset 2 Root: {dataset2_root}")
print(f"Output Directory: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

def extract_speed(desc):
    match = re.search(r'\d+', desc)
    return int(match.group()) if match else None

df1 = pd.read_csv(labels1_path)
global_labels = {}
description_to_global = {}
max_global_id = -1

for _, row in df1.iterrows():
    class_id = str(row['id']).strip()
    desc = row['description'].strip()
    global_labels[class_id] = desc
    description_to_global[desc] = class_id
    current_id = int(class_id)
    if current_id > max_global_id:
        max_global_id = current_id

df2 = pd.read_csv(labels2_path)
dataset2_mapping = {}
next_id = max_global_id + 1

for _, row in df2.iterrows():
    class_id = str(row['ClassId']).strip()
    desc = row['Name'].strip()
    speed = extract_speed(desc)
    global_id = None

    if desc in description_to_global:
        global_id = description_to_global[desc]
    else:
        global_id = str(next_id)
        global_labels[global_id] = desc
        description_to_global[desc] = global_id
        next_id += 1

    dataset2_mapping[class_id] = global_id

for class_folder in os.listdir(dataset1_root):
    src_dir = os.path.join(dataset1_root, class_folder)
    if not os.path.isdir(src_dir):
        continue
    dest_dir = os.path.join(output_dir, class_folder)
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy(src, dst)

for class_folder in os.listdir(dataset2_root):
    src_dir = os.path.join(dataset2_root, class_folder)
    if not os.path.isdir(src_dir):
        continue
    global_id = dataset2_mapping.get(class_folder, None)
    if global_id is None:
        continue
    dest_dir = os.path.join(output_dir, global_id)
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy(src, dst)

global_df = pd.DataFrame(list(global_labels.items()), columns=['id', 'description'])
global_df = global_df.sort_values(by='id')
global_csv_path = os.path.join(output_dir, 'global_labels.csv')
global_df.to_csv(global_csv_path, index=False)

print("Datasets merged successfully. Global labels saved to:", global_csv_path)