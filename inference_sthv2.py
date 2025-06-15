import os
import torch
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
import numpy as np
import json
import random

import torch_npu
from torch_npu import *
from torch_npu.contrib import transfer_to_npu

from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
from PIL import Image

from laq_model import LatentActionQuantization
torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)




print(dir(torch_npu))

# Argument parsing
parser = argparse.ArgumentParser(description='Process some integers.')
# input file should already contain instruction and vision fields (which is the output of vqgan model)
parser.add_argument('--input_file', type=str, required=True, help='Path to the original data file.')
parser.add_argument('--dist_number', type=int, required=True, help='Distribution number')
parser.add_argument('--codebook_size', type=int, required=True, help='Codebook size')
parser.add_argument('--laq_checkpoint', type=str, required=True, help='Path to the laq checkpoint')
parser.add_argument('--divider', type=int, required=False, default=1, help='Divider')
parser.add_argument('--window_size', type=int, required=True, help='Window size')
parser.add_argument('--code_seq_len', type=int, required=True, help='Window size')
parser.add_argument('--layer', type=int, required=True, help='Layer')
parser.add_argument('--unshuffled_jsonl', type=str, required=True, help='Path to the unshuffled JSONL file')

args = parser.parse_args()

# Constants
dist_number = args.dist_number
batch_size = 32

cnt = 0
# Load processed JSONL data
processed_jsonl_data = []
print("input_file", args.input_file)

with open(args.input_file, 'r') as file:
    for line in file:
        processed_jsonl_data.append(json.loads(line))

print(f"processed_jsonl_data: {len(processed_jsonl_data)}")


frame_root = "./epic_kitchen"###

window_size = args.window_size
image_paths = []
whole_action_list = []

# TODO: make a separate file that have folder length as value

# v3 两层 帧序号不需要从0开始
file_length_dict = {}
frame_name_dict = {}

for i, elem in enumerate(processed_jsonl_data):
    image_rel_path = elem['image']  # 例：images/P01_01_P01_01_0/frame_000009.jpg
    full_image_path = os.path.join(frame_root, image_rel_path)

    folder = os.path.dirname(full_image_path)  # 例：.../images/P01_01_P01_01_0
    rel_folder = os.path.dirname(image_rel_path)  # 相对路径 images/P01_01_P01_01_0

    # 只扫描一次每个子文件夹
    if folder not in frame_name_dict:
        all_frames = sorted(
            [f for f in os.listdir(folder) if f.endswith(".jpg")]
        )
        frame_name_dict[folder] = all_frames
        file_length_dict[folder] = len(all_frames)

    filename = os.path.basename(full_image_path)  # 如 frame_000009.jpg
    frame_list = frame_name_dict[folder]

    if filename not in frame_list:
        print(f"[Warning] {filename} not found in {folder}")
        continue

    current_index = frame_list.index(filename)
    next_index = min(current_index + window_size, len(frame_list) - 1)
    next_frame_name = frame_list[next_index]

    # 重新拼接相对路径用于训练
    next_image_rel_path = os.path.join(rel_folder, next_frame_name)

    image_paths.append([image_rel_path, next_image_rel_path])

unshuffled_jsonl = args.unshuffled_jsonl

laq = LatentActionQuantization(
    dim=1024,
    quant_dim=32,
    codebook_size=args.codebook_size,
    image_size=256,
    patch_size=32,
    spatial_depth=args.layer,
    temporal_depth=args.layer,
    dim_head=64,
    heads=16,
    code_seq_len=args.code_seq_len,
).to("npu")
#laq.load(args.laq_checkpoint)

# Define a function to load images asynchronously
'''def load_image(file_paths):
    img1 = Image.open(file_paths[0])
    img2 = Image.open(file_paths[1])
    return img1, img2'''

frame_root = "./epic_kitchen"

def load_image(file_paths):
    full_path1 = os.path.join(frame_root, file_paths[0])
    full_path2 = os.path.join(frame_root, file_paths[1])
    img1 = Image.open(full_path1)
    img2 = Image.open(full_path2)
    return img1, img2


# Create a thread pool
executor = ThreadPoolExecutor(max_workers=16)
lock = threading.Lock()

class AsyncImageDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        future = executor.submit(load_image, self.file_paths[index])
        img1, img2 = future.result()
        with lock:
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
        return torch.cat([img1.unsqueeze(1), img2.unsqueeze(1)], dim=1)



image_size = 256
transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((256, 256)),
            T.ToTensor()
        ])



# Process data function
def process_data(processed_jsonl_data, laq, transform, image_paths, batch_size):
    cnt2 = 0
    dataset = AsyncImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)
    
    for img_batch in tqdm(dataloader):
        final_list = []
        with torch.no_grad():
            index_batch = laq(img_batch.to("npu"), return_only_codebook_ids=True)
        print("Batch indices:", index_batch.shape)
        
        index = 0
        for idx in range(batch_size * cnt2, min(batch_size * (cnt2 + 1), len(image_paths)), 1):
            elem_dict = {}
            elem_dict['image'] = image_paths[idx][0]
            #elem_dict['vision'] = processed_jsonl_data[idx]['vision']
            elem_dict['instruction'] = processed_jsonl_data[idx]['instruction']
            elem_dict['delta'] = [str(i) for i in index_batch[index].tolist()]
            elem_dict['fields'] = "[instruction],[vision],delta"
            final_list.append(elem_dict)
            index += 1
        cnt2 += 1
        yield final_list

        
parent_dir = os.path.dirname(unshuffled_jsonl)
if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)

# Write unshuffled JSONL data
with open(unshuffled_jsonl, 'w') as file:
    cnt = 0
    for entry in process_data(processed_jsonl_data, laq, transform, image_paths, batch_size):
        for elem in entry:
            file.write(json.dumps(elem) + '\n')
        cnt += 1


