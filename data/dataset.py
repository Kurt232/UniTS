import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import cv2

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
def transform_train(imu_data: list):
    imu_data = torch.tensor(imu_data, dtype=torch.float32)
    return imu_data


class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            meta_l = json.load(open(meta_path))
            print(f"{meta_path}: len {len(meta_l)}")
            ann += meta_l
        self.ann = ann
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item.keys():
            filename = data_item['image']
            question = data_item['instruction']
            answer = data_item['output']
     
            image = cv2.imread(filename)
            image = Image.fromarray(image)
            image = self.transform(image)
            format_instruction = question
            format_input = None
        else:
            image = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        data_list = []
        for meta_path in self.config['META']:
            meta_l = json.load(open(meta_path))
            print(f"{meta_path}: len {len(meta_l)}")
            data_list += meta_l

        self.data_list = data_list
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        imu_data, caption = sample['imu_input'], sample['output']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        imu_input = self.transform(imu_data)

        format_instruction = sample['instruction']
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, imu_input
