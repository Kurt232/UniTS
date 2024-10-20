import json
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
import torch

from models.units import UniTSPretrainedModel

device = 'cuda'

load_path = '/data/wenhao/wjdu/realworld_thigh/UniTS_HEAD_8_4s/checkpoint-99.pth'
save_path = '/data/wenhao/wjdu/realworld_thigh/results/'

num_class = 7
test_paths = ["/data/wenhao/wjdu/data/realworld_thigh_4s/test_thigh_1.json"]

os.makedirs(save_path, exist_ok=True)

def main():
    # define the model
    model = UniTSPretrainedModel(d_enc_in=6, num_class=num_class)
    if load_path is not None and os.path.exists(load_path):        
        pretrained_mdl = torch.load(load_path, map_location='cpu')
        msg = model.load_state_dict(pretrained_mdl['model'], strict=False)
        print(msg)

    model.to(device) # device is cuda
    # set trainable parameters

    data_list = []
    print("Dataset:")
    for i, meta_path in enumerate(test_paths):
        print(f"\t{i}. {meta_path.split('/')[-1]}")
        meta_l = json.load(open(meta_path))
        data_list.append(meta_l)
    
    # mapping = {l: i for i, l in enumerate(labels)}
    ['downstairs', 'jog', 'lie', 'sit', 'stand', 'upstairs', 'walk']
    mapping = {
        'downstairs': 0,
        'jog': 1,
        'lie': 2,
        'sit': 3,
        'stand': 4,
        'upstairs': 5,
        'walk': 6
    }

    _mapping = {v: k for k, v in mapping.items()}

    acc_total = {}
    for data_item, data_path in zip(data_list, test_paths):
        predictions = []
        correct_pred = 0
        
        with torch.no_grad():
            for data in tqdm(data_item, desc=f"Testing ..."):
                imu_input = torch.tensor(data['imu_input'], dtype=torch.float32)
                label = mapping[data['output']]

                imu_input = imu_input.unsqueeze(0).to(device, non_blocking=True)
                    
                output = model(imu_input) # [1, 5]

                # Calculate accuracy
                _, pred_index = torch.max(output, 1)
                if pred_index.item() == label:
                    correct_pred += 1

                predictions.append({'pred': _mapping[pred_index.item()], 'ref': data['output'], 'data_id': data['data_id']})

        result_file = load_path.split('/')[-2] + '_' + data_path.split('/')[-1]
        prediction_file = os.path.join(save_path, result_file)
        json.dump(predictions, open(prediction_file, 'w'), indent=2)

        print(f"{result_file} ", "Accuracy: {:.4f}%".format(correct_pred / len(data_item) * 100))
        acc_total[result_file] = correct_pred / len(data_item)
    
    print(json.dumps(acc_total, indent=2))

if __name__ == '__main__':
    main()