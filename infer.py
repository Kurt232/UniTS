import json
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
import torch

from models.tst import TestModel, ModelArgs

device = 'cuda'

load_path = '/data/wjdu/TST_HEAD/checkpoint-199.pth'
save_path = '/data/wjdu/test/'

num_class = 7
test_paths = ["/data/wjdu/data4/realworld1/realworld_10_thigh_TEST.json"]

os.makedirs(save_path, exist_ok=True)

def data_preprocess(imu_data):
    imu_input = torch.tensor(imu_data, dtype=torch.float32)
    assert imu_input.shape == (6, 200), f"imu_input shape: {imu_input.shape}"
    imu_input = torch.stack((imu_input[0:3, :], imu_input[3:6, :]))
    assert imu_input.shape == (2, 3, 200), f"imu_input shape: {imu_input.shape}"

    return imu_input

def main():
    # define the model
    model_args = ModelArgs()
    model = TestModel(model_args, num_class)
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
                imu_input = data_preprocess(data['imu_input'])
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