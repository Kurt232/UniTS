import json
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
import torch

from models.units import UniTSPretrainedModel

load_path = '/data/wenhao/wjdu/location/output/UniTS_HEAD/ankle/checkpoint-39.pth'
save_path = '/data/wenhao/wjdu/location/results/UniTS_HEAD'

num_class = 8

device = 'cuda'
test_paths = ['/data/wenhao/wjdu/location/data/test_pure_cla_hand.json', '/data/wenhao/wjdu/location/data/test_pure_cla_chest.json', '/data/wenhao/wjdu/location/data/test_pure_cla_ankle.json']

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
    
    mapping = {
            "climbing stairs": 0,
            "descending stairs": 1,
            "sitting": 2,
            "standing": 3,
            "walking": 4,
            "lying": 5,
            "biking": 6,
            "jogging": 7
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