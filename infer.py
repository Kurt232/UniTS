import json
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
import torch

from models.units import UniTSPretrainedModel

load_path = '/data/wenhao/wjdu/adapterv2/benchmark/UniTS_HEAD/checkpoint-39.pth'
save_path = '/data/wenhao/wjdu/adapterv2/benchmark/UniTS_HEAD/results/'

device = 'cuda'
test_paths = ['/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_hhar_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_hhar_uncommon.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_motion_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_motion_uncommon.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_shoaib_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_shoaib_uncommon.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_uci_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_uci_uncommon.json']

os.makedirs(save_path, exist_ok=True)

def main():

    # define the model
    model = UniTSPretrainedModel(d_enc_in=6, num_class=8)
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
            "biking": 5,
            "jogging": 6,
            "lying": 7,
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

        result_file = data_path.split('/')[-1]
        prediction_file = os.path.join(save_path, result_file)
        json.dump(predictions, open(prediction_file, 'w'), indent=2)

        print(f"{result_file} ", "Accuracy: {:.4f}%".format(correct_pred / len(data_item) * 100))
        acc_total[result_file] = correct_pred / len(data_item)
    
    print(json.dumps(acc_total, indent=2))

if __name__ == '__main__':
    main()