import json
import numpy as np
import os
import time
import yaml
from tqdm import tqdm
import torch

from models.units import UniTSPretrainedModel

# load_path = '/data/wenhao/wjdu/benchmark/output/UniTS_HEAD_H/checkpoint-99.pth'
# save_path = '/data/wenhao/wjdu/benchmark/results/UniTS_HEAD_H'

# labels = ['biking', 'climbing stairs', 'descending stairs', 'jogging', 'lying', 'sitting', 'standing', 'walking', 'car driving', 'computer work', 'elevator down', 'elevator up', 'folding laundry', 'house cleaning', 'ironing', 'jumping', 'nordic walking', 'playing soccer', 'vacuum cleaning', 'walking left', 'walking right', 'watching TV']
# labels = ['biking', 'climbing stairs', 'descending stairs', 'jogging', 'lying', 'sitting', 'standing', 'walking']
# num_class = len(labels)

device = 'cuda'
# test_paths = ['/data/wenhao/wjdu/benchmark/data/test_pure_cla.json']
# test_paths = [
#     "/data/wenhao/wjdu/fusion_norm/imu/test_pure_cla_hhar.json",
#     "/data/wenhao/wjdu/fusion_norm/imu/test_pure_cla_motion.json",
#     "/data/wenhao/wjdu/fusion_norm/imu/test_pure_cla_shoaib.json",
#     "/data/wenhao/wjdu/fusion_norm/imu/test_pure_cla_uci.json"
# ]

load_path = '/data/wenhao/wjdu/adapterv2/benchmark1/UniTS_HEAD/checkpoint-99.pth'
save_path = '/data/wenhao/wjdu/adapterv2/benchmark1/UniTS_HEAD/heter'

num_class = 8
# test_paths = ['/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_hhar_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_hhar_uncommon.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_motion_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_motion_uncommon.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_shoaib_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_shoaib_uncommon.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_uci_common.json', '/data/wenhao/wjdu/fusion_norm/benchmark/test_pure_cla_uci_uncommon.json']
test_paths = "/data/wenhao/wjdu/fusion_norm/win/test_pure_cla_motion_60_common.json /data/wenhao/wjdu/fusion_norm/win/test_pure_cla_motion_80_common.json /data/wenhao/wjdu/fusion_norm/freq/test_pure_cla_motion_10_common.json /data/wenhao/wjdu/fusion_norm/freq/test_pure_cla_motion_40_common.json /data/wenhao/wjdu/fusion_norm/chan/test_pure_cla_motion_accel_common.json /data/wenhao/wjdu/fusion_norm/chan/test_pure_cla_motion_gyro_common.json".split()

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
    mapping = {
            "climbing stairs": 0,
            "sitting": 1,
            "biking": 2,
            "standing": 3,
            "walking": 4,
            "descending stairs": 5,
            "jogging": 6,
            "lying": 7
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
                # predictions.append({'pred': _mapping[pred_index.item()], 'ref': data['output'], 'data_id': data['data_id'], 'ds': data['ds'], 'loc': data['loc'], 'freq': data['freq'], 'duration': data['duration'], 'is_seen': data['is_seen']})

        result_file = load_path.split('/')[-2] + '_' + data_path.split('/')[-1]
        prediction_file = os.path.join(save_path, result_file)
        json.dump(predictions, open(prediction_file, 'w'), indent=2)

        print(f"{result_file} ", "Accuracy: {:.4f}%".format(correct_pred / len(data_item) * 100))
        acc_total[result_file] = correct_pred / len(data_item)
    
    print(json.dumps(acc_total, indent=2))

if __name__ == '__main__':
    main()