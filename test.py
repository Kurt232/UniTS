from units_train import UniTSDataset

dataset = UniTSDataset(['/data/wenhao/wjdu/fusion_norm/benchmark/train_pure_cla_hhar_common.json'])

print(dataset[0][1].shape)