from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
from mslib.macro import *

def custom_collate_fn(batch):
    # batch: list of dictionaries
    seq_op = torch.stack([item["seq_op"] for item in batch], dim=1).contiguous() # (max_total_len, N)
    seq_zs = torch.stack([item["seq_zs"] for item in batch], dim=1).contiguous() # (max_total_len, N, 256)
    ids = [item["data_id"] for item in batch]
    return {"seq_op": seq_op, "seq_zs": seq_zs, "data_id": ids}

def get_dataloader(phase, cfg, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = DatasetSeqAE(phase, cfg)
    # TODO: check if pin_memory is necessary
    dataloader = DataLoader(dataset, 
                            collate_fn=custom_collate_fn,
                            batch_size=cfg.batch_size, 
                            shuffle=is_shuffle, 
                            num_workers=cfg.num_workers,
                            #pin_memory=True, # faster data transfer to GPU
                            worker_init_fn=np.random.seed())
    return dataloader


class DatasetSeqAE(Dataset):
    def __init__(self, phase, cfg):
        super(DatasetSeqAE, self).__init__()
        self.raw_data = os.path.join(cfg.data_root, "seq_h5") # h5 data root
        self.phase = phase
        self.path = os.path.join(cfg.data_root, "train_val_test_split.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.max_total_len = cfg.max_total_len

        # Load mean and std from the h5 file
        mean_std_path = os.path.join(cfg.data_root, "zs_mean_std.h5")
        with h5py.File(mean_std_path, "r") as fp:
            self.mean = fp["mean"][:]
            self.std = fp["std"][:]

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            seq_op = fp["operations"][:-1] # (len, )
            seq_zs = fp["zs"][:-1] # (len, 256)
            
        # Normalize volume
        seq_zs = (seq_zs - self.mean) / self.std

        pad_len = self.max_total_len - seq_op.shape[0]
        seq_op = np.concatenate([seq_op, [EOS_IDX] * pad_len], axis=0) # EOS is added till the max_total_len
        seq_zs = np.concatenate([seq_zs, np.zeros((pad_len, 256))], axis=0) # EOS vectors are added till the max_total_len

        seq_op = torch.tensor(seq_op, dtype=torch.long) # (max_total_len, )
        seq_zs = torch.tensor(seq_zs, dtype=torch.float) # (max_total_len, 256)
        
        return {"seq_op": seq_op, "seq_zs": seq_zs, "data_id": data_id}

    def __len__(self):
        return len(self.all_data)

