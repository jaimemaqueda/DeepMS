from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
from mslib.macro import *


def get_dataloader(phase, cfg, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = DatasetPartE(phase, cfg)
    print("Dataset length for phase {}: {}".format(phase, len(dataset)))
    # TODO: check if pin_memory is necessary
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=is_shuffle, 
                            num_workers=cfg.num_workers,
                            pin_memory=True, # faster data transfer to GPU
                            worker_init_fn=np.random.seed())
    return dataloader


class DatasetPartE(Dataset):
    def __init__(self, phase, cfg):
        super(DatasetPartE, self).__init__()
        self.raw_data = os.path.join(cfg.data_root, "seq_h5") # h5 data root
        self.phase = phase
        self.path = os.path.join(cfg.data_root, "train_val_test_split.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            fp_vox = fp["voxels"][-1] # (128, 128, 128)
            seq_z = fp["zs"][-1] # (256, )
        fp_vox = fp_vox[np.newaxis] # (1, 128, 128, 128)

        fp_vox = torch.tensor(fp_vox, dtype=torch.float) # (1, 128, 128, 128)
        seq_z = torch.tensor(seq_z, dtype=torch.float) # (256, )
        return {"seq_z": seq_z, "fp_vox": fp_vox, "data_id": data_id}

    def __len__(self):
        return len(self.all_data)
