from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
from mslib.macro import *


def get_dataloader(phase, cfg, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = DatasetVoxAE(phase, cfg)
    print("Dataset length for phase {}: {}".format(phase, len(dataset)))
    # TODO: check if pin_memory is necessary
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.batch_size, 
                            shuffle=is_shuffle, 
                            num_workers=cfg.num_workers,
                            pin_memory=True, # faster data transfer to GPU
                            worker_init_fn=np.random.seed())
    return dataloader


class DatasetVoxAE(Dataset):
    def __init__(self, phase, cfg):
        super(DatasetVoxAE, self).__init__()
        self.raw_data = os.path.join(cfg.data_root, "seq_h5") # h5 data root
        self.phase = phase
        self.path = os.path.join(cfg.data_root, "train_val_test_split_voxels.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        folder, filename, idx = data_id.rsplit("/", 2)
        h5_path = os.path.join(self.raw_data, folder, filename + ".h5")
        idx = int(idx)
        with h5py.File(h5_path, "r") as fp:
            vox = fp["voxels"][idx] # (128, 128, 128)
            op = fp["operations"][idx] # (1, )

        vox = vox[np.newaxis] # (1, 128, 128, 128)
        if op == RS_IDX:
            vox = ~vox # invert Raw Stock voxels

        voxel = torch.tensor(vox, dtype=torch.float) # (1, 128, 128, 128)
        operation = torch.tensor(op, dtype=torch.long) # (1, )
        return {"operation": operation, "voxel": voxel, "data_id": data_id}

    def __len__(self):
        return len(self.all_data)
