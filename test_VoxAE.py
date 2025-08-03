from tqdm import tqdm
from dataset.datasetVoxAE import get_dataloader
from config import ConfigVoxAE
from utils import ensure_dir
from trainer import TrainerVoxAE
from mslib.macro import *
import torch
import numpy as np
import os
import h5py
import json


def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigVoxAE('test')

    if cfg.mode == 'rec':
        reconstruct(cfg)
    elif cfg.mode == 'enc':
        encode(cfg)
    elif cfg.mode == 'dec':
        decode(cfg)
    else:
        raise ValueError


def reconstruct(cfg):
    # create network and training agent
    tr_agent = TrainerVoxAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # create the list to store the evaluation results of the test set
    all_rs_iou = []
    all_mill_iou = []
    all_drill_iou = []
    all_slant_iou = []

    # create dataloader
    test_loader = get_dataloader('test', cfg)

    if cfg.outputs is None:
        cfg.outputs = '{}/voxel_rec/test_VoxAE_{}'.format(cfg.data_root, cfg.ckpt)
    ensure_dir(cfg.outputs)

    # create a new h5 file for each data_id in the test set to store the reconstructed voxels
    pbar = tqdm(test_loader)
    for b, data in enumerate(pbar):
        with torch.no_grad():
            #forward pass of the voxel autoencoder
            voxel = data['voxel'].cuda(non_blocking=True) # (N, 1, vox_dim, vox_dim, vox_dim)
            outputs = tr_agent.net(voxel)
            batch_voxel_out = tr_agent.binarize_voxel_logits(outputs) # (N, 1, vox_dim, vox_dim, vox_dim)
            batch_size = batch_voxel_out.shape[0]

        batch_voxel_gt = voxel.detach().cpu().numpy() # (N, 1, vox_dim, vox_dim, vox_dim)
        batch_operation_gt = data['operation'].to(torch.uint8).numpy() # (N, )

        for i in range(batch_size):
            # get the reconstructed voxel and ground truth voxel
            voxel_out = np.squeeze(batch_voxel_out[i]) # (dim_z, dim_z, dim_z)

            voxel_gt = np.squeeze(batch_voxel_gt[i]) # (dim_z, dim_z, dim_z)
            operation_gt = batch_operation_gt[i] # ()

            # calculate IoU for each voxel
            intersection = np.logical_and(voxel_out, voxel_gt)
            union = np.logical_or(voxel_out, voxel_gt)
            iou = np.sum(intersection) / np.sum(union)

            if operation_gt == RS_IDX:
                all_rs_iou.append(iou)
            elif operation_gt == MILL_IDX:
                all_mill_iou.append(iou)
            elif operation_gt == DRILL_IDX:
                all_drill_iou.append(iou)
            elif operation_gt == SLANT_IDX:
                all_slant_iou.append(iou)

    # calculate the mean IoU for each operation
    rs_iou = np.mean(all_rs_iou)
    mill_iou = np.mean(all_mill_iou)
    drill_iou = np.mean(all_drill_iou)
    slant_iou = np.mean(all_slant_iou)

    results = {
        'rs_iou': f'{rs_iou:.6f}',
        'mill_iou': f'{mill_iou:.6f}',
        'drill_iou': f'{drill_iou:.6f}',
        'slant_iou': f'{slant_iou:.6f}'
    }

    # save the evaluation results
    results_path = os.path.join(cfg.outputs, 'reconstruction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)




def encode(cfg):
    # create network and training agent
    tr_agent = TrainerVoxAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # create a list to store the latent vectors and compute the mean and std deviation
    all_zs = []

    # create a dataset in every h5 file from the train, validation, and test sets to store latent vectors
    save_dir = os.path.join(cfg.data_root, 'seq_h5')
    ensure_dir(save_dir)
    for phase in ['train', 'validation', 'test']:
        loader = get_dataloader(phase, cfg, shuffle=False)

        # encode and save latent vectors
        pbar = tqdm(loader)
        for b, data in enumerate(pbar):
            with torch.no_grad():
                batch_z = tr_agent.encode(data) # (N, dim_z)
                batch_z = batch_z.detach().cpu().numpy()
                batch_size = batch_z.shape[0]
            
            for i in range(batch_size):
                z = batch_z[i] # (dim_z, )
                if phase == 'train':    
                    all_zs.append(z)

                data_id = data['data_id'][i]
                folder, filename, idx = data_id.rsplit('/', 2)
                save_path = os.path.join(save_dir, folder, filename + '.h5')

                with h5py.File(save_path, 'a') as h5file:
                    if 'zs' not in h5file:
                        length = h5file['operations'].shape[0]
                        shape = (length, ) + z.shape
                        maxshape = (None, ) + z.shape
                        chunks = (1, ) + z.shape
                        h5file.create_dataset('zs', shape, maxshape=maxshape, chunks=chunks, compression='gzip', dtype=z.dtype)

                    h5file['zs'][int(idx)] = z

    # compute the mean and std deviation of the latent vectors
    all_zs = np.array(all_zs)
    mean = np.mean(all_zs, axis=0)
    std = np.std(all_zs, axis=0)
    mean_std_path = os.path.join(cfg.data_root, 'zs_mean_std.h5')
    with h5py.File(mean_std_path, 'w') as h5file:
        h5file.create_dataset('mean', data=mean)
        h5file.create_dataset('std', data=std)



def decode(cfg):
    # create network and training agent
    tr_agent = TrainerVoxAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # load latent zs
    cfg.sample_path = os.path.join(cfg.data_root, 'seq_h5', cfg.sample_path)
    with h5py.File(cfg.sample_path, 'r') as h5file:
        batch_z = h5file['zs'][:-1] # (N, dim_z)
        batch_operation = h5file['operations'][:-1] # (N, )
        batch_voxel_gt = h5file['voxels'][:] # (N+1, vox_dim, vox_dim, vox_dim)
        length = batch_z.shape[0] + 1

    if cfg.outputs is None:
        cfg.outputs = '{}/voxel_rec/test_VoxAE_{}'.format(cfg.data_root, cfg.ckpt)
    ensure_dir(cfg.outputs)

    # create dictionary to map operation index to operation name
    operations_dict = {
        RS_IDX: 'Raw Stock',
        MILL_IDX: 'Milling',
        DRILL_IDX: 'Drilling',
        SLANT_IDX: 'Slant',
        FP_IDX: 'Final Part'
    }

    # decode
    with torch.no_grad():
        batch_z = torch.tensor(batch_z).cuda()
        outputs = tr_agent.decode(batch_z)
        batch_voxel_out = tr_agent.binarize_voxel_logits(outputs) # (N, 1, vox_dim, vox_dim, vox_dim)
        batch_size = batch_voxel_out.shape[0]

    fp_out = np.ones(batch_voxel_out[0, 0].shape, dtype=bool)

    # save the decoded voxels
    for i in range(batch_size):
        _, folder, filename = cfg.sample_path.rsplit("/", 2)
        filename = filename.split('.')[0]
        idx = str(i).zfill(2)
        voxel_out = np.squeeze(batch_voxel_out[i])
        voxel_gt = np.squeeze(batch_voxel_gt[i])
        operation = batch_operation[i]

        fp_out = np.logical_and(fp_out, ~voxel_out)

        if operation == RS_IDX:
            voxel_out = ~voxel_out

        intersection = np.logical_and(voxel_out, voxel_gt)
        union = np.logical_or(voxel_out, voxel_gt)
        iou = np.sum(intersection) / np.sum(union)
        print(f'{i + 1}: {operations_dict[operation]}, IoU: {iou:.6f}')
            
        save_folder = os.path.join(cfg.outputs, folder)
        ensure_dir(save_folder)
        save_path = os.path.join(save_folder, filename + '.h5')
        with h5py.File(save_path, 'a') as h5file:
            if 'voxels' not in h5file:
                shape = (length, ) + voxel_out.shape
                maxshape = (None, ) + voxel_out.shape
                chunks = (1, ) + voxel_out.shape
                h5file.create_dataset('voxels', shape, maxshape=maxshape, chunks=chunks, compression='gzip', dtype=voxel_out.dtype)

            h5file['voxels'][int(idx)] = voxel_out
            # save the final part
            if i == batch_size - 1:
                h5file['voxels'][-1] = fp_out
                fp_gt = np.squeeze(batch_voxel_gt[-1])
                intersection = np.logical_and(fp_out, fp_gt)
                union = np.logical_or(fp_out, fp_gt)
                iou = np.sum(intersection) / np.sum(union)
                print(f'{i + 2}: {operations_dict[FP_IDX]}, IoU: {iou:.6f}')
                print(f'Predicted sample saved in {save_path}')

if __name__ == '__main__':
    main()

