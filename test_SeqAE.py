from tqdm import tqdm
from mslib.macro import *
from dataset.datasetSeqAE import get_dataloader
from config import ConfigSeqAE, ConfigVoxAE
from utils import ensure_dir
from trainer import TrainerSeqAE, TrainerVoxAE
import torch
import numpy as np
import os
import h5py
import json
import sys

def main():
    # create experiment cfg containing all hyperparameters for SeqAE
    cfg_seq_ae = ConfigSeqAE('test')

    # load hyperparameters for VoxAE from the configuration file
    with open(cfg_seq_ae.path_vox_ae_cfg, 'r') as f:
        cfg_vox_dict = json.load(f)

    # parse hyperparameters for VoxAE from the configuration file
    sys.argv = sys.argv[:1]
    for key, value in cfg_vox_dict.items():
        sys.argv.append(f'--{key}')
        sys.argv.append(str(value))

    # create experiment cfg containing all hyperparameters for VoxAE
    cfg_vox_ae = ConfigVoxAE('test')

    if cfg_seq_ae.mode == 'rec':
        reconstruct(cfg_seq_ae, cfg_vox_ae)
    elif cfg_seq_ae.mode == 'enc':
        encode(cfg_seq_ae)
    elif cfg_seq_ae.mode == 'dec':
        decode(cfg_seq_ae, cfg_vox_ae)
    else:
        raise ValueError


def reconstruct(cfg_seq_ae, cfg_vox_ae):
    # create training agents for the sequence and voxel autoencoders
    tr_agent_seq = TrainerSeqAE(cfg_seq_ae)
    tr_agent_vox = TrainerVoxAE(cfg_vox_ae)

    # load trained models from checkpoints
    tr_agent_seq.load_ckpt(cfg_seq_ae.ckpt)
    tr_agent_seq.net.eval()
    tr_agent_vox.load_ckpt(cfg_vox_ae.ckpt)
    tr_agent_vox.net.eval()

    # create the list to store the evaluation results of the test set
    all_rs_iou = []
    all_rs_acc = []
    all_mill_iou = []
    all_mill_acc = []
    all_drill_iou = []
    all_drill_acc = []
    all_slant_iou = []
    all_slant_acc = []
    all_fp_iou = []
    all_eos_acc = []
    all_sequences_acc = []
    all_wrong_files = []

    # create dataloader
    test_loader = get_dataloader('test', cfg_seq_ae)

    if cfg_seq_ae.outputs is None:
        cfg_seq_ae.outputs = '{}/voxel_rec/test_SeqAE_{}'.format(cfg_seq_ae.data_root, cfg_seq_ae.ckpt)
    ensure_dir(cfg_seq_ae.outputs)

    # Load mean and std from the h5 file
    mean_std_path = os.path.join(cfg_seq_ae.data_root, "zs_mean_std.h5")
    with h5py.File(mean_std_path, "r") as h5file:
        mean = torch.tensor(h5file["mean"][:]).cuda()
        std = torch.tensor(h5file["std"][:]).cuda()


    # create a new h5 file for each data_id in the test set to store the reconstructed volumes
    pbar = tqdm(test_loader)
    for b, data in enumerate(pbar):
        with torch.no_grad():
            # forward pass of the sequence autoencoder
            outputs, _ = tr_agent_seq.forward(data)
            batch_seq_zs = outputs['seq_zs_out'] * std + mean # (MAX_S, N, dim_z)
            batch_seq_op = tr_agent_seq.logits2vec(outputs)['seq_op_out'] # (MAX_S, N)
            batch_size = batch_seq_op.shape[1]

        for i in range(batch_size):
            # load ground truth data for the current data_id
            data_id = data['data_id'][i]
            folder, filename = data_id.rsplit("/", 1)
            with  h5py.File(os.path.join(cfg_seq_ae.data_root, 'seq_h5', folder, filename + '.h5'), 'r') as h5file:
                seq_op_gt = h5file['operations'][:] # (S, )
                seq_vox_gt = h5file['voxels'][:] # (S, vox_dim, vox_dim, vox_dim)

            # load predicted data for the current data_id
            seq_zs_out = batch_seq_zs[:, i] # (MAX_S, dim_z)
            seq_op_out = batch_seq_op[:, i] # (MAX_S, )
            eos_index = np.where(seq_op_out == EOS_IDX)[0]
            if len(eos_index) > 0:
                seq_op_out = seq_op_out[:eos_index[0]] # (S, )
                seq_zs_out = seq_zs_out[:eos_index[0]] # (S, dim_z)
            length = seq_op_out.shape[0]
            sequence_acc = int(length == seq_op_gt.shape[0] - 1) # 1 if the sequence length is correct, 0 otherwise
            all_eos_acc.append(sequence_acc)

            # decode the sequence of latent vectors into a sequence of voxel volumes
            with torch.no_grad():
                outputs = tr_agent_vox.decode(seq_zs_out)
                seq_vox_out = tr_agent_vox.binarize_voxel_logits(outputs) # (S, 1, vox_dim, vox_dim, vox_dim)

            fp_out = np.ones(seq_vox_out[0, 0].shape, dtype=bool)

            for j in range(length):
                op_out = seq_op_out[j] # (1, )
                vox_out = np.squeeze(seq_vox_out[j]) # (128, 128, 128)
                fp_out = np.logical_and(fp_out, ~vox_out)
                if op_out == RS_IDX:
                    vox_out = ~vox_out
                if j < seq_op_gt.shape[0] - 1:
                    op_gt = seq_op_gt[j]
                    vox_gt = seq_vox_gt[j]

                    # Calculate IoU for each volume
                    intersection = np.logical_and(vox_gt, vox_out)
                    union = np.logical_or(vox_gt, vox_out)
                    iou = np.sum(intersection) / np.sum(union)

                    # Calculate accuracy for each operation
                    acc = int(op_out == op_gt)
                    sequence_acc = sequence_acc * acc

                    if op_gt == RS_IDX:
                        all_rs_iou.append(iou)
                        all_rs_acc.append(acc)
                    if op_gt == MILL_IDX:
                        all_mill_iou.append(iou)
                        all_mill_acc.append(acc)
                    if op_gt == DRILL_IDX:
                        all_drill_iou.append(iou)
                        all_drill_acc.append(acc)
                    if op_gt == SLANT_IDX:
                        all_slant_iou.append(iou)
                        all_slant_acc.append(acc)
            

            all_sequences_acc.append(sequence_acc)
            if sequence_acc == 0:
                all_wrong_files.append(data_id)

            # calculate IoU for the final part
            fp_gt = seq_vox_gt[-1]
            intersection = np.logical_and(fp_gt, fp_out)
            union = np.logical_or(fp_gt, fp_out)
            iou = np.sum(intersection) / np.sum(union)
            all_fp_iou.append(iou)

    # Calculate the average IoU and accuracy for each operation and write the results to a file
    rs_iou = np.mean(all_rs_iou)
    mill_iou = np.mean(all_mill_iou)
    drill_iou = np.mean(all_drill_iou)
    slant_iou = np.mean(all_slant_iou)
    fp_iou = np.mean(all_fp_iou)
    rs_acc = np.mean(all_rs_acc)
    mill_acc = np.mean(all_mill_acc)
    drill_acc = np.mean(all_drill_acc)
    slant_acc = np.mean(all_slant_acc)
    eos_acc = np.mean(all_eos_acc)
    seq_acc = np.mean(all_sequences_acc)

    results = {
        'rs_iou': f"{rs_iou:.6f}",
        'mill_iou': f"{mill_iou:.6f}",
        'drill_iou': f"{drill_iou:.6f}",
        'slant_iou': f"{slant_iou:.6f}",
        'fp_iou': f"{fp_iou:.6f}",
        'rs_acc': f"{rs_acc:.6f}",
        'mill_acc': f"{mill_acc:.6f}",
        'drill_acc': f"{drill_acc:.6f}",
        'slant_acc': f"{slant_acc:.6f}",
        'eos_acc': f"{eos_acc:.6f}",
        'seq_acc': f"{seq_acc:.6f}",
        'wrong_files': all_wrong_files
    }

    results_path = os.path.join(cfg_seq_ae.outputs, 'reconstruction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)


def encode(cfg):
    # create network and training agent
    tr_agent = TrainerSeqAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # create a dataset in every h5 file from the train, validation, and test sets to store latent vectors
    save_dir = os.path.join(cfg.data_root, 'seq_h5')
    ensure_dir(save_dir)
    for phase in ['train', 'validation', 'test']:
        loader = get_dataloader(phase, cfg, shuffle=False)

        # encode and save latent vectors
        pbar = tqdm(loader)
        for b, data in enumerate(pbar):
            with torch.no_grad():
                batch_seq_z = tr_agent.encode(data) # (N, dim_z)
                batch_seq_z = batch_seq_z.detach().cpu().numpy()
                batch_size = batch_seq_z.shape[0]
            
            for i in range(batch_size):
                seq_z = batch_seq_z[i] # (dim_z, )

                data_id = data['data_id'][i]
                folder, filename = data_id.rsplit('/', 1)
                save_path = os.path.join(save_dir, folder, filename + '.h5')

                with h5py.File(save_path, 'a') as h5file:
                    
                    h5file['zs'][-1] = seq_z


def decode(cfg_seq_ae, cfg_vox_ae):
    # create network and training agent
    tr_agent_seq = TrainerSeqAE(cfg_seq_ae)
    tr_agent_vox = TrainerVoxAE(cfg_vox_ae)

    # load from checkpoint if provided
    tr_agent_seq.load_ckpt(cfg_seq_ae.ckpt)
    tr_agent_seq.net.eval()
    tr_agent_vox.load_ckpt(cfg_vox_ae.ckpt)
    tr_agent_vox.net.eval()

    # load latent zs
    cfg_seq_ae.sample_path = os.path.join(cfg_seq_ae.data_root, 'seq_h5', cfg_seq_ae.sample_path)
    with h5py.File(cfg_seq_ae.sample_path, 'r') as h5file:
        seq_zs = h5file['zs'][:] # (S, dim_z)
        seq_z = seq_zs[-1] # (dim_z, )
    
    # load mean and std from the h5 file
    mean_std_path = os.path.join(cfg_seq_ae.data_root, "zs_mean_std.h5")
    with h5py.File(mean_std_path, "r") as h5file:
        mean = torch.tensor(h5file["mean"][:]).cuda()
        std = torch.tensor(h5file["std"][:]).cuda()

    if cfg_seq_ae.outputs is None:
        cfg_seq_ae.outputs = '{}/voxel_rec/test_SeqAE_{}'.format(cfg_seq_ae.data_root, cfg_seq_ae.ckpt)
    ensure_dir(cfg_seq_ae.outputs)

    # decode
    with torch.no_grad():
        seq_z = torch.tensor(seq_z).unsqueeze(0).cuda() # (1, dim_z)
        outputs = tr_agent_seq.decode(seq_z)
        seq_zs_out = outputs['seq_zs_out'] * std + mean # (MAX_S, 1, dim_z)
        seq_op_out = tr_agent_seq.logits2vec(outputs)['seq_op_out'] # (MAX_S, 1)
        eos_index = np.where(seq_op_out == EOS_IDX)[0]
        if len(eos_index) > 0:
            seq_zs_out = seq_zs_out[:eos_index[0]] # (S, 1, dim_z)
            seq_op_out = seq_op_out[:eos_index[0]] # (S, 1)
        length = seq_op_out.shape[0]

        # decode the sequence of latent vectors into a sequence of voxel volumes
        outputs = tr_agent_vox.decode(seq_zs_out[:, 0])
        seq_vox_out = tr_agent_vox.binarize_voxel_logits(outputs) # (S, 1, vox_dim, vox_dim, vox_dim)

    fp_out = np.ones(seq_vox_out[0, 0].shape, dtype=bool)
    for i in range(length):
        _, folder, filename = cfg_seq_ae.sample_path.rsplit("/", 2)
        filename = filename.split('.')[0]
        vox_out = np.squeeze(seq_vox_out[i])
        op_out = np.squeeze(seq_op_out[i])

        fp_out = np.logical_and(fp_out, ~vox_out)

        if op_out == RS_IDX:
            vox_out = ~vox_out
            
        save_folder = os.path.join(cfg_seq_ae.outputs, folder)
        ensure_dir(save_folder)
        save_path = os.path.join(save_folder, filename + '.h5')
        with h5py.File(save_path, 'a') as h5file:
            if 'voxels' not in h5file:
                shape = (length + 1, ) + vox_out.shape
                maxshape = (None, ) + vox_out.shape
                chunks = (1, ) + vox_out.shape
                h5file.create_dataset('voxels', shape, maxshape=maxshape, chunks=chunks, compression='gzip', dtype=vox_out.dtype)

            if 'operations' not in h5file:
                shape = (length + 1, )
                maxshape = (None, )
                chunks = (1, )
                h5file.create_dataset('operations', shape, maxshape=maxshape, chunks=chunks, compression='gzip', dtype=op_out.dtype)

            h5file['operations'][i] = op_out
            h5file['voxels'][i] = vox_out

            # save the final part
            if i == length - 1:
                h5file['voxels'][-1] = fp_out
                h5file['operations'][-1] = FP_IDX


if __name__ == '__main__':
    main()


