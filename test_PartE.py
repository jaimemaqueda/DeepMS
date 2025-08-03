from tqdm import tqdm
from collections import OrderedDict
from mslib.macro import *
from dataset.datasetPartE import get_dataloader
from config import ConfigPartE
from utils import ensure_dir
from trainer import TrainerPartE
import torch
import numpy as np
import os
import h5py
import json
import sys
import csv
from scipy.ndimage import shift as ndi_shift

# === Centroid and Alignment Utilities ===
def compute_centroid(vox):
    coords = np.argwhere(vox > 0)
    if len(coords) == 0:
        return np.array([0., 0., 0.])
    return coords.mean(axis=0)

def shift_to_center(vox, centroid, target_center):
    shift_vec = np.array(target_center) - centroid
    return ndi_shift(vox.astype(float), shift=shift_vec, order=0, mode='constant', cval=0) > 0.5  # binarize

def centroid_aligned_iou(vox_pred, vox_gt, vox_dim):
    centroid_pred = compute_centroid(vox_pred)
    centroid_gt = compute_centroid(vox_gt)
    center = np.array([(vox_dim - 1) / 2.0] * 3)
    aligned_pred = shift_to_center(vox_pred, centroid_pred, center)
    aligned_gt = shift_to_center(vox_gt, centroid_gt, center)
    intersection = np.logical_and(aligned_pred, aligned_gt)
    union = np.logical_or(aligned_pred, aligned_gt)
    aligned_iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 1.0
    centroid_dist = np.linalg.norm(centroid_pred - centroid_gt)
    return aligned_iou, centroid_dist

def main():
    cfg = ConfigPartE('test')

    if cfg.mode == 'rec_data':
        reconstruct_data(cfg)
    if cfg.mode == 'rec_sample':
        reconstruct_sample(cfg)
    else:
        raise ValueError


def reconstruct_data(cfg):

    # create training agents for the sequence and voxel autoencoders
    tr_agent = TrainerPartE(cfg)

    # load trained models from checkpoints
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

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
    drill_precedance_acc_all = []

    # New: lists for centroid-aligned IoU and centroid distance
    all_rs_iou_aligned = []
    all_rs_centroid_dist = []
    all_mill_iou_aligned = []
    all_mill_centroid_dist = []
    all_drill_iou_aligned = []
    all_drill_centroid_dist = []
    all_slant_iou_aligned = []
    all_slant_centroid_dist = []
    all_fp_iou_aligned = []
    all_fp_centroid_dist = []

    # create dataloader
    test_loader = get_dataloader('test', cfg)

    if cfg.outputs is None:
        cfg.outputs = '{}/voxel_rec/test_PartE_{}'.format(cfg.data_root, cfg.ckpt)
    ensure_dir(cfg.outputs)

    # create a csv file to store the results of the reconstruction
    csv_path = os.path.join(cfg.outputs, 'reconstruction_results.csv')
    fieldname = ['folder', 'filename', 'seq_type', 'num_mill', 'num_drill', 'num_slant', 'rs_iou', 'mill_iou', 'drill_iou', 'slant_iou', 'fp_iou', 'avg_iou', 'seq_acc', 'drill_1_iou', 'drill_2_iou', 'drill_3_iou', 'drill_4_iou', 'drill_5_iou', 'mill_1_iou', 'mill_2_iou', 'mill_3_iou', 'mill_4_iou',
        # New fields for aligned IoU and centroid distance
        'fp_iou_aligned', 'fp_centroid_dist',
        'drill_1_iou_aligned', 'drill_1_centroid_dist',
        'drill_2_iou_aligned', 'drill_2_centroid_dist',
        'drill_3_iou_aligned', 'drill_3_centroid_dist',
        'drill_4_iou_aligned', 'drill_4_centroid_dist',
        'drill_5_iou_aligned', 'drill_5_centroid_dist',
        'mill_1_iou_aligned', 'mill_1_centroid_dist',
        'mill_2_iou_aligned', 'mill_2_centroid_dist',
        'mill_3_iou_aligned', 'mill_3_centroid_dist',
        'mill_4_iou_aligned', 'mill_4_centroid_dist',
    ]
    csvfile = open(csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldname)
    csv_writer.writeheader()

    # create a new h5 file for each data_id in the test set to store the reconstructed voxels
    pbar = tqdm(test_loader)
    for b, data in enumerate(pbar):
        with torch.no_grad():
            # forward pass of the full model
            fp_vox = data['fp_vox'].cuda(non_blocking=True)  # (N, 1, 128, 128, 128)
            output = tr_agent.net(fp_vox)
            batch_seq_z = output["seq_z_out"] # (N, 256)
            batch_size = batch_seq_z.shape[0]

        for i in range(batch_size):
            # decode the latent vector into a sequence of operations and voxels
            seq_z_out = batch_seq_z[i].unsqueeze(0) # (1, 256)
            seq_op_out, seq_vox_out = tr_agent.decode_z(seq_z_out) # (S, ), (S, 1, 128, 128, 128)

            # get ground truth of the sequence of operations and voxels
            data_id = data['data_id'][i]
            folder, filename = data_id.rsplit("/", 1)
            with h5py.File(os.path.join(cfg.data_root, "seq_h5", folder, filename + ".h5"), "r") as h5file:
                seq_op_gt = h5file["operations"][:] # (S, )
                seq_vox_gt = h5file["voxels"][:] # (S, 128, 128, 128)

            length = seq_op_out.shape[0]
            sequence_acc = int(length == seq_op_gt.shape[0] - 1) # 1 if the sequence length is correct, 0 otherwise (FP voxel is not included)
            all_eos_acc.append(sequence_acc)
            fp_vox_out = np.ones(seq_vox_gt[0].shape, dtype=bool)

            # calculate the number of each operation in the ground truth sequence for the csv file
            num_mill = len([op for op in seq_op_gt if op == MILL_IDX])
            num_drill = len([op for op in seq_op_gt if op == DRILL_IDX])
            num_slant = len([op for op in seq_op_gt if op == SLANT_IDX])
            unique_operations = list(OrderedDict.fromkeys(seq_op_gt[1:-1]))
            seq_type = '_'.join([str(op) for op in unique_operations])
            sample_rs_iou = 0
            sample_mill_iou = 0
            sample_drill_iou = 0
            sample_slant_iou = 0
            drill_ious = [0] * 5
            drill_aligned_ious = [0] * 5  # New
            drill_centroid_dists = [0] * 5  # New
            drill_count = 0
            mill_ious = [0] * 4
            mill_aligned_ious = [0] * 4  # New
            mill_centroid_dists = [0] * 4  # New
            mill_count = 0

            drill_voxels_count = []

            for j in range(length):
                op_out = seq_op_out[j]
                vox_out = seq_vox_out[j].squeeze()
                fp_vox_out = np.logical_and(fp_vox_out, ~vox_out)

                if op_out == RS_IDX:
                    vox_out = ~vox_out
                if j < seq_op_gt.shape[0] - 1:
                    op_gt = seq_op_gt[j]
                    vox_gt = seq_vox_gt[j]

                    # calculate IoU for each voxel
                    intersection = np.logical_and(vox_gt, vox_out)
                    union = np.logical_or(vox_gt, vox_out)
                    iou = np.sum(intersection) / np.sum(union)

                    # New: centroid-aligned IoU and centroid distance
                    aligned_iou, centroid_dist = centroid_aligned_iou(vox_out, vox_gt, VOX_DIM)

                    # calculate accuracy for each operation
                    acc = int(op_out == op_gt)
                    sequence_acc = sequence_acc * acc

                    if op_gt == RS_IDX:
                        all_rs_iou.append(iou)
                        all_rs_acc.append(acc)
                        sample_rs_iou += iou
                        all_rs_iou_aligned.append(aligned_iou)
                        all_rs_centroid_dist.append(centroid_dist)
                    if op_gt == MILL_IDX:
                        all_mill_iou.append(iou)
                        all_mill_acc.append(acc)
                        sample_mill_iou += iou
                        all_mill_iou_aligned.append(aligned_iou)
                        all_mill_centroid_dist.append(centroid_dist)
                        if mill_count < 4:
                            mill_ious[mill_count] = iou
                            mill_aligned_ious[mill_count] = aligned_iou
                            mill_centroid_dists[mill_count] = centroid_dist
                        mill_count += 1
                    if op_gt == DRILL_IDX:
                        all_drill_iou.append(iou)
                        all_drill_acc.append(acc)
                        sample_drill_iou += iou
                        drill_voxels_count.append(np.sum(vox_out))  # Store the volume of the drill operation
                        all_drill_iou_aligned.append(aligned_iou)
                        all_drill_centroid_dist.append(centroid_dist)
                        if drill_count < 5:
                            drill_ious[drill_count] = iou
                            drill_aligned_ious[drill_count] = aligned_iou
                            drill_centroid_dists[drill_count] = centroid_dist
                        drill_count += 1
                    if op_gt == SLANT_IDX:
                        all_slant_iou.append(iou)
                        all_slant_acc.append(acc)
                        sample_slant_iou += iou
                        all_slant_iou_aligned.append(aligned_iou)
                        all_slant_centroid_dist.append(centroid_dist)

            # Calculate drill volume precedence accuracy
            if len(drill_voxels_count) > 1:
                precedence_acc = [
                    drill_voxels_count[k] > drill_voxels_count[k + 1] for k in range(len(drill_voxels_count) - 1)
                ]
                drill_precedance_acc_all.extend(precedence_acc)

            # calculate IoU for the final part of the voxel
            fp_vox_gt = seq_vox_gt[-1]
            intersection = np.logical_and(fp_vox_gt, fp_vox_out)
            union = np.logical_or(fp_vox_gt, fp_vox_out)
            iou = np.sum(intersection) / np.sum(union)
            all_fp_iou.append(iou)
            all_sequences_acc.append(sequence_acc)

            # New: centroid-aligned IoU and centroid distance for final part
            aligned_fp_iou, fp_centroid_dist = centroid_aligned_iou(fp_vox_out, fp_vox_gt, VOX_DIM)
            all_fp_iou_aligned.append(aligned_fp_iou)
            all_fp_centroid_dist.append(fp_centroid_dist)

            # store results for csv
            sample_rs_iou = sample_rs_iou
            sample_mill_iou /= num_mill if num_mill > 0 else 1
            sample_drill_iou /= num_drill if num_drill > 0 else 1
            sample_slant_iou /= num_slant if num_slant > 0 else 1
            sample_avg_iou = (
                (sample_rs_iou + sample_mill_iou * num_mill + sample_drill_iou * num_drill + sample_slant_iou * num_slant) 
                / (1 + num_mill + num_drill + num_slant)
            )
            csv_writer.writerow({
                'folder': folder,
                'filename': filename,
                'seq_type': seq_type,
                'num_mill': num_mill,
                'num_drill': num_drill,
                'num_slant': num_slant,
                'rs_iou': round(sample_rs_iou, 6),
                'mill_iou': round(sample_mill_iou, 6),
                'drill_iou': round(sample_drill_iou, 6),
                'slant_iou': round(sample_slant_iou, 6),
                'avg_iou': round(sample_avg_iou, 6),
                'fp_iou': round(iou, 6),
                'seq_acc': sequence_acc,
                'drill_1_iou': round(drill_ious[0], 6),
                'drill_2_iou': round(drill_ious[1], 6),
                'drill_3_iou': round(drill_ious[2], 6),
                'drill_4_iou': round(drill_ious[3], 6),
                'drill_5_iou': round(drill_ious[4], 6),
                'mill_1_iou': round(mill_ious[0], 6),
                'mill_2_iou': round(mill_ious[1], 6),
                'mill_3_iou': round(mill_ious[2], 6),
                'mill_4_iou': round(mill_ious[3], 6),
                # New fields for aligned IoU and centroid distance
                'fp_iou_aligned': round(aligned_fp_iou, 6),
                'fp_centroid_dist': round(fp_centroid_dist, 6),
                'drill_1_iou_aligned': round(drill_aligned_ious[0], 6),
                'drill_1_centroid_dist': round(drill_centroid_dists[0], 6),
                'drill_2_iou_aligned': round(drill_aligned_ious[1], 6),
                'drill_2_centroid_dist': round(drill_centroid_dists[1], 6),
                'drill_3_iou_aligned': round(drill_aligned_ious[2], 6),
                'drill_3_centroid_dist': round(drill_centroid_dists[2], 6),
                'drill_4_iou_aligned': round(drill_aligned_ious[3], 6),
                'drill_4_centroid_dist': round(drill_centroid_dists[3], 6),
                'drill_5_iou_aligned': round(drill_aligned_ious[4], 6),
                'drill_5_centroid_dist': round(drill_centroid_dists[4], 6),
                'mill_1_iou_aligned': round(mill_aligned_ious[0], 6),
                'mill_1_centroid_dist': round(mill_centroid_dists[0], 6),
                'mill_2_iou_aligned': round(mill_aligned_ious[1], 6),
                'mill_2_centroid_dist': round(mill_centroid_dists[1], 6),
                'mill_3_iou_aligned': round(mill_aligned_ious[2], 6),
                'mill_3_centroid_dist': round(mill_centroid_dists[2], 6),
                'mill_4_iou_aligned': round(mill_aligned_ious[3], 6),
                'mill_4_centroid_dist': round(mill_centroid_dists[3], 6),
            })

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
    sequence_acc = np.mean(all_sequences_acc)
    drill_precedance_acc = np.mean(drill_precedance_acc_all)

    # New: global means for aligned IoU and centroid distance
    rs_iou_aligned = np.mean(all_rs_iou_aligned)
    rs_centroid_dist = np.mean(all_rs_centroid_dist)
    mill_iou_aligned = np.mean(all_mill_iou_aligned)
    mill_centroid_dist = np.mean(all_mill_centroid_dist)
    drill_iou_aligned = np.mean(all_drill_iou_aligned)
    drill_centroid_dist = np.mean(all_drill_centroid_dist)
    slant_iou_aligned = np.mean(all_slant_iou_aligned)
    slant_centroid_dist = np.mean(all_slant_centroid_dist)
    fp_iou_aligned = np.mean(all_fp_iou_aligned)
    fp_centroid_dist = np.mean(all_fp_centroid_dist)

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
        'sequence_acc': f"{sequence_acc:.6f}",
        'drill_volume_precedance_acc': f"{drill_precedance_acc:.6f}",
        # New global metrics
        'rs_iou_aligned': f"{rs_iou_aligned:.6f}",
        'rs_centroid_dist': f"{rs_centroid_dist:.6f}",
        'mill_iou_aligned': f"{mill_iou_aligned:.6f}",
        'mill_centroid_dist': f"{mill_centroid_dist:.6f}",
        'drill_iou_aligned': f"{drill_iou_aligned:.6f}",
        'drill_centroid_dist': f"{drill_centroid_dist:.6f}",
        'slant_iou_aligned': f"{slant_iou_aligned:.6f}",
        'slant_centroid_dist': f"{slant_centroid_dist:.6f}",
        'fp_iou_aligned': f"{fp_iou_aligned:.6f}",
        'fp_centroid_dist': f"{fp_centroid_dist:.6f}",
    }

    results_path = os.path.join(cfg.outputs, 'reconstruction_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    csvfile.flush()
    csvfile.close()

def reconstruct_sample(cfg):
    # create training agents for the sequence and voxel autoencoders
    tr_agent = TrainerPartE(cfg)

    # load trained models from checkpoints
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # load sample data
    folder, filename = cfg.sample_path.rsplit("/", 1)
    cfg.sample_path = os.path.join(cfg.data_root, 'seq_h5', folder, filename + '.h5')
    with h5py.File(cfg.sample_path, 'r') as h5file:
        seq_op_gt = h5file['operations'][:] # (S, )
        seq_vox_gt = h5file['voxels'][:] # (S, 128, 128, 128)
        fp_vox = seq_vox_gt[-1]
        fp_vox = fp_vox[np.newaxis, np.newaxis] # (1, 1, 128, 128, 128)

    # create the directory to store the reconstructed voxels
    if cfg.outputs is None:
        cfg.outputs = '{}/voxel_rec/test_PartE_{}'.format(cfg.data_root, cfg.ckpt)
    ensure_dir(cfg.outputs)

    # run the forward pass of the full model
    with torch.no_grad():
        fp_vox = torch.tensor(fp_vox, dtype=torch.float).cuda(non_blocking=True) # (1, 1, 128, 128, 128)
        output = tr_agent.net(fp_vox)
        seq_z = output["seq_z_out"] # (1, 256)
        seq_op_out, seq_vox_out = tr_agent.decode_z(seq_z) # (S, ), (S, 1, 128, 128, 128)
        length = seq_op_out.shape[0]
    
    # create dictionary to map operation index to operation name
    operations_dict = {
        RS_IDX: 'Raw Stock',
        MILL_IDX: 'Milling',
        DRILL_IDX: 'Drilling',
        SLANT_IDX: 'Slant',
        FP_IDX: 'Final Part'
    }

    # compute and print Iou for each voxel and the overall final part IoU
    fp_vox_out = np.ones(seq_vox_gt[0].shape, dtype=bool)
    save_folder = os.path.join(cfg.outputs, folder)
    ensure_dir(save_folder)
    save_path = os.path.join(save_folder, filename + '.h5')
    for i in range(length):
        op_out = seq_op_out[i]
        vox_out = seq_vox_out[i].squeeze()
        fp_vox_out = np.logical_and(fp_vox_out, ~vox_out)

        if op_out == RS_IDX:
            vox_out = ~vox_out
        if i < seq_op_gt.shape[0] - 1:
            op_gt = seq_op_gt[i]
            vox_gt = seq_vox_gt[i]

            intersection = np.logical_and(vox_gt, vox_out)
            union = np.logical_or(vox_gt, vox_out)
            iou = np.sum(intersection) / np.sum(union)
            # New: centroid-aligned IoU and centroid distance
            aligned_iou, centroid_dist = centroid_aligned_iou(vox_out, vox_gt, VOX_DIM)
            print(f"{i} - GT: {operations_dict[op_gt]:<12}, Out: {operations_dict[op_out]:<12}, IoU: {iou:.6f}, IoU_aligned: {aligned_iou:.6f}, CentroidDist: {centroid_dist:.3f}")
        else:
            print(f"{i} - GT: End of Sequence, Out: {operations_dict[op_out]:<12}")

        # save the reconstructed voxels
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

            if i == length - 1:
                h5file['operations'][-1] = FP_IDX
                h5file['voxels'][-1] = fp_vox_out

    # calculate IoU for the final part of the voxel
    fp_vox_gt = seq_vox_gt[-1]
    intersection = np.logical_and(fp_vox_gt, fp_vox_out)
    union = np.logical_or(fp_vox_gt, fp_vox_out)
    iou = np.sum(intersection) / np.sum(union)
    # New: centroid-aligned IoU and centroid distance for final part
    aligned_fp_iou, fp_centroid_dist = centroid_aligned_iou(fp_vox_out, fp_vox_gt, VOX_DIM)
    print(f"{length} - Final Part IoU: {iou:.6f}, IoU_aligned: {aligned_fp_iou:.6f}, CentroidDist: {fp_centroid_dist:.3f}")

      

if __name__ == '__main__':
    main()


