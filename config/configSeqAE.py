import os
from utils import ensure_dirs
import argparse
import json
import shutil
from mslib.macro import *


class ConfigSeqAE(object):
    def __init__(self, phase):
        self.is_train = phase == "train"

        self.set_configuration()

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("----Sequence Autoencoder Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name)
        if phase == "train" and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)

        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        ensure_dirs([self.log_dir, self.model_dir])

        # GPU usage
        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        # save this configuration
        if self.is_train:
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    def set_configuration(self):
        self.n_operations = N_OPERATIONS  # Mill, Drill, Slant, Raw Stock, End of Sequence (5)

        self.n_layers = 12                # Number of Encoder blocks
        self.n_layers_decode = 12         # Number of Decoder blocks
        self.n_heads = 8                  # Transformer config: number of heads
        self.dim_feedforward = 512        # Transformer config: FF dimensionality
        self.dim_z = 256                  # Latent vector dimensionality
        self.dropout = 0.0                # Dropout rate used in basic layers and Transformers

        self.max_total_len = MAX_TOTAL_LEN

        self.loss_weights = {
            "loss_seq_op_weight": 1.0,
            "loss_seq_zs_mse_weight": 1.0,
            "loss_seq_zs_cos_weight": 1.0
        }

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()

        parser.add_argument('--proj_dir', type=str, default="train_log", help="path to project folder where models and logs will be saved")
        parser.add_argument('--data_root', type=str, default="../data_deepms", help="path to source data folder")
        parser.add_argument('--exp_name', type=str, default=os.getcwd().split('/')[-1], help="name of this experiment")
        parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

        parser.add_argument('--batch_size', type=int, default=512, help="batch size")
        parser.add_argument('--num_workers', type=int, default=24, help="number of workers for data loading")

        parser.add_argument('--path_vox_ae_cfg', type=str, default='train_log/train_vox_1/config.txt', help="path to the config file of the voxel autoencoder model")

        parser.add_argument('--nr_epochs', type=int, default=200, help="total number of epochs to train")
        parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
        parser.add_argument('--grad_clip', type=float, default=1.0, help="gradient clipping threshold")
        parser.add_argument('--warmup_step', type=int, default=2000, help="step size for learning rate warm up")
        parser.add_argument('--continue', dest='cont',  action='store_true', help="continue training from checkpoint")
        parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
        parser.add_argument('--save_frequency', type=int, default=10, help="save models every x epochs")
        parser.add_argument('--val_frequency', type=int, default=10, help="run validation every x iterations")
        
        if not self.is_train:
            parser.add_argument('-m', '--mode', type=str, choices=['rec', 'enc', 'dec'])
            parser.add_argument('-o', '--outputs', type=str, default=None)
            parser.add_argument('--sample_path', type=str, default='0000/00000000.h5')
        
        args = parser.parse_args()
        return parser, args
