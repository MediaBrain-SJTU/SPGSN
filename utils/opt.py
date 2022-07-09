#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_dir', type=str, default='./data/h3.6m/dataset', help='path to H36M dataset')
        self.parser.add_argument('--data_dir_3dpw', type=str, default='/home/wei/Downloads/3DPW/sequenceFiles/', help='path to 3DPW dataset')
        self.parser.add_argument('--data_dir_cmu', type=str, default='/GPFS/data/msli/My_datasets/CMUMocap/', help='path to CMU dataset')
        self.parser.add_argument('--exp', type=str, default='test', help='ID of experiment')
        self.parser.add_argument('--ckpt', type=str, default='checkpoint/', help='path to save checkpoint')

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--max_norm', dest='max_norm', action='store_true', help='maxnorm constraint to weights')
        self.parser.add_argument('--linear_size', type=int, default=256, help='size of each model layer')
        self.parser.add_argument('--num_stage', type=int, default=9, help='# layers in linear model')
        self.parser.add_argument('--J', type=int, default=1, help='The number of wavelet filters')
        self.parser.add_argument('--tree_num', type=int, default=2, help='The number of scattering tree')
        self.parser.add_argument('--edge_prob', type=float, default=0.4, help='The probability of edge preservation')
        self.parser.add_argument('--W_pg', type=float, default=0.6, help='The weight of information propagation between part') 
        self.parser.add_argument('--W_p', type=float, default=0.6, help='The weight of part on the whole body')

        # ===============================================================
        #                     Running options
        # ===============================================================
        self.parser.add_argument('--lr', type=float, default=1.0e-3)
        self.parser.add_argument('--lr_decay', type=int, default=2, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96)
        self.parser.add_argument('--input_n', type=int, default=10, help='observed seq length')
        self.parser.add_argument('--output_n', type=int, default=10, help='future seq length')
        self.parser.add_argument('--dct_n', type=int, default=15, help='number of DCT coeff. preserved for 3D')
        self.parser.add_argument('--actions', type=str, default='all', help='path to save checkpoint')
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability, 1.0 to make no dropout')
        self.parser.add_argument('--train_batch', type=int, default=32)
        self.parser.add_argument('--test_batch', type=int, default=128)
        self.parser.add_argument('--job', type=int, default=10, help='subprocesses to use for data loading')
        self.parser.add_argument('--is_load', dest='is_load', action='store_true', help='wether to load existing model')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='frame sampling rate')
        self.parser.add_argument('--is_norm_dct', dest='is_norm_dct', action='store_true', help='whether to normalize the dct coeff')
        self.parser.add_argument('--is_norm', dest='is_norm', action='store_true', help='whether to normalize the angles/3d coordinates')

        self.parser.set_defaults(max_norm=True)
        self.parser.set_defaults(is_load=False)
        # self.parser.set_defaults(is_norm_dct=True)
        # self.parser.set_defaults(is_norm=True)

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        # do some pre-check
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self._print()
        return self.opt
