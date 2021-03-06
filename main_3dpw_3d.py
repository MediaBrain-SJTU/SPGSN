#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.pose3dpw3d import Pose3dPW3D
import utils.model as nnmodel
import utils.data_utils as data_utils


def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_in{:d}_out{:d}_dct{:d}_L{:d}_J{:d}_T{:d}_P{:.1f}'.format(opt.input_n, opt.output_n, opt.dct_n, opt.num_stage, opt.J, opt.tree_num, opt.edge_prob)
    checkpoint_dir = '3D_in{:d}_out{:d}_dct{:d}_L{:d}_J{:d}_T{:d}_P{:.1f}'.format(opt.input_n, opt.output_n, opt.dct_n, opt.num_stage, opt.J, opt.tree_num, opt.edge_prob)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    
    upJ = np.array([5,8,11,12,13,14,15,16,17,18,19,20,21,22]) 
    downJ = np.array([0,1,2,3, 4,6,7, 9,10])
    dim_up = np.concatenate((upJ * 3, upJ * 3 + 1, upJ * 3 + 2))
    dim_down = np.concatenate((downJ * 3, downJ * 3 + 1, downJ * 3 + 2))
    n_up = dim_up.shape[0]
    n_down = dim_down.shape[0]
    part_sep = (dim_up, dim_down, n_up, n_down)

    model = nnmodel.GCN(in_d=dct_n, hid_d=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=69, J=opt.J, part_sep=part_sep,
                        W_pg=opt.W_pg, W_p=opt.W_p)

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.is_load:
        model_path_len = 'checkpoint/test/ckpt_main_last.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    train_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=0,
                               dct_n=dct_n)
    dim_used = train_dataset.dim_used
    test_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=1,
                              dct_n=dct_n)
    val_dataset = Pose3dPW3D(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, split=2,
                             dct_n=dct_n)
    # load dadasets for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> test data {}".format(test_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)

        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_3d = train(train_loader, model,
                             optimizer,
                             lr_now=lr_now,
                             max_norm=opt.max_norm,
                             is_cuda=is_cuda,
                             dct_n=dct_n,
                             dim_used=dim_used)
        ret_log = np.append(ret_log, [lr_now, t_3d * 1000])
        head = np.append(head, ['lr', 't_3d'])

        v_3d = val(val_loader, model,
                   is_cuda=is_cuda,
                   dct_n=dct_n,
                   dim_used=dim_used)

        ret_log = np.append(ret_log, v_3d * 1000)
        head = np.append(head, ['v_3d'])

        test_3d = test(test_loader, model,
                       input_n=input_n,
                       output_n=output_n,
                       is_cuda=is_cuda,
                       dim_used=dim_used,
                       dct_n=dct_n)

        ret_log = np.append(ret_log, test_3d * 1000)
        if output_n == 15:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d'])
        elif output_n == 30:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d', '6003d', '7003d', '8003d', '9003d',
                                    '10003d'])

        # update log file
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            if not os.path.exists(opt.ckpt + '/' + checkpoint_dir):
              os.makedirs(opt.ckpt + '/' + checkpoint_dir)
            df.to_csv(opt.ckpt + '/' + checkpoint_dir + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + checkpoint_dir + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        # save ckpt
        is_best = v_3d < err_best
        err_best = min(v_3d, err_best)
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dct_n=15, dim_used=[]):
    t_3d = utils.AccumLoss()

    model.train()
    st = time.time()
    # bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        batch_size = inputs.shape[0]
        if batch_size == 1:
            break
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)
        m_err = loss_funcs.mpjpe_error_3dpw(outputs, all_seq, dct_n, dim_used)

        # calculate loss and backward
        optimizer.zero_grad()
        m_err.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        n, seq_len, _ = all_seq.data.shape
        t_3d.update(m_err.cpu().data.item() * n * seq_len, n * seq_len)
        
        if i%100==0:
            print('{}/{} | TrainLoss {:.4f}, PointLoss {:4f} batch time {:.4f}s|total time{:.2f}s'\
                       .format(i+1, len(train_loader), t_3d.avg, m_err.item(), time.time() - bt, time.time() - st))

        #bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt, time.time() - st)
        #bar.next()
    #bar.finish()
    return lr_now, t_3d.avg


def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
    N = 0
    if output_n == 15:
        eval_frame = [2, 5, 8, 11, 14]
    elif output_n == 30:
        eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    # bar = Bar('>>>', fill='>', max=len(train_loader))
    with torch.no_grad():
        for i, (inputs, targets, all_seq) in enumerate(train_loader):
            bt = time.time()
    
            if is_cuda:
                inputs = Variable(inputs.cuda()).float()
                # targets = Variable(targets.cuda(async=True)).float()
                all_seq = Variable(all_seq.cuda(async=True)).float()
            else:
                inputs = Variable(inputs).float()
                # targets = Variable(targets).float()
                all_seq = Variable(all_seq).float()
            outputs = model(inputs)

            n, seq_len, dim_full_len = all_seq.data.shape

            _, idct_m = data_utils.get_dct_matrix(seq_len)
            idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
            outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
            outputs_exp = torch.matmul(idct_m[:, 0:dct_n], outputs_t).transpose(0, 1).contiguous().view \
                (-1, dim_full_len - 3, seq_len).transpose(1, 2)
            pred_3d = all_seq.clone()
            pred_3d[:, :, dim_used] = outputs_exp
            pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
            targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
    
            for k in np.arange(0, len(eval_frame)):
                j = eval_frame[k]
                t_3d[k] += torch.mean(torch.norm(
                    targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                    1)).cpu().data.item() * n
            
            N += n
        
    print('A ErrT: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}, TestError {:.4f}, total time{:.2f}s'\
             .format(float(t_3d[0])/N, float(t_3d[1])/N, float(t_3d[2])/N, float(t_3d[3])/N, float(t_3d[4])/N, 
                     float(t_3d.mean())/N, time.time() - st))

        #bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt, time.time() - st)
        #bar.next()
    #bar.finish()
    return t_3d / N


def val(train_loader, model, is_cuda=False, dim_used=[], dct_n=15):
    t_3d = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    with torch.no_grad():
        for i, (inputs, targets, all_seq) in enumerate(train_loader):
            bt = time.time()
    
            if is_cuda:
                inputs = Variable(inputs.cuda()).float()
                # targets = Variable(targets.cuda(async=True)).float()
                all_seq = Variable(all_seq.cuda(async=True)).float()
            else:
                inputs = Variable(inputs).float()
                # targets = Variable(targets).float()
                all_seq = Variable(all_seq).float()
            outputs = model(inputs)
            m_err = loss_funcs.mpjpe_error_3dpw(outputs, all_seq, dct_n=dct_n, dim_used=dim_used)

            n, seq_len, _ = all_seq.data.shape
            # update the training loss
            t_3d.update(m_err.cpu().data.item() * n * seq_len, n * seq_len)

            bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
            bar.next()
        bar.finish()
    return t_3d.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)
