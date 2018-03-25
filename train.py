#coding=utf-8
from __future__ import print_function
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from layer import CompletionNet, Discriminator
import torchvision as tv
from PIL import Image
import numpy as np

def data_load(image_path, batch_size, num_workers):
    transforms = tv.transforms.Compose([
                    tv.transforms.Scale(256),
                    tv.transforms.CenterCrop(256),
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    dataset= tv.datasets.ImageFolder(image_path, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            drop_last=True)
    return dataloader

def sample(image_batch, batch_size, use_gpu):
    if use_gpu:
        batch_return = torch.cuda.FloatTensor(batch_size, 3, 256, 256).fill_(-1)
    else:
        batch_return = torch.FloatTensor(batch_size, 3, 256, 256).fill_(-1)
    image_len = len(image_batch[0][0])
    image_hei = len(image_batch[0][0][0])
    if image_len < 256 and image_hei <256:
        k_l = 256//image_len
        k_h = 256//image_hei
        for i in range(k_l):
            for ii in range(k_h):
                batch_return[:, :, i:256:k_l, ii:256:k_h] = image_batch
    if image_len >= 256 and image_hei < 256:
        k_l = image_len//256
        k_h = 256//image_hei
        for i in range(k_h):
            batch_return[:, :, :, i:256:k_h] = image_batch[:, :, 0:image_len:k_l, :]
    if image_len < 256 and image_hei >= 256:
        k_l = 256//image_len
        k_h = image_hei//256
        for i in range(k_l):
            batch_return[:, :, 0:256:k_h, :] = image_batch[:, :, :, 0:image_hei:k_l]
    if image_len >= 256 and image_hei >= 256:
        k_l = image_len//256
        k_h = image_hei//256
        batch_return = image_batch[:, :, 0:image_len:k_l, 0:image_hei:k_h]
    return batch_return[:, :, 0:256, 0:256]

def train(opt):
    # define the net
    net_c = CompletionNet()
    net_d = Discriminator()
    loss_net = []
    loss_epoch = 0.0
    i_n = 0

    # define optimizer
    optimizer_c = torch.optim.Adadelta(net_c.parameters(), rho=0.95)
    optimizer_d = torch.optim.Adadelta(net_d.parameters(), rho=0.95)

    # define loss function
    loss_c = nn.MSELoss()
    # loss_d = nn.CrossEntropyLoss()
    loss_d = nn.BCELoss()

    # load the data, and reshape to 256*256
    data_in = data_load(opt.data_path, opt.batch_size, opt.num_workers)
    if opt.use_gpu:
        net_c.cuda()
        net_d.cuda()
        loss_c.cuda()
        loss_d.cuda()

    image_lenght_list = []
    image_height_list = []
    len_begin_list = []
    hei_begin_list = []
    len_v_list = []
    hei_v_list = []
    len_m_list = []
    hei_m_list = []
    len_mask_m_list = []
    hei_mask_m_list = []
    print('begin trainning')
    for i in range(opt.max_epoch):
        print('epoch:' + str(i))
        for ii, (img, _) in enumerate(data_in):
            # raw data input
            img_raw = Variable(img)
            if opt.use_gpu:
                img_raw = img_raw.cuda()

            # generate white area at the first epoch
            if i == 0:
                image_lenght = len(img_raw[0][0])
                image_height = len(img_raw[0][0][0])
                len_begin = random.randint(0, image_lenght//2)
                hei_begin = random.randint(0, image_height//2)
                len_v = random.randint(image_lenght//8, image_lenght//4)
                hei_v = random.randint(image_height//8, image_height//4)

                # get the center of the white area
                len_m = len_begin + len_v//2
                hei_m = hei_begin + hei_v//2

                # avoid crossing the border
                len_mask_m = min(max(256*len_m//image_lenght, 64), 190)
                hei_mask_m = min(max(256*hei_m//image_height, 64), 190)

                # save the location of the white area
                image_lenght_list.append(image_lenght)
                image_height_list.append(image_height)
                len_begin_list.append(len_begin)
                hei_begin_list.append(hei_begin)
                len_v_list.append(len_v)
                hei_v_list.append(hei_v)
                len_m_list.append(len_m)
                hei_m_list.append(hei_m)
                len_mask_m_list.append(len_mask_m)
                hei_mask_m_list.append(hei_mask_m)
            else:
                image_lenght = image_lenght_list[ii]
                image_height = image_height_list[ii]
                len_begin = len_begin_list[ii]
                hei_begin = hei_begin_list[ii]
                len_v = len_v_list[ii]
                hei_v = hei_v_list[ii]
                len_m = len_m_list[ii]
                hei_m_ = hei_m_list[ii]
                len_mask_m = len_mask_m_list[ii]
                hei_mask_m = hei_mask_m_list[ii]

            # mask the photo input
            img_in = img_raw.clone()
            if opt.use_gpu:
                mask_c = torch.cuda.FloatTensor(opt.batch_size, 3, len_v, hei_v).fill_(1)
            else:
                mask_c = torch.FloatTensor(opt.batch_size, 3, len_v, hei_v).fill_(1)
            img_in[:, :, len_begin:len_begin+len_v, hei_begin:hei_begin+hei_v] = mask_c

            # sample the size of photo to [256, 256]
            img_d_in_raw = img_raw.clone()
            # img_d_in_real = sample(img_d_in_raw, opt.batch_size, opt.use_gpu)
            # img_d_in_real[:, :, len_mask_m-64:len_mask_m+64, hei_mask_m-64:hei_mask_m+64].fill_(1)
            # mask_d = img_d_in_real.clone().fill_(-1)
            # mask_d[:, :, len_mask_m-64:len_mask_m+64, hei_mask_m-64:hei_mask_m+64].fill_(1)
            if opt.use_gpu:
                img_d_dl = Variable(torch.cuda.FloatTensor(opt.batch_size, 3, 128, 128))
                img_d_cl = Variable(torch.cuda.FloatTensor(opt.batch_size, 3, 128, 128))
            else:
                img_d_dl = Variable(torch.FloatTensor(opt.batch_size, 3, 128, 128))
                img_d_cl = Variable(torch.FloatTensor(opt.batch_size, 3, 128, 128))

            # save the local image which include the white area, as the input of local descrimitor
            img_d_dl = img_d_in_raw[:, :, len_mask_m-64:len_mask_m+64, hei_mask_m-64:hei_mask_m+64].clone()
            if i%4<opt.c_epoch:
                optimizer_c.zero_grad()
                img_c_out_raw = net_c(img_in)
                error_c = loss_c(img_c_out_raw, img_raw)
                error_c.backward()
                optimizer_c.step()

            # if opt.c_epoch<=i%5 and i%5<(opt.c_epoch + opt.d_epoch):
            else:
                optimizer_d.zero_grad()
                img_c_out_raw = net_c(img_in)
                img_c_out = img_raw.clone()
                img_c_out[:, :, len_begin:len_begin+len_v, hei_begin:hei_begin+hei_v] = img_c_out_raw[:, :, len_begin:len_begin+len_v, hei_begin:hei_begin+hei_v]
                img_d_cl = img_c_out[:, :, len_mask_m-64:len_mask_m+64, hei_mask_m-64:hei_mask_m+64].clone()
                img_dc_out = net_d(img_d_cl, img_c_out)
                # the output of net_d which the input is the raw photo, the result will be used in analyse later
                img_dr_out = net_d(img_d_dl, img_d_in_raw)
                # img_d_in_c = sample(img_c_out, opt.batch_size, opt.use_gpu)
                img_dr_out_v = img_dr_out.data
                img_dr_error = Variable(img_dr_out_v)
                error_d = loss_d(img_dc_out, img_dr_error)
                error_d.backward()
                optimizer_d.step()
                if i%4 >= (opt.c_epoch + opt.d_epoch):
                    # optimizer_c = torch.optim.Adadelta(net_c.parameters(), rho=0.95)
                    optimizer_c.zero_grad()
                    img_c_out_raw = net_c(img_in)
                    img_c_out = img_raw.clone()
                    img_c_out[:, :, len_begin:len_begin+len_v, hei_begin:hei_begin+hei_v] = img_c_out_raw[:, :, len_begin:len_begin+len_v, hei_begin:hei_begin+hei_v]
                    img_d_cl = img_c_out[:, :, len_mask_m-64:len_mask_m+64, hei_mask_m-64:hei_mask_m+64].clone()
                    img_dc_out = net_d(img_d_cl, img_c_out)
                    img_dr_out = net_d(img_d_dl, img_d_in_raw)
                    img_dr_out_v = img_dr_out.data
                    img_dr_error = Variable(img_dr_out_v)
                    error_c = loss_c(img_c_out_raw, img_raw)
                    error_dc = loss_d(img_dc_out, img_dr_error)
                    # print('error_c:%f, error_dc:%f, ii:%d'  %(error_c, error_dc, ii))
                    error_cd = error_c + opt.alpha*error_dc
                    error_cd.backward()
                    optimizer_c.step()
                    loss_epoch = loss_epoch+error_c
                    # if(i>i_n):
                    #     loss_net.append(loss_epoch)
                    #     i_n = i
                    #     loss_epoch = 0
            if (i+1)%opt.save_epoch==0:
                # print(img_c_out)
                tv.utils.save_image(img_c_out.data, '%s/%s.png' %(opt.save_path, ii))
                tv.utils.save_image(img_in.data, '%s/%s_in.jpg' %(opt.save_path, ii))
                torch.save(net_c.state_dict(), './checkpoints/net_c_%s.pth' %i)
                torch.save(net_d.state_dict(), './checkpoints/net_d_%s.pth' %i)
                optimizer_c = torch.optim.Adadelta(net_c.parameters(), rho=0.95)
                optimizer_d = torch.optim.Adadelta(net_d.parameters(), rho=0.95)
