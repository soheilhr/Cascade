#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:37:51 2020

@author: soheil
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def read_raw_bins(fnames,num_frames=359,num_chirps=16,num_chirp_loops=12,samples_per_chirp=128,num_RX_antennas=4):
    data_shape=(num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas,2)
    dcube_list=[]
    for fidx, fname in enumerate(fnames):
        print(fname)
        dat_raw=np.fromfile(fname,dtype=np.uint16)    
        dat=dat_raw.reshape(data_shape)
        dat=dat[...,0]+dat[...,1]*1j
        dcube_list.append(dat.transpose((0,1,3,2,4)))
    return dcube_list

def data_rearange(dcube_list,num_frames=359,num_chirps=16,num_chirp_loops=12,samples_per_chirp=128,num_RX_antennas=4):
    dcube=np.concatenate(dcube_list,-1)
    idxs=np.array([8,7,6,5,12,11,10,9,4,3,2,1,16,15,14,13])-1
    dcube=dcube[:,:,:,:,idxs]
    dcube=dcube[:,:,:,3:,:]
    dcube=np.concatenate([
    dcube[:,:,:,:3,:4].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,:3,4:8].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,-3:,:4].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,-3:,4:8].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,::2,8:].reshape((num_frames,num_chirps,samples_per_chirp,5*2*num_RX_antennas))
    ],-1)        
    return dcube

def plot_RDRA(dat,plot_name='',fout='./'):
    for idx in range(dat.shape[0]):
        DAT=np.fft.fftn(dat[idx,...])#,s=(64,256,128))        
        DAT=np.fft.fftshift(DAT,(0,2))
        plt.imsave(fout+'RA_'+plot_name+'_{:03}.jpg'.format(idx),np.log(1+np.sum(np.abs(DAT),0)))
        plt.imsave(fout+'RD_'+plot_name+'_{:03}.jpg'.format(idx),np.log(1+np.sum(np.abs(DAT),2)))




folder_name='/home/soheil/Downloads/jun18_sar4_p16_const_vel_1m5_2m_indoor_1-20200623T190035Z-001/jun18_sar4_p16_const_vel_1m5_2m_indoor_1/'
folder_name='/home/soheil/Downloads/'
fnames=[
        folder_name+'/master_0000_data.bin',
        folder_name+'/slave1_0000_data.bin',
        folder_name+'/slave2_0000_data.bin',
        folder_name+'/slave3_0000_data.bin']

(127,64,12,256,4,2)

num_frames=127
num_chirps=64
num_chirp_loops=12
samples_per_chirp=256
num_RX_antennas=4

dcube_list = read_raw_bins(fnames,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas)
dcube = data_rearange(dcube_list,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas)

plot_RDRA(dcube[:,:,:,:])


