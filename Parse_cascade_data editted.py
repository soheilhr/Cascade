
"""
Created on Thu Jul 16 17:37:51 2020

@author: Soheil, editted by Christine
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def read_raw_bins(fnames,num_frames=359,num_chirps=16,num_chirp_loops=12,samples_per_chirp=128,num_RX_antennas=4):
    data_shape=(num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas,2)
    dcube_list=[]
    for fidx, fname in enumerate(fnames):
        dat_raw=np.fromfile(fname,dtype=np.uint16) 
        dat=dat_raw.reshape(data_shape)        
        dat=dat[...,0]+dat[...,1]*1j
        dcube_list.append(dat.transpose((0,1,3,2,4)))
    return dcube_list

def data_rearange(dcube_list,num_frames=359,num_chirps=16,num_chirp_loops=12,samples_per_chirp=128,num_RX_antennas=4, rev_TX=1, rev_RX=1):
    dcube=np.concatenate(dcube_list,-1)
    if (rev_RX == 0) :
        idxs=np.array([8,7,6,5,12,11,10,9,4,3,2,1,16,15,14,13])-1 # the og RX
    else: 
        idxs=np.array([13, 14, 15, 16, 1, 2, 3,4, 9, 10, 11, 12, 5, 6, 7, 8])-1 # Reversed RX
    
    dcube=dcube[:,:,:,:,idxs]
    if (rev_TX == 1):
        idxsTRev = np.arange(start=12, stop=0, step=-1)-1 # Rev TX
        dcube=dcube[:,:,:,idxsTRev,:] #Rev TX
    dcube=dcube[:,:,:,3:,:]
    return dcube
    
# Convolution, no repeats
def data_rearangeThrow(dcube,num_frames=359,num_chirps=16,num_chirp_loops=12,
                  samples_per_chirp=128,num_RX_antennas=4):
    dcube=np.concatenate([
    dcube[:,:,:,:3,:4].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,:3,4:8].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,-3:,:4].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,-3:,4:8].reshape((num_frames,num_chirps,samples_per_chirp,3*num_RX_antennas)),
    dcube[:,:,:,::2,8:].reshape((num_frames,num_chirps,samples_per_chirp,5*2*num_RX_antennas))
    ],-1)    
    return dcube

# Convolution, sum repeats
def data_rearangeSum(dcube,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas):      
    newcube = np.concatenate([dcube[:,:,:,8,:4], 
                        dcube[:,:,:,7,:4], 
                        dcube[:,:,:,6,:3], 
                        ((dcube[:,:,:, 6, 3] + dcube[:,:,:,8, 4])[:,:,:, np.newaxis])], 
                        axis = -1)
    for i in reversed(range(6)):
        newcube = np.concatenate([newcube, 
                        dcube[:,:,:,i,:3] + dcube[:,:,:,i+3, 5:8], 
                        (dcube[:,:,:,i,3] + dcube[:,:,:,i+2, 4])[:,:,:,np.newaxis]], 
                        axis = -1)
    newcube = np.concatenate([newcube, 
                        dcube[:,:,:,2, 5:8], 
                        dcube[:,:,:,1, 4:8], 
                        dcube[:,:,:,0, 4:7], 
                        ((dcube[:,:,:, 0, 7] + dcube[:,:,:,8, 8])[:,:,:, np.newaxis]), 
                        dcube[:,:,:,8, 9:12]],
                        axis = -1)
    for i in reversed(range(1, 9)):
        newcube = np.concatenate([newcube, 
                        dcube[:,:,:,i,12:16] + dcube[:,:,:,i-1, 8:12]], 
                        axis = -1)
    
    newcube = np.append(newcube, dcube[:,:,:,0, 12:16], axis = -1)
    return newcube # size should be (127, 64, 256, 86)
    

def plot_RDRA(dat,plot_name='',fout='./'):
    for idx in range(dat.shape[0]):
        DAT=np.fft.fftn(dat[idx,...])#,s=(64,256,128))        
        DAT=np.fft.fftshift(DAT,(0,2))
        plt.imsave(fout+'RA_'+plot_name+'_{:03}.jpg'.format(idx),np.log(1+np.sum(np.abs(DAT),0)))
        plt.imsave(fout+'RD_'+plot_name+'_{:03}.jpg'.format(idx),np.log(1+np.sum(np.abs(DAT),2)))
        

        
# Main
folder_name='/Users/cc/downloads/jun05_1130_cas1'
folder_name='/Users/cc/downloads/jun05_1130_cas1'
fnames=[
        folder_name+'/master_0000_data.bin',
        folder_name+'/slave1_0000_data.bin',
        folder_name+'/slave2_0000_data.bin',
        folder_name+'/slave3_0000_data.bin']

num_frames=127
num_chirps=64
num_chirp_loops=12
samples_per_chirp=256
num_RX_antennas=4
rev_TX = 1
rev_RX = 1

# Rearrange data to prep for conv
dcube_list = read_raw_bins(fnames,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas)
dcube = data_rearange(dcube_list,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas, rev_TX, rev_RX)

# Throw Conv
convoluted_dcube = data_rearangeThrow(dcube,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas)
plot_RDRA(convoluted_dcube[:,:,:,:])

# Sum Conv
dcubeNew = data_rearangeSum(dcube,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas)
plot_RDRA(dcubeNew[:,:,:,:])


