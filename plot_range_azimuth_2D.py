"""
Created on Thu Jul 16 2020

@author: Christine
"""

import numpy as np
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
from pathlib import Path
from ipynb.fs.full.Parse_cascade_data import read_raw_bins, data_rearange, plot_RDRA

# Function to plot range and azimuth heat map

#input
#   range_resolution: range resolution to calculate axis to plot
#   radar_data_pre_3dfft: input 3D matrix, rangeFFT x DopplerFFT x virtualArray
#   TDM_MIMO_numTX: number of TXs used for processing
#   numRxAnt: : number of RXs used for processing
#   antenna_azimuthonly: azimuth array ID
#   LOG: 1:plot non-linear scale, ^0.4 by default
#   STATIC_ONLY: 1 = plot heatmap for zero-Doppler; 0 = plot heatmap for nonzero-Doppler
#   PLOT_ON: 1 = plot on; 0 = plot off
#   minRangeBinKeep: start range index to keep
#   rightRangeBinDiscard: number of right most range bins to discard

#output
#   mag_data_static: zero Doppler range/azimuth heatmap
#   mag_data_dynamic: non-zero Doppler range/azimuth heatmap
#   y_axis: y axis used for visualization
#   x_axis: x axis used for visualization

# Function returns mag_data_static, mag_data_dynamic, y_axis, x_axis
def plot_range_azimuth_2D(range_resolution, radar_data_pre_3dfft, TDM_MIMO_numTX, 
                          numRxAnt, antenna_azimuthonly, LOG, STATIC_ONLY, PLOT_ON, 
                          minRangeBinKeep,  rightRangeBinDiscard):
    dopplerFFTSize = radar_data_pre_3dfft.shape[1]
    rangeFFTSize = radar_data_pre_3dfft.shape[0]
    angleFFTSize = 256
    # Ratio to decide energy threshold used to pick non-zero Doppler bins
    ratio = 0.5
    DopplerCorrection = 0; 
    
    if (DopplerCorrection == 1):
        # Add Doppler correction before generating the heatmap
        np.zeros((numRxAnt * numRxAnt, dopplerFFTSize, numRxAnt * TDM_MIMO_numTX))
        for dopplerInd in range(dopplerFFTSize):
            deltaPhi = 2 * pi * (dopplerInd - 1 - deopplerFFTSize / 2) / (TDM_MIMO_numTX*dopplerFFTSize) # Does it need the -1?
            sig_bin_org = np.squeeze(radar_data_pre_3dfft[:, dopplerInd, :]) # Can delete last : for default : for rest
            for i_TX in range(TDM_MIMO_numTX):
                RX_ID = np.arange((i_TX - 1) * numRxAnt + 1, i_TX * numRxAnt)
                corVec = repmat(exp(-1j * (i_TX - 1) * deltaPhi), rangeFFTSize, numRxAnt) 
                radar_data_pre_3dfft_DopCor[:, dopplerInd, RX_ID] = sig_bin_org[:, RX_ID] * corVec
        
        radar_data_pre_3dfft = radar_data_pre_3dfft_DopCor
        
    radar_data_pre_3dfft = radar_data_pre_3dfft[:, :, antenna_azimuthonly, 0]
    
    # radar_data_angle_range_win = numpy.fft.fft(radar_data_pre_3dfft * angleWin, angleFFTSize, 2)
    radar_data_angle_range = np.fft.fft(radar_data_pre_3dfft, n = angleFFTSize, axis = 2)
    n_angle_fft_size = radar_data_angle_range.shape[2]
    n_range_fft_size = radar_data_angle_range.shape[0]
    
    #Decide non-zero doppler bins to be used for dynamic range-azimuth heatmap
    DopplerPower = np.sum(np.mean((np.absolute(radar_data_pre_3dfft)), axis = 2), axis = 0) 
    DopplerPower_noDC = np.concatenate([DopplerPower[0:(dopplerFFTSize//2 - 1)], DopplerPower[(dopplerFFTSize//2 + 2):]])
    peakVal = np.amax(DopplerPower_noDC)
    peakInd = (np.where(DopplerPower_noDC == peakVal))[0][0]
    threshold = peakVal * ratio
    indSel = np.where(DopplerPower_noDC > threshold)[0]
    for ii in range(len(indSel)):
        if (indSel[ii] > (dopplerFFTSize / 2 - 2)): 
            indSel[ii] = indSel[ii] + 3

    radar_data_angle_range_dynamic = np.squeeze(np.sum(np.absolute(radar_data_angle_range[:, indSel, :]), axis = 1))
    radar_data_angle_range_Static = np.squeeze(np.sum(np.absolute(radar_data_angle_range[:, (dopplerFFTSize // 2), np.newaxis, :]), axis = 1))
    # radar_data_angle_range_Static_win = np.squeeze(np.sum(np.absolute(radar_data_angle_range_win[:, dopplerFFTSize // 2, np.newaxis, :]), axis = 1))
    
    indices_1D = np.arange(minRangeBinKeep, n_range_fft_size - rightRangeBinDiscard + 1)
    max_range = (n_range_fft_size - 1) * range_resolution
    max_range = max_range / 2
    d = 1;

    # Generate range/angleFFT for zeroDoppler and non-zero Doppler respectively
    radar_data_angle_range_dynamic = np.fft.fftshift(radar_data_angle_range_dynamic, axes = 1)
    radar_data_angle_range_Static = np.fft.fftshift(radar_data_angle_range_Static, axes = 1)
    # radar_data_angle_range_static_win = np.fft.fftshift(radar_data_angle_range_Static_win, axes = 1)
    
    sine_theta = -2 * (np.arange((-n_angle_fft_size / 2), (n_angle_fft_size / 2) + 1) / n_angle_fft_size) / d
    cos_theta = np.sqrt(1 - np.power(sine_theta, 2))
    [R_mat, sine_theta_mat] = np.meshgrid(indices_1D * range_resolution, sine_theta)
    [_, cos_theta_mat] = np.meshgrid(indices_1D, cos_theta)

    x_axis = R_mat * cos_theta_mat
    y_axis = R_mat * sine_theta_mat
    mag_data_dynamic = np.squeeze(np.absolute(np.append(radar_data_angle_range_dynamic[indices_1D, :], 
                                                        radar_data_angle_range_dynamic[indices_1D, 0, np.newaxis], 
                                                        axis = 1))) 
    mag_data_static = np.squeeze(np.absolute(np.append(radar_data_angle_range_Static[indices_1D, :], 
                                                        radar_data_angle_range_Static[indices_1D, 0, np.newaxis], 
                                                        axis = 1))) 
    # mag_data_static_win = np.squeeze(np.absolute(np.append(radar_data_angle_range_Static[indices_1D, :], 
    #                                                   radar_data_angle_range_Static[indices_1D, 0, np.newaxis], 
    #                                                   axis = 1)))
  
    mag_data_dynamic = np.transpose(mag_data_dynamic)
    mag_data_static = np.transpose(mag_data_static)
    
    if (PLOT_ON):
        log_plot = LOG
        fig = plt.figure()
        ax = Axes3D(fig)
        
        if (STATIC_ONLY == 1):
            if (log_plot): 
                
                surf = ax.plot_surface(y_axis, x_axis, np.power(mag_data_static, 0.4), cmap = 'gray')
                
            else:
                surf = ax.plot_surface(y_axis, x_axis, np.absolute(mag_data_static), edgecolor = 'none', cmap = 'gray')
        else:
            if (log_plot):
                surf = ax.plot_surface(y_axis, x_axis, np.power(mag_data_dynamic, 0.4), edgecolor = 'none', cmap = 'gray')
            else:
                surf = ax.plot_surface(y_axis, x_axis, np.absolute(mag_data_dynamic), edgecolor = 'none', cmap = 'gray')
          
        # ax.view_init(azim = 270, elev = 90) # for top view
        ax.set_xlabel('meters')
        ax.set_ylabel('meters')
        plt.show() 
        
    return mag_data_static, mag_data_dynamic, y_axis, x_axis
    
       

# Read data files and input data into dcube
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

dcube_list = read_raw_bins(fnames,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas)
dcube = data_rearange(dcube_list,num_frames,num_chirps,num_chirp_loops,samples_per_chirp,num_RX_antennas)

plot_RDRA(dcube[:,:,:,:])

# Analyze and graph dcube data
range_resolution = 0.1
TDM_MIMO_numTX = 12
numRxAnt = 4
antenna_azimuthonly = np.arange(16)
LOG = 1
STATIC_ONLY = 0
PLOT_ON = 1
minRangeBinKeep = 1
rightRangeBinDiscard = 1

mag_data = plot_range_azimuth_2D(range_resolution, dcube,
                     TDM_MIMO_numTX, numRxAnt, antenna_azimuthonly, LOG, STATIC_ONLY, 
                     PLOT_ON, minRangeBinKeep,  rightRangeBinDiscard)



    
    
    
    
    
            
        
    
    
    
    
    
    
    
    







