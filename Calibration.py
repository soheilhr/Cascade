# This code calculates calibration parameters and applies them to calibrate raw ADC data.

from scipy.io import loadmat
import math
import numpy as np
from ipynb.fs.full.Parse_cascade_data import read_raw_bins, data_rearange, plot_RDRA, data_rearangeThrow, data_rearangeSum
from ipynb.fs.full.read_ADC_bin_TDA2_separateFiles import read_ADC_bin_TDA2_separateFiles, readBinFile
import numpy.matlib

#Read and extract variables from calibrateResults_high.mat file
# calibResults = loadmat('calibrateResults_high.mat')
# calibResultStruct = calibResults['calibResult']
# calibResultList = np.array(calibResultStruct.tolist()) # (1, 1, 6)
# AngleMat = (calibResultList[..., 0])[0, 0] # (12, 16)
# RangeMat = (calibResultList[..., 1])[0, 0] # (12, 16)
# PeakValMat = (calibResultList[..., 2])[0, 0] # (12, 16)
# RxMisMatch = (calibResultList[..., 3])[0, 0] # (1, 16)
# TxMisMatch = (calibResultList[..., 4])[0, 0] # (3, 4)
# Rx_fft = (calibResultList[..., 5])[0, 0] # (640, 16, 12)

###
# Data 
adcSampleRate = 4.000000e+06; #Hz/s  
TxToEnable = np.arange(11, -1, -1)
chirpSlope = 7.898600e+13;
Slope_calib = 78986000000000
fs_calib = 8000000; 
Sampling_Rate_sps = adcSampleRate;
calibrationInterp = 5
phaseCalibOnly = 1
adcCalibrationOn = 1
numTX = 12
numRX = 16
dataPlatform = 'TDA2'
interp_fact = 5
targetRange = 3; # Cannot be less than two
num_frames = 359
num_chirp_per_loop = 12 
num_chirp_loops= 32
Samples_per_Chirp = 128
ADC_sample_index = np.arange(Samples_per_Chirp)
num_RX_antennas=4
rev_TX = 1
rev_RX = 1

folder_name='/Users/cc/desktop/research/Calibration/cr_3m_awf_horiz/'
fnames=[
        folder_name+'/master_0000_data.bin',
        folder_name+'/slave1_0000_data.bin',
        folder_name+'/slave2_0000_data.bin',
        folder_name+'/slave3_0000_data.bin']

class fileNameCascadeStruct:
    def __init__(self, dataFolderName, master, slave1, slave2, slave3):
        self.dataFolderName = dataFolderName
        self.master = master
        self.slave1 = slave1
        self.slave2 = slave2
        self.slave3 = slave3

        
class calibResultStruct:
    def __init__(self, AngleMat, RangeMat, PeakValMat, RxMismatch, TxMismatch, Rx_fft):
        self.AngleMat = AngleMat
        self.RangeMat = RangeMat
        self.PeakValMat = PeakValMat
        self.RxMismatch = RxMismatch
        self.TxMismatch = TxMismatch
        self.Rx_fft = Rx_fft
        
fileNameCascade = fileNameCascadeStruct("/Users/cc/desktop/research/Calibration/cr_3m_awf_horiz/", "master_0000_data.bin", "slave1_0000_data.bin", "slave2_0000_data.bin", "slave3_0000_data.bin")

###
# Functions

"""
Function: hann_local.m
Implements hanning window

Input:
   len: Window length
Output:
   win: Generated windowing coefficiens.
"""
def hann_local(len):
    vec = np.arange(1,len + 1)
    win = (vec.conj().T) / (len + 1)
    win = 0.5 - 0.5*np.cos(2 * np.pi * win)
    return win


"""
Function: radar_fft_find_peak
This function finds the peak location and complex value that corresponds to the
calibration target.

Input
   calibrationInterp
   Rx_Data: input adc data 
   range_bin_search_min: start of the range bin to search for peak
   range_bin_search_max: end of the range bin to search for peak


 Output:
   Angle_FFT_Peak: Phase at bin specified by FRI_fixed OR Phase at
                         highest peak after neglecting DC peaks
   Val_FFT_Peak: Complex value at bin specified by FRI_fixed OR
                   Complex value at highest peak after neglecting DC peaks
   Fund_range_Index: Bin number for highest peak after neglecting DC
                     peaks
   Rx_fft: Complex FFT values
"""
def radar_fft_find_peak(calibrationInterp, Rx_Data,range_bin_search_min,range_bin_search_max):

    Effective_Num_Samples = len(Rx_Data) # num samples per chirp

    wind = hann_local(Effective_Num_Samples)
    wind = wind / np.sqrt(np.mean(wind**2))
    interp_fact = calibrationInterp
    Rx_Data_prefft = Rx_Data * wind
    Rx_fft = np.fft.fft(Rx_Data_prefft, n = interp_fact*Effective_Num_Samples) # INTERP_FACT * EFFECTIVE_NUM_SAMPLES MUST = 640
    
    Rx_fft_searchwindow = abs(Rx_fft[range_bin_search_min - 1: range_bin_search_max + 1])
    Fund_range_Index = Rx_fft_searchwindow[:].argmax(0)
    Fund_range_Index = Fund_range_Index + range_bin_search_min - 1
    
    Angle_FFT_Peak = np.angle(Rx_fft[Fund_range_Index], deg = True) # Multiple copies of the same thing?
    Val_FFT_Peak = Rx_fft[Fund_range_Index] # Multiple copies of the same thing?
    
    return Rx_fft, Angle_FFT_Peak, Val_FFT_Peak, Fund_range_Index
    

"""
Function: Average_Ph.m

Output: 
    Avg_Ph: Average of the angles in RADIAN units

Input: 
    Ph_Arr_Rad: an array of complex angles in RADIAN units
"""
def Average_Ph(Ph_Arr_Rad):
    diff_Ph = np.angle(np.exp(1j*(Ph_Arr_Rad - Ph_Arr_Rad[0])))
    Ph_Arr_Rad = Ph_Arr_Rad[0] + diff_Ph
    Avg_Ph = np.mean(Ph_Arr_Rad, axis=0)
    return Avg_Ph

###
# Main: Find calibration parameters
range_bin_search_min = round((Samples_per_Chirp) * interp_fact * 
                             ((targetRange - 2) * 2 * Slope_calib / 
                              (3e8 * Sampling_Rate_sps)) + 1)

range_bin_search_max = round((Samples_per_Chirp) * interp_fact * 
                             ((targetRange + 2) * 2 * Slope_calib / 
                              (3e8 * Sampling_Rate_sps)) + 1)
if (dataPlatform == 'TDA2'):
    numRXPerDevice = 4 
    radar_data_Rxchain = read_ADC_bin_TDA2_separateFiles(fileNameCascade, num_frames, 
                                                         Samples_per_Chirp, num_chirp_per_loop, 
                                                         num_chirp_loops, num_RX_antennas, numRXPerDevice)
else:
    raise Exception("Not supported data capture platform!") 
        
# Raw Data, Shape: (128, 32, 16, 12)
radar_data_Rxchain = radar_data_Rxchain[:, :, :, TxToEnable]

AngleMat = np.zeros((numTX, numRX), np.float64) # (12, 16)
RangeMat = np.zeros((numTX, numRX), np.int64) # (12, 16)
PeakValMat = np.zeros((numTX, numRX), numpy.complex128) # (12, 16)
Rx_fft = np.zeros((calibrationInterp * Samples_per_Chirp, numRX, numTX), np.complex128) # (640, 16, 12)
RxMismatch = np.zeros((1, numRX)) # (1, 16)
TxMismatch = np.zeros((1, numTX)) # (!, 12)

for iTX in range(0, numTX):
    for iRx in range(0, numRX):
        Rx_Data = np.mean(radar_data_Rxchain[:, :, iRx, iTX], 1) # Average chirps within a frame

        Rx_fft[:, iRx, iTX], Angle, Val_peak, Rangebin = radar_fft_find_peak(calibrationInterp, Rx_Data, range_bin_search_min, range_bin_search_max)
        AngleMat[iTX, iRx] = Angle
        RangeMat[iTX, iRx] = Rangebin

        PeakValMat[iTX, iRx] = Val_peak

TX_ind_calib = TxToEnable[0]; 

ind = AngleMat
Num_Rxs = len(ind[0,:])
temp = ind[:, 0]
temp = ind - np.matlib.repmat(temp, Num_Rxs, 1).T

for i in range(0, Num_Rxs):
    RxMismatch[0, i] = Average_Ph(temp[:, i] * (np.pi / 180)) * 180 / np.pi
    
Num_Txs = len(ind[:, 0])
temp = ind[TX_ind_calib, :]
temp = ind - np.matlib.repmat(temp, Num_Txs, 1)

for i in range(0, Num_Txs):
    TxMismatch[0, i] = Average_Ph(temp[i, :] * np.pi/180) * 180 / np.pi
TxMismatch = np.reshape(TxMismatch, [3, 4], order = 'F')

"""
Calibration Parameters Found:
    AngleMat (12 x 16): Phase of FFT peak index matrix
    RangeMat (12 x 16): Peak index from all 192 channels for frequency calibration step
                    Should be 1 less than matlab bc of index differences btwn matlab and python
    PeakValMat (12 x 16): Complex calibration matrix with complex values at peaks from all 
            192 channels for phase and amplitude calibration
    RxMismatch (1 x 16): RX phase calibration vector
    TxMismatch (3 x 4): TX phase calibration vector
    Rx_fft (640 x 16 x 12): Complex matrix

"""
calibResult = calibResultStruct(AngleMat, RangeMat, PeakValMat, RxMismatch, TxMismatch, Rx_fft)

###
# Apply Calibration parameters to raw data
TX_ref = TxToEnable[0]

outData = np.zeros((Samples_per_Chirp, num_chirp_loops, Num_Rxs, Num_Txs), np.complex64)
for iTX in range(0, Num_Txs):
    TX_ind = TxToEnable[iTX]
    
    # Calculate frequency calibration vector
    freq_calib = (RangeMat[TX_ind:TX_ind + 1, :] - RangeMat[TX_ref, 0]) * (fs_calib / adcSampleRate) *(chirpSlope / Slope_calib)
    freq_calib = 2 * np.pi * freq_calib / (Samples_per_Chirp * calibrationInterp)
    correction_vec = np.exp(-1j * (np.arange(Samples_per_Chirp)[:, np.newaxis]) * freq_calib)
    
    # Apply frequency calibration vector to calibrate data
    outData1TX = radar_data_Rxchain[:, :, :, iTX] * correction_vec[:, None, :]
    
    # Calculate phase calibration vector
    phase_calib_matrix = PeakValMat[TX_ref,0] / PeakValMat[TX_ind, :]
    
    # Remove amplitude calibration
    if phaseCalibOnly == 1:
        phase_calib_matrix = phase_calib_matrix / np.abs(phase_calib_matrix)
    
    # Apply calibration to obtain outData, the final phase and frequency calibrated data
    outData[:, :, :, iTX] = outData1TX * phase_calib_matrix[None, None, :]

