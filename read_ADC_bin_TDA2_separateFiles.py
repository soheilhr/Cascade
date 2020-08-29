# This code reads data files and shapes the data

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

"""
Description : Given a folder name and fileIdx, return the FileName 
              corresponding to the Master, slave1, slave2, slave3, slave3 binary files
Input : 
    folderName : The name of the folder in which the data is saved
"""
class fileNameCascadeStruct:
    def __init__(self, dataFolderName, master, slave1, slave2, slave3):
        self.dataFolderName = dataFolderName
        self.master = master
        self.slave1 = slave1
        self.slave2 = slave2
        self.slave3 = slave3
        
"""
Function: read_ADC_bin_TDA2_separateFiles
    Read raw adc data with MIMO
Output:
    radar_data_RXchain: Shaped, raw ADC data
"""
def read_ADC_bin_TDA2_separateFiles(fileNameCascade, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, numDevices):
    dataFolder = fileNameCascade.dataFolderName
    fileFullPath_master = os.path.join(dataFolder, fileNameCascade.master)
    fileFullPath_slave1 = os.path.join(dataFolder, fileNameCascade.slave1)
    fileFullPath_slave2 = os.path.join(dataFolder, fileNameCascade.slave2)
    fileFullPath_slave3 = os.path.join(dataFolder, fileNameCascade.slave3)

    radar_data_Rxchain_master = readBinFile(fileFullPath_master, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, numDevices)
    radar_data_Rxchain_slave1 = readBinFile(fileFullPath_slave1, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, numDevices)
    radar_data_Rxchain_slave2 = readBinFile(fileFullPath_slave2, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, numDevices)
    radar_data_Rxchain_slave3 = readBinFile(fileFullPath_slave3, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, numDevices)

    # radar_data_Rxchain[:, :, 4:7, :] = radar_data_Rxchain_master
    # radar_data_Rxchain[:, :, 12:15, :] = radar_data_Rxchain_slave1
    # radar_data_Rxchain[:, :, 8:11, :] = radar_data_Rxchain_slave2
    # radar_data_Rxchain[:, :, 0:3, :] = radar_data_Rxchain_slave3
    
    # Arranged based on Master RxChannels, Slave1 RxChannels, slave2 RxChannels, slave3 RxChannels
    # The RX channels are re-ordered according to "TI_Cascade_RX_ID" defined in
    # "module_params.m"
        
    radar_data_Rxchain = np.zeros((numSamplePerChirp, numLoops, numRXPerDevice * numDevices, numChirpPerLoop), dtype = np.complex64) # TODO: use the variables given

    radar_data_Rxchain[:, :, 0:4, :] = radar_data_Rxchain_master
    radar_data_Rxchain[:, :, 4:8, :] = radar_data_Rxchain_slave1
    radar_data_Rxchain[:, :, 8:12, :] = radar_data_Rxchain_slave2
    radar_data_Rxchain[:, :, 12:16, :] = radar_data_Rxchain_slave3
    
    return radar_data_Rxchain

"""
Function:readBinFile
    Read files and store data in adcData1Complex
Output:
    adcData1Complex: Raw ADC Data
"""
def readBinFile(fileFullPath, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, numDevices):
    Expected_Num_SamplesPerFrame = numSamplePerChirp * numChirpPerLoop * numLoops * numRXPerDevice * 2; 

    fp = open(fileFullPath, "rb")
    fp.seek((frameIdx - 1) * Expected_Num_SamplesPerFrame * 2, 0)
    
    adcData1 = np.fromfile(fp, dtype = np.int16, count = Expected_Num_SamplesPerFrame)
    adcData1 = adcData1[0::2] + 1j * adcData1[1::2]

    adcData1Complex = np.reshape(adcData1, (numRXPerDevice, numSamplePerChirp, numChirpPerLoop, numLoops), order = 'F')
    adcData1Complex = np.transpose(adcData1Complex, (1, 3, 0, 2)) 
    
    fp.close()
    return adcData1Complex

###
# Main
fileNameCascade = fileNameCascadeStruct("/Users/cc/downloads/jun05_1130_cas1", "master_0000_data.bin", "slave1_0000_data.bin", "slave2_0000_data.bin", "slave3_0000_data.bin")
frameIdx = 127
numSamplePerChirp = 256
numChirpPerLoop = 64
numLoops = 12
numRXPerDevice = 4
numDevices = 4
arrayTest = read_ADC_bin_TDA2_separateFiles(fileNameCascade, frameIdx, numSamplePerChirp, numChirpPerLoop, numLoops, numRXPerDevice, numDevices)

