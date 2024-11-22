import math
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Subcarrier selection for wallhack1.8k dataset
# 52 L-LTF subcarriers 
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_index += [i for i in range(6, 32)]
csi_vaid_subcarrier_index += [i for i in range(33, 59)]
# 56 HT-LTF: subcarriers 
#csi_vaid_subcarrier_index += [i for i in range(66, 94)]     
#csi_vaid_subcarrier_index += [i for i in range(95, 123)] 
CSI_SUBCARRIERS = len(csi_vaid_subcarrier_index) 

# wallhack1.8k dataset class
class WallhackDataset(Dataset):
    def __init__(self, dataPath,opt):
        self.dataPath = dataPath
        folderEndIndex = dataPath.rfind('/')
        self.dataFolder = self.dataPath[0:folderEndIndex] 
        self.fileName = os.path.basename(dataPath)
        self.fileName = self.fileName[:self.fileName.rfind('.')]
        self.imagePath = os.path.dirname(dataPath)
        self.windowSize = opt.ws
        assert self.windowSize % 2 == 1
        self.windowSizeH = math.ceil(self.windowSize/2)

        # read data from .csv file
        data = pd.read_csv(dataPath)
        csi = data['data']
        self.x = data['x']
        self.y = data['y']
        self.z = data['z']
        self.c = data['class']

        # pre-compute or load complex CSI cache to speed up training
        if os.path.exists(self.dataFolder + f"/"+self.fileName+".npy"):
            csiComplex = np.load(self.dataFolder + f"/"+self.fileName+".npy")
        else:
            csiComplex = np.zeros(
                [len(csi), CSI_SUBCARRIERS], dtype=np.complex64)
            for s in tqdm(range(len(csi))):
                for i in range(CSI_SUBCARRIERS):
                    sample = csi[s][1:-1].split(',')
                    sample = np.array([int(x) for x in sample])
                    csiComplex[s][i] = complex(sample[csi_vaid_subcarrier_index[i] * 2], sample[csi_vaid_subcarrier_index[i] * 2 - 1])
            np.save(self.dataFolder + f"/"+self.fileName+".npy", csiComplex)

        # extract feature from CSI
        self.features = np.abs(csiComplex) # amplitude
        #self.features = np.angle(csiComplex) # phase

        # compute number of samples excluding border regions
        self.dataSize = len(self.features)-self.windowSize  

    def __len__(self):
        return self.dataSize

    def __getitem__(self, index):
        index = index + self.windowSizeH # add index offset to avoid border regions
        c = self.c[index] # class labels in Wallhack2k dataset start with 0
        featureWindow = self.features[index-self.windowSizeH:index+self.windowSizeH-1] # get feature window
        featureWindow = np.transpose(featureWindow, (1, 0)) # transpose to [self.windowSize,CSI_SUBCARRIERS]
        featureWindow = np.expand_dims(featureWindow, axis=0) # add channel dimension

        return featureWindow, c
