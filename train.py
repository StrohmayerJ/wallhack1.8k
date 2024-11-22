import torch
import torchvision.models as models
import torch.nn as nn
import datasets as data
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train(opt):
    if torch.cuda.is_available():
        device = "cuda:"+opt.device
    else:
        device = "cpu"

    print("Loading Wallhack1.8k dataset...")
    def train_val_test_dataset(dataset):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=False, random_state=42)
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5, shuffle=False, random_state=42)
        return Subset(dataset, train_idx), Subset(dataset, val_idx),Subset(dataset, test_idx),train_idx

    def load_sequence(scenario,antenna, sequence):
        sequenceDataset = data.WallhackDataset(f"{opt.data}/{scenario}/{antenna}/{sequence}.csv", opt=opt)
        train, val, test,train_idx = train_val_test_dataset(sequenceDataset)
        trainLabels = sequenceDataset.c[train_idx]
        return train, val, test, trainLabels

    sequences = ["b1","w1","w2","w3","w4","w5","ww1","ww2","ww3","ww4","ww5"]
    trainSubsets = []
    valSubsets = []
    testSubsets = []

    # perform 8:1:1 split on each sequence and combine them into train, val and test datasets
    for s in sequences:
        train, val, test,l = load_sequence(opt.scenario,opt.antenna, s)
        trainSubsets.append(train)
        valSubsets.append(val)
        testSubsets.append(test)

    # create training, validation and test dataloader
    datasetTrain = torch.utils.data.ConcatDataset(trainSubsets)
    datasetVal = torch.utils.data.ConcatDataset(valSubsets)
    datasetTest = torch.utils.data.ConcatDataset(testSubsets) 
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,batch_size=opt.bs,num_workers=opt.workers,drop_last=True,shuffle=True)
    dataloaderVal = torch.utils.data.DataLoader(datasetVal,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    dataloaderTest = torch.utils.data.DataLoader(datasetTest,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)

    '''
    # overwrite dataloaderTest for cross-domain testing
    SCENARIO = 'LOS' # ['LOS', 'NLOS']
    ANTENNA = 'BQ' # ['BQ', 'PIFA']
    testSubsets = []
    for s in sequences:
        testSubsets.append(data.WallhackDataset(f"{opt.data}/{SCENARIO}/{ANTENNA}/{s}.csv", opt=opt))
    datasetTest = torch.utils.data.ConcatDataset(testSubsets)
    dataloaderTest = torch.utils.data.DataLoader(datasetTest,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    '''

    # create dummy resnet18 model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # set number of input channels to 1
    model.fc = nn.Linear(512, 3) # set number of output classes to 3
    model.to(device)

    print("Training...")
    for epoch in tqdm(range(opt.epochs), desc='Epochs', unit='epoch'):
        
        # training loop
        for batch in tqdm(dataloaderTrain, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='batch', leave=False):
            feature_window, c = [x.to(device) for x in batch]
            feature_window = feature_window.float()
            prediction = model(feature_window) # TODO: add your model for training here

        # calidation loop
        with torch.no_grad():
            for batch in tqdm(dataloaderVal):
                feature_window, c = [x.to(device) for x in batch]
                feature_window = feature_window.float()
                prediction = model(feature_window) # TODO: add your model for validation here

    # test loop
    print("Testing...")
    with torch.no_grad():
        for batch in tqdm(dataloaderTest):
            feature_window, c = [x.to(device) for x in batch]
            feature_window = feature_window.float()
            prediction = model(feature_window) # TODO: add your model for testing here

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/wallhack1.8k', help='directory of the wallhack1.8k dataset')
    parser.add_argument('--ws', type=int, default=401, help='feature window size (i.e. the number of WiFi packets)')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--scenario', default='LOS', type=str, help='Type of scenario (LOS or NLOS)')
    parser.add_argument('--antenna', default='BQ', type=str, help='Type of antenna (BQ or PIFA)')
    opt = parser.parse_args()

    train(opt)
    print("Done!")
    torch.cuda.empty_cache()


















