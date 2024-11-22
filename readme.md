# PyTorch Dataloader for the Wallhack1.8k Dataset

### Papers
**Strohmayer, J., and Kampel, M.** (2024). “Data Augmentation Techniques for Cross-Domain WiFi CSI-Based Human Activity Recognition”, In IFIP International Conference on Artificial Intelligence Applications and Innovations (pp. 42-56). Cham: Springer Nature Switzerland, doi: https://doi.org/10.1007/978-3-031-63211-2_4

**Strohmayer J. and Kampel M.**, “Directional Antenna Systems for Long-Range Through-Wall Human Activity Recognition,” 2024 IEEE International Conference on Image Processing (ICIP), Abu Dhabi, United Arab Emirates, 2024, pp. 3594-3599, doi: https://doi.org/10.1109/ICIP51287.2024.10647666.

BibTeX:
```
@inproceedings{strohmayer2024data,
  title={Data augmentation techniques for cross-domain wifi csi-based human activity recognition},
  author={Strohmayer, Julian and Kampel, Martin},
  booktitle={IFIP International Conference on Artificial Intelligence Applications and Innovations},
  pages={42--56},
  year={2024},
  organization={Springer}
}

@INPROCEEDINGS{10647666,
  author={Strohmayer, Julian and Kampel, Martin},
  booktitle={2024 IEEE International Conference on Image Processing (ICIP)}, 
  title={Directional Antenna Systems for Long-Range Through-Wall Human Activity Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={3594-3599},
  doi={10.1109/ICIP51287.2024.10647666}}
```

### Prerequisites
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Datasets
Get the Wallhack1.8k dataset from https://zenodo.org/records/13950918 and put it in the `/data` directory.

### Training & Testing 
Example command for training and testing a dummy ResNet18 model on CSI amplitude features with a window size of 401 WiFi packets (~4 seconds), collected in the Line-of-Sight (LOS) scenario with the Biquad antenna system (BQ):

```
python3 train.py --data /data/wallhack1.8k --bs 128 --ws 351 --scenario LOS --anetnna BQ
```
In this configuration, the samples will have a shape of [128, 1, 52, 401] = [batch size, channels, subcarriers, window size].

### Cross-Domain Generalization
The `--scenario` and `--antenna` arguments allow the selection of the scenario (LOS or NLOS) and the antenna type (BQ or PIFA).

If you want to use a different scenario or antenna type for testing cross-domain generalization, uncomment the following lines in `train.py` and set the `SCENARIO` and `ANTENNA` variables accordingly:

```
# dataloader for cross-domain testing
SCENARIO = 'LOS' # ['LOS', 'NLOS']
ANTENNA = 'BQ' # ['BQ', 'PIFA']
testSubsets = []
for s in sequences:
    testSubsets.append(data.WallhackDataset(f"{opt.data}/{SCENARIO}/{ANTENNA}/{s}.csv", opt=opt))
datasetTest = torch.utils.data.ConcatDataset(testSubsets)
dataloaderTest = torch.utils.data.DataLoader(datasetTest,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
```

For example, to test generalization between LOS and NLOS scenarios on data collected with the BQ system, set:

```
SCENARIO = 'NLOS' # ['LOS', 'NLOS']
ANTENNA = 'BQ' # ['BQ', 'PIFA']
```

Then run:

```
python3 train.py --data /data/wallhack1.8k --bs 128 --ws 351 --scenario LOS --anetnna BQ
```


