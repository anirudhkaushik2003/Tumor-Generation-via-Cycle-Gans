from tqdm import tqdm
import os
import time
from random import randint
 
import gc 
import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm

import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import seaborn as sns
import imageio
import PIL
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output
from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

# !pip install albumentations==0.4.6
import albumentations as A
# from albumentations.pytorch import ToTensor, ToTensorV2

import cv2


from albumentations import Compose, HorizontalFlip
# from albumentations.pytorch import ToTensor, ToTensorV2 

import warnings
warnings.simplefilter("ignore")

IMAGE_SIZE = 256

import time

st = time.time()
# organize as per volumes
PATH= "/ssd_scratch/cvit/anirudhkaushik/datasets/brats2020-training-data/BraTS2020_training_data/content/data/"

print("=============================")
# organize as per volumes
volumes = {}
import glob
for file in glob.glob(PATH+'volume_*'):
    file = file.split('/')[-1]
    vol_ind = file.split('_')[1] 
    if vol_ind not in volumes:
        volumes[vol_ind] = []
    volumes[vol_ind].append(file)

# sort the slices
for vol in volumes:
    volumes[vol] = sorted(volumes[vol], key=lambda x: int(x.split('_')[3].split('.')[0]))

# get slices with tumor (mask2 has white pixels for tumor)
slices = []
for vol in volumes:
    slice = volumes[vol][80]
    slices.append(slice)

    slice = volumes[vol][90]
    slices.append(slice)
    
    slice = volumes[vol][100]
    slices.append(slice)




# randomly select 1000 slices
slices = np.array(slices)
np.random.shuffle(slices)
print("Number of Tumor slices: ", len(slices))
et = time.time()

print(f"Time taken to load data: {(et-st):.2f}s" )

class BRATS(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.slices = slices
        self.path = PATH
        self.image_size = IMAGE_SIZE

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        img = h5py.File(self.path+self.slices[idx], 'r')
        # keep only T1 slice
        f = img['image']
        y = img['mask']
        f = np.array(f)
        f = f[:,:,0] # T1 slice
        y = y[:,:,1]

        f = np.array(f)
        f = f.astype(np.float32)

        y = np.array(y)
        y = y.astype(np.float32)

        if self.transform:
            f = self.transform(f)
            y = self.transform(y)
            return f,y
        
# load npy files
PATH2 = "/ssd_scratch/cvit/anirudhkaushik/datasets/healthy_brain/train/"
healthy_brain = []
for file in glob.glob(PATH2 + "*"):
    healthy_brain.append(file)


healthy_brain = np.array(healthy_brain)
# select 1200 slices randomly
np.random.shuffle(healthy_brain)
healthy_brain = healthy_brain[:len(slices)]

class Healthy(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.slices = healthy_brain
        self.path = PATH2
        self.image_size = IMAGE_SIZE

    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        file = self.slices[idx]
        img = np.load(file)
        img = cv2.normalize(img, None, norm_type=cv2.NORM_MINMAX)
        img = img.reshape(img.shape[2:])
        f = img[:,:, img.shape[2]//2]

        # convert to cv2 img
        f = np.array(f)
        f = f.astype(np.float32)

        if self.transform:
            f = self.transform(f)
            return f

print("Number of healthy slices: ", len(healthy_brain))
print("=============================")
print()
print()
