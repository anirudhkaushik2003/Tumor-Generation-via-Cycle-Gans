import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import PIL
import numpy as np
import torchvision
import glob
import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from generator import Generator
from discriminator import Discriminator, UNet
from datasets import *
import random
import itertools

from utils import *
from logger import wandb_init, log_images

BATCH_SIZE = 1
IMAGE_SIZE = 256
IMG_CHANNELS = 1

multiGPU = False
epochs = 100

lambda_1 = 5
lambda_2 = 10


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# torch.use_deterministic_algorithms(True) # Needed for reproducible results


# init dataset
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

data_transforms2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(200),
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

dataset = Healthy(data_transforms)
dataset2 = BRATS(data_transforms2)

min_length = min(len(dataset), len(dataset2))

#make both datasets of equal length
dataset = torch.utils.data.Subset(dataset, np.arange(min_length))
dataset2 = torch.utils.data.Subset(dataset2, np.arange(min_length))

# sampler
sampler = torch.utils.data.RandomSampler(dataset, num_samples=min_length, replacement=True)
sampler2 = torch.utils.data.RandomSampler(dataset2, num_samples=min_length, replacement=True)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(sampler is None), sampler=sampler )
dataloader2 = DataLoader(dataset2, batch_size=BATCH_SIZE, shuffle=(sampler2 is None), sampler=sampler2  )



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize models
modelG_1 = Generator(IMAGE_SIZE, IMG_CHANNELS)
modelD_1 = UNet(IMG_CHANNELS, 1)
modelG_1.apply(weights_init)
# modelD_1.apply(weights_init)
modelG_2 = Generator(256, IMG_CHANNELS)
modelD_2 = UNet(IMG_CHANNELS, 1)
modelG_2.apply(weights_init)
# modelD_2.apply(weights_init)

dummy_img = np.ones((BATCH_SIZE, IMG_CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
patch_shape = modelD_1(torch.FloatTensor(dummy_img)).shape[-1]

if multiGPU:
    modelG_1 = torch.nn.DataParallel(modelG_1)
    modelD_1 = torch.nn.DataParallel(modelD_1)
    modelG_2 = torch.nn.DataParallel(modelG_2)
    modelD_2 = torch.nn.DataParallel(modelD_2)

modelG_1 = modelG_1.to(device)
modelG_2 = modelG_2.to(device)
modelD_1 = modelD_1.to(device)
modelD_2 = modelD_2.to(device)

# optimizers
learning_rateG = 2e-4
learning_rateD = 2e-4

optimizerD = torch.optim.Adam(itertools.chain(modelD_1.parameters(), modelD_2.parameters()), lr=learning_rateD, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(itertools.chain(modelG_1.parameters(), modelG_2.parameters()), lr=learning_rateG, betas=(0.5, 0.999))


# scheduler to linearly decay learning rate to 0
schedulerG = torch.optim.lr_scheduler.LinearLR(optimizerG, start_factor=1, end_factor=0.0, total_iters=epochs, last_epoch=-1)
schedulerD = torch.optim.lr_scheduler.LinearLR(optimizerD, start_factor=1, end_factor=0.0, total_iters=epochs, last_epoch=-1)

# loss functions
criterionGAN = nn.MSELoss() # adversarial loss
criterionID = nn.L1Loss() # identity loss
criterionCycle = nn.L1Loss() # cycle loss (forward)

patch_shape = IMAGE_SIZE
architechture = "UNET Discriminator"
### PRINT STATS ###
print("***********************")
print(f"Architechture: {architechture}")
print("Discriminator Patch shape: ", patch_shape)
print("Number of samples per epoch: ", len(dataset))
print("Multi GPU: ", multiGPU)
print("Learning rate: ", learning_rate)
print("Batch size: ", BATCH_SIZE)
print("Image size: ", IMAGE_SIZE)
print("Number of epochs: ", epochs)
print("Lambda 1: ", lambda_1)
print("Lambda 2: ", lambda_2)
print("Device: ", device)
print("***********************")
print("\n\n\n\n\n\n")
################# TRAINING #####################
    
n_epoch, n_batch = epochs, BATCH_SIZE

pool_tumor, pool_healthy = list(), list()
batch_per_epoch = int(len(dataset) / BATCH_SIZE)

n_steps = min_length


def train_composite_model(modelD_Z, modelD_H, fake_tumor, fake_healthy, rec_tumor, rec_healthy, tumor, healthy, ones, id=True):
    # adversarial loss
    pred_fake_tumor = modelD_Z(fake_tumor)
    loss_GAN_tumor = criterionGAN(pred_fake_tumor, ones)

    pred_fake_healthy = modelD_H(fake_healthy)
    loss_GAN_healthy = criterionGAN(pred_fake_healthy, ones)

    # forward cycle loss
    loss_forward_cycle = criterionCycle(rec_tumor, tumor) * lambda_2
    
    # backward cycle loss
    loss_backward_cycle = criterionCycle(rec_healthy, healthy) * lambda_2

    if id:
        loss_id_tumor = criterionID(tumor, fake_tumor) * lambda_1
        loss_id_healthy = criterionID(healthy, fake_healthy) * lambda_1

    else: 
        loss_id_tumor = 0
        loss_id_healthy = 0

    lossG = loss_GAN_tumor + loss_GAN_healthy + loss_forward_cycle + loss_backward_cycle + loss_id_tumor + loss_id_healthy
    lossG.backward()
    return lossG


def train_D(D, real, fake, ones, zeros):
    # real batch
    pred_real = D(real)
    loss_D_real = criterionGAN(pred_real, ones)
    # fake batch
    pred_fake = D(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, zeros)

    loss_D = (loss_D_real + loss_D_fake)*0.5
    loss_D.backward()

    return loss_D, pred_fake

wandb_init(learning_rateG, learning_rateD, epochs*2, BATCH_SIZE, patch_shape, architechture, "BHB->BRATS", multiGPU)

for epoch in range(epochs*2):
    lossG_list = []
    lossD_list = []

    D_tumor = 0
    D_healthy = 0

    # randomly choose samples from dataloader


    for step, (healthy, (tumor, tumor_label)) in enumerate(zip(dataloader, dataloader2)):

        # ones = torch.ones((int(healthy.shape[0]), 1, patch_shape, patch_shape)).to(device)
        ones = tumor_label.to(device)
        zeros = torch.zeros((int(healthy.shape[0]), 1, patch_shape, patch_shape)).to(device)
        
        healthy = healthy.to(device)
        tumor = tumor.to(device)

        fake_tumor = modelG_1(healthy)
        fake_healthy = modelG_2(tumor)
        rec_tumor = modelG_1(fake_healthy)
        rec_healthy = modelG_2(fake_tumor)

        #  Ds require no gradients when optimizing Gs
        set_model_grad(modelD_1, False)  
        set_model_grad(modelD_2, False)

        # Train Gs
        optimizerG.zero_grad()

        loss_G = train_composite_model(modelD_1, modelD_2, fake_tumor, fake_healthy, rec_tumor, rec_healthy, tumor, healthy, ones)
        # add to lost list
        lossG_list.append(loss_G.item())
        optimizerG.step()


        # set Ds to require gradients
        set_model_grad(modelD_1, True)
        set_model_grad(modelD_2, True)

        # Train Ds
        optimizerD.zero_grad()

        fake_tumor_pool = torch.FloatTensor(update_image_pool(pool_tumor, fake_tumor)).to(device)
        fake_healthy_pool = torch.FloatTensor(update_image_pool(pool_healthy, fake_healthy)).to(device)
        # tumor D
        loss_D_1, pred_tumor = train_D(modelD_1, tumor, fake_tumor_pool, ones, zeros)
        loss_D_2, pred_healthy = train_D(modelD_2, healthy, fake_healthy_pool, ones, zeros)

        loss_D = (loss_D_1 + loss_D_2)*0.5
        lossD_list.append(loss_D.item())

        D_tumor += pred_tumor.mean().item()
        D_healthy += pred_healthy.mean().item()

        optimizerD.step()

        if (step+1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step+1}/{n_steps}], lossD: {np.mean(lossD_list):.2f}, lossG: {np.mean(lossG_list):.2f}")
            print(f"pred_tumor: {(D_tumor/(step+1)):.2f}, pred_real: {(D_healthy/(step+1)):.2f}")
            print()


    create_checkpoint(modelG_1, epoch, multiGPU, "G1")
    create_checkpoint(modelG_2, epoch, multiGPU, "G2")
    create_checkpoint(modelD_1, epoch, multiGPU, "D1")
    create_checkpoint(modelD_2, epoch, multiGPU, "D2")

    log_images(healthy[0].cpu().detach().numpy().transpose((1,2,0)), fake_healthy[0].cpu().detach().numpy().transpose((1,2,0)), epoch, real=True)

    if epoch > epochs:
        schedulerD.step()
        schedulerG.step()

        learning_rateG = schedulerG.get_last_lr()[0]
        learning_rateD = schedulerD.get_last_lr()[0]

        print(f"Resetting Discriminator Learning Rate to {learning_rateD}") 
        print(f"Resetting Generator Learning Rate to {learning_rateG}")
        print("***********************", end="\n\n\n\n")
        


