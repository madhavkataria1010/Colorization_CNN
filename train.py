import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import siggraph17
from dataloader import data_loader, ImageNetColorizationDataset

from siggraph17 import SIGGRAPHGenerator
from tqdm import tqdm
from image_transform import load_img, resize_img, preprocess_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs =  100
model = SIGGRAPHGenerator().to(device)

def train(epochs, model, data_loader):
    model.train()
    for i in epochs:
        for i, data in enumerate(data_loader):
            print(f" i : {i}   data : {data}")

train(epochs, model, data_loader)