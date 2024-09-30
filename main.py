import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils, transforms
from PIL import Image
import os
import numpy as np
from siggraph17 import SIGGRAPHGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colorizer_siggraph17 = siggraph17(pretrained=True).eval()
colorizer_siggraph17.to(device)

optimizer = torch.optim.Adam(colorizer_siggraph17.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
