import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import operator
import cv2
import urllib
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
import random
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import List, Tuple
from axial_attention import AxialAttention, AxialPositionalEmbedding
from copy import deepcopy

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



class LoGoEncoder(nn.Module):
    def __init__(self):
        super(LoGoEncoder, self).__init__() 
        
        
        self.conv1 = nn.Sequential(
            # Layer 1: 256x256 -> 128x128
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        
        self.conv1_p = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        

        
        self.global_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([512, 64, 64]),
            AxialPositionalEmbedding(dim = 512, shape = (64, 64)),
            AxialAttention(dim = 512, heads = 8, dim_index = 1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([512, 64, 64]),
        )
        self.local_block = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([512, 16, 16]),
            AxialPositionalEmbedding(dim = 512, shape = (64, 64)),
            AxialAttention(dim = 512, heads = 8, dim_index = 1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([512, 16, 16]),
        )
        
        self.adjust_p = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
        

    def forward(self, x):
        img_size = x.shape[-1]  # 256
        
        xin = x.clone()
        x = self.conv1(x)       # 256x256 -> 64x64

        x_attn = self.global_block(x)
        x = x + x_attn
        
        
        x_loc = x.clone()       # Shape: [1, 512, 64, 64]
        patch_size = img_size // 4  # 64
        output_size = x_loc.shape[-1] // 4  # 16
        
        for i in range(4):
            for j in range(4):
                # Extract patch
                x_p = xin[:, :, 
                         patch_size*i:patch_size*(i+1),
                         patch_size*j:patch_size*(j+1)]
                
                # Process patch
                x_p = self.conv1_p(x_p)

                x_p_attn = self.local_block(x_p)  # Shape: [1, 512, 16, 16]
                
                x_p = x_p + x_p_attn

                # Place processed patch in correct location
                x_loc[:, :,
                      output_size*i:output_size*(i+1),
                      output_size*j:output_size*(j+1)] = x_p
                
                
        weights = F.softmax(self.weights, dim=0)
        x_combine = weights[0] * x + weights[1] * x_loc
        x_combine = self.adjust_p(x_combine) 
        
        
        return x_combine