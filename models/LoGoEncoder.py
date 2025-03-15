import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import numpy as np
import operator
import cv2
import urllib
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
import random

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import List

from copy import deepcopy
from typing import Tuple

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""
    
def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
    
class AxialAttention_gated_data(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=64,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_gated_data, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        

        self.fcn1 = nn.Linear(in_planes, in_planes)
        self.fcn2 = nn.Linear(in_planes, 4)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        
        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)
        
        self.reset_parameters()
        
        # self.print_para()

    def forward(self, x):
        
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # import pdb
        # pdb.set_trace()
        xn = self.pool(x.unsqueeze(3))
        xn = F.relu(self.fcn1(xn.squeeze(2).squeeze(2)))
        xn = F.relu(self.fcn2(xn))

        sig = F.sigmoid(xn) 

        sig1 = sig[:,0]
        sig2 = sig[:,1]
        sig3 = sig[:,2]
        sig4 = sig[:,3]

        # Transformations
        # import pdb
        # pdb.set_trace()
        qkv = self.bn_qkv(self.qkv_transform(x))
        
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        # print(qk.shape, qr.shape, kr.shape)
        # import pdb
        # pdb.set_trace()

        # multiply by factors
        # print(x.shape, qr.shape)
        # import pdb
        # pdb.set_trace()
        qr = sig1.reshape(-1, 1, 1, 1).contiguous()*qr
        kr = sig2.reshape(-1, 1, 1, 1).contiguous()*kr
        # kr = torch.mul(kr, torch.sigmoid(self.f_kr))

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)

        
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = sig3.reshape(-1, 1, 1, 1).contiguous()*sv
        sve = sig4.reshape(-1, 1, 1, 1).contiguous()*sve

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)
        
        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def print_para(self):
        print(self.f_qr)
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
        
class AxialBlock_gated_data(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=256):
        super(AxialBlock_gated_data, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention_gated_data(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention_gated_data(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)  # Changed from True to False
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        #out = self.bn1(out)
        
        out = self.relu(out)

        # 2 layers: hight and width
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        #out = self.bn2(out)
        
        out = self.relu(out)  # Changed from in-place to not in-place

        return out
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.nn.ReLU(inplace=False)(out).unsqueeze(2).unsqueeze(3)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.nn.ReLU(inplace=False)(x)

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        
        return x

class LoGoEncoder(nn.Module):
    def __init__(self):
        super(LoGoEncoder, self).__init__() 
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Sequential(
            # Layer 1: 256x256 -> 128x128
            nn.Conv2d(3, 512, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=False), 
            
            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias = True),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=False), 
        )
        
        self.bn1 = self.norm_layer(512)
        self.conv1_p = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=False), 
            
            # Layer 2: 128x128 -> 64x64
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias = True),
            #nn.BatchNorm2d(512),
            nn.ReLU(inplace=False), 
        )
        self.bn1_p = self.norm_layer(512)
        self.relu = nn.ReLU(inplace=False)
        
        self.layer = AxialBlock_gated_data(
                        inplanes=512,     
                        planes=512,   
                        kernel_size=64
                    )
        self.layer_p = AxialBlock_gated_data(
                        inplanes=512,     
                        planes=512,   
                        kernel_size=16
                    )
        self.adjust_p = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            CBAM(channels=512)
        )
        self.layer_norm = nn.LayerNorm([512, 64, 64])
        self.layer_norm_p = nn.LayerNorm([512, 16, 16])
        self.position_embeddings = nn.Parameter(torch.randn(4, 4, 512))
        
    def forward(self, x):
        img_size = x.shape[-1]  # 256
        
        xin = x.clone()
        x = self.conv1(x)       # 256x256 -> 64x64
        x = self.layer_norm(x)
        x = self.layer(x)
        x_norm = self.layer_norm(x)
        
        #LOCAL FROM THIS
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
                x_p = self.layer_norm_p(x_p)
                x_p = self.layer_p(x_p)  # Shape: [1, 512, 16, 16]
                # Add positional encoding
                pos_embed = self.position_embeddings[i, j]  # Shape: [embed_dim]
                pos_embed = pos_embed.view(1, -1, 1, 1)  # Reshape to [1, embed_dim, 1, 1] for broadcasting
                x_p_pos = x_p +  pos_embed 
                
                # Place processed patch in correct location
                x_loc[:, :,
                      output_size*i:output_size*(i+1),
                      output_size*j:output_size*(j+1)] = x_p_pos 
        #COMBINE LOCAL AND GLOBAL

        x_loc_norm = self.layer_norm(x_loc)
        
        x_combine = torch.add(x_norm, x_loc_norm)  ## Shape: [1, 512, 64, 64]
        x_combine = self.adjust_p(x_combine) #CBAM
        # x_combine = self.layer_norm(x_combine)
        out = self.relu(x_combine)
        
        return out