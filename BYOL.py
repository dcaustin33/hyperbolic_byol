#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import ResNets
import torch.nn.functional as F
import torchvision
    
import copy
import torch
import torch.nn.functional as F


# In[6]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ResNet18(cifar = True):
    model = torchvision.models.resnet18()
    model.fc = nn.Identity()
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    return model



def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True):
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.
    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.
    Returns:
        torch.Tensor: BYOL's loss.
    """
    return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()


def update_target_params(online_params, target_params, tau):

    #update the backbone first
    for op, mp in zip(online_params, target_params):
        mp.data = tau * mp.data + (1 - tau) * op.data


# In[ ]:


class BYOL_module(nn.Module):
    
    def __init__(self, input_size = (3, 32, 32), classes = 100, projection_size = 256, projection_hidden_size = 4096):
        super().__init__()
        self.network = ResNet18()
        example = torch.ones(input_size)
        self.feature_dim = (self.network(example.unsqueeze(0))).shape[1]
        self.classifier = nn.Linear(self.feature_dim, classes)
        
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size))
        
        self.predictor = nn.Sequential(
            nn.Linear(projection_size, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size))
        
    def momentum_forward(self, x):
        with torch.no_grad():
            representation = self.network(x)
            z = self.projector(representation)
        logits = self.classifier(representation.detach())
        out = {'Representation': representation,
              "logits": logits,
              "z": z}    
        return out
    
    def forward(self, x):
        representation = self.network(x)
        logits = self.classifier(representation.detach())
        z = self.projector(representation)
        p = self.predictor(z)
        
        out = {'Representation': representation,
              "logits": logits,
              "z": z,
              "p": p} 
        return out