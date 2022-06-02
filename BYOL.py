#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import resnets
import torch.nn.functional as F
import torchvision
    
import copy
import torch
import torch.nn.functional as F

from geoopt.manifolds.stereographic import math as gmath


# In[3]:


from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ResNet18(cifar = True, classification = False):
    model = torchvision.models.resnet18()
    if not classification:
        model.fc = nn.Identity()
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    return model

def ResNet34(cifar = True):
    model = torchvision.models.resnet34()
    model.fc = nn.Identity()
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    return model


# In[4]:


def euclidean_byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True):
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.
    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.
    Returns:
        torch.Tensor: BYOL's loss.
    """
    return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, keepdim=False, dim=-1):
    """
    Find the geodisic between x and y for the given curvature
    """
    return gmath.dist(x, y, k = k, keepdim=False, dim=-1)

def update_target_params(online_params, target_params, tau):

    #update the backbone first
    for op, mp in zip(online_params, target_params):
        mp.data = tau * mp.data + (1 - tau) * op.data


# In[27]:



class euclidean_BYOL_module(nn.Module):
    
    def __init__(self, input_size = (3, 32, 32), classes = 100, projection_size = 256, projection_hidden_size = 4096, coarse_classes = 20):
        super().__init__()
        self.network = ResNet18()
        example = torch.ones(input_size)
        self.feature_dim = (self.network(example.unsqueeze(0))).shape[1]
        self.classifier = nn.Linear(self.feature_dim, classes)
        self.coarse_classifier = nn.Linear(self.feature_dim, coarse_classes)
        
        self.h_classifier = None
        
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
        coarse_logits = self.coarse_classifier(representation.detach())
        out = {'Representation': representation,
              "logits": logits,
              "coarse_logits": coarse_logits,
              "z": z}    
        return out
    
    def forward(self, x):
        representation = self.network(x)
        logits = self.classifier(representation.detach())
        coarse_logits = self.coarse_classifier(representation.detach())
        z = self.projector(representation)
        p = self.predictor(z)
        
        out = {'Representation': representation,
              "logits": logits,
              "coarse_logits": coarse_logits,
              "z": z,
              "p": p} 
        return out


# In[28]:


import mobius

class hyperbolic_BYOL_module(nn.Module):
    
    def __init__(self, input_size = (3, 32, 32), classes = 100, projection_size = 256, projection_hidden_size = 4096, coarse_classes = 20):
        super().__init__()
        self.network = ResNet18()
        example = torch.ones(input_size)
        self.feature_dim = (self.network(example.unsqueeze(0))).shape[1]
        self.classifier = nn.Linear(self.feature_dim, classes)
        self.coarse_classifier = nn.Linear(self.feature_dim, coarse_classes)
        self.h_representation = mobius.MobiusLinear(self.feature_dim, classes, hyperbolic_input = False, fp64_hyper = False)
        self.h_representation_coarse = mobius.MobiusLinear(self.feature_dim, coarse_classes, hyperbolic_input = False, fp64_hyper = False)
        
        self.h_classifier = mobius.MobiusDist2Hyperplane(classes, classes)
        self.h_coarse_classifier = mobius.MobiusDist2Hyperplane(coarse_classes, coarse_classes)
        
        '''self.projector = nn.Sequential(
            mobius.MobiusLinear(self.feature_dim, projection_hidden_size, hyperbolic_input = False, nonlin = nn.ReLU(), fp64_hyper = False),
            mobius.MobiusLinear(projection_hidden_size, projection_size, nonlin = nn.ReLU(), fp64_hyper = False))
            
            self.predictor = nn.Sequential(
            mobius.MobiusLinear(projection_size, projection_hidden_size, nonlin = nn.ReLU(), fp64_hyper = False),
            mobius.MobiusLinear(projection_hidden_size, projection_size, nonlin = nn.ReLU(), fp64_hyper = False))'''
        
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size))
        
        self.hyperbolic_projector = mobius.MobiusLinear(projection_size, projection_size, hyperbolic_input = False, fp64_hyper = False)
        
        self.predictor = nn.Sequential(
            nn.Linear(projection_size, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_size))
        
        
        
        

    def momentum_forward(self, x):
        with torch.no_grad():
            representation = self.network(x)
            h_representation = gmath.expmap0(representation, k=torch.Tensor([-1]).to(device))
            z = self.projector(representation)
            z = self.hyperbolic_projector(z)
        logits = self.classifier(representation.detach())
        coarse_logits = self.coarse_classifier(representation.detach())
        
        h_logits = self.h_classifier(h_representation.detach())
        h_coarse_logits = self.h_coarse_classifier(h_representation.detach())
        
        
        out = {'Representation': representation,
              "logits": logits,
              "coarse_logits": coarse_logits,
              "h_logits": h_logits,
              "h_coarse_logits": h_coarse_logits,
              "z": z}    
        return out
    
    def forward(self, x):
        representation = self.network(x)
        h_representation = self.h_representation(representation)
        h_coarse_representation = self.h_representation_coarse(representation)
        
        h_logits = self.h_classifier(h_representation.detach())
        h_coarse_logits = self.h_coarse_classifier(h_coarse_representation.detach())
        
        logits = self.classifier(representation.detach())
        coarse_logits = self.coarse_classifier(representation.detach())
        z = self.projector(representation)
        p = self.predictor(z)
        #z = self.hyperbolic_projector(z)
        
        
        out = {'Representation': representation,
              "logits": logits,
              "coarse_logits": coarse_logits,
              "h_logits": h_logits,
              "h_coarse_logits": h_coarse_logits,
              "z": z,
              "p": p} 
        return out
    
    


# In[ ]:




