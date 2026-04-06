import os
import json
import torch
import data_utils
import clip
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import utils
import math

class CBM_model(torch.nn.Module):
    def __init__(self, backbone_name, weights, proj_type='mlp', device="cuda"):
        super().__init__()
        
        backbone_name = backbone_name[5:]
        backbone_name = backbone_name.rsplit("-", 1)
        backbone_name = "/".join(backbone_name)
        model, _ = clip.load(backbone_name, device=device)
        model.eval()

        self.img = lambda x: model.encode_image(x).float()
        self.txt = lambda x: model.encode_text(x).float()
            
        if proj_type == 'linear':
            self.proj = ProjModel_2CBL(input_size1 = weights['fc1.weight'].shape[1], input_size2 = weights['fc2.weight'].shape[1],
                    output_size = weights['fc1.weight'].shape[0], proj_type=proj_type, device = device)
        elif proj_type == 'mlp':
            self.proj = ProjModel_2CBL(input_size1 = weights['fc1.projection.weight'].shape[1], input_size2 = weights['fc2.projection.weight'].shape[1],
                    output_size = weights['fc1.projection.weight'].shape[0], proj_type=proj_type, device = device)
        else:
            raise NotImplementedError

        self.proj.load_state_dict(weights)
        
        self.concepts = None
        
    def forward(self, image, text):
        img = self.img(image)
        img /= torch.norm(img, dim=-1, keepdim=True)
        txt = self.txt(text)
        txt /= torch.norm(txt, dim=-1, keepdim=True)
        img, txt, temp = self.proj(img, txt)
        
        img = torch.clamp(img, min=0)
        I_e = F.normalize(img, p=2, dim=-1)

        txt = torch.clamp(txt, min=0)
        T_e = F.normalize(txt, p=2, dim=-1)

        logits = (I_e @ T_e.T) * torch.exp(temp)
        pred = torch.softmax(logits, dim=1)
        return pred, img, txt, temp
    
    def encode_image(self, image):
        img = self.img(image)
        img /= torch.norm(img, dim=-1, keepdim=True)
        img = self.proj.encode_image(img)
        return img
    
    def encode_text(self,text):
        txt = self.txt(text)
        txt /= torch.norm(txt, dim=-1, keepdim=True)
        txt = self.proj.encode_text(txt)
        return txt
    
    def temp(self):
        return self.proj.temperature

class CBM_model_NEC(torch.nn.Module):
    def __init__(self, backbone_name, weights, proj_type='mlp', nec=5, device="cuda"):
        super().__init__()
        
        backbone_name = backbone_name[5:]
        backbone_name = backbone_name.rsplit("-", 1)
        backbone_name = "/".join(backbone_name)
        model, _ = clip.load(backbone_name, device=device)
        model.eval()

        self.img = lambda x: model.encode_image(x).float()
        self.txt = lambda x: model.encode_text(x).float()
            
        if proj_type == 'linear':
            self.proj = ProjModel_2CBL(input_size1 = weights['fc1.weight'].shape[1], input_size2 = weights['fc2.weight'].shape[1],
                    output_size = weights['fc1.weight'].shape[0], proj_type=proj_type, device = device)
        elif proj_type == 'mlp':
            self.proj = ProjModel_2CBL(input_size1 = weights['fc1.projection.weight'].shape[1], input_size2 = weights['fc2.projection.weight'].shape[1],
                    output_size = weights['fc1.projection.weight'].shape[0], proj_type=proj_type, device = device)
        else:
            raise NotImplementedError
        self.proj.load_state_dict(weights)
        
        self.concepts = None
        self.nec = nec
        
    def forward(self, image, text):
        img = self.img(image)
        img /= torch.norm(img, dim=-1, keepdim=True)
        txt = self.txt(text)
        txt /= torch.norm(txt, dim=-1, keepdim=True)
        img, txt ,temp = self.proj(img, txt)

        img = torch.clamp(img, min=0)
        I_e = F.normalize(img, p=2, dim=-1)

        txt = torch.clamp(txt, min=0)
        T_e = F.normalize(txt, p=2, dim=-1)
        
        logits = torch.sum(torch.topk((I_e.unsqueeze(1) * T_e), self.nec)[0], dim=2) * torch.exp(temp)
        pred = torch.softmax(logits, dim=1)
        return pred, img, txt, temp
    
    def encode_image(self, image):
        img = self.img(image)
        img /= torch.norm(img, dim=-1, keepdim=True)
        img = self.proj.encode_image(img)
        return img
    
    def encode_text(self,text):
        txt = self.txt(text)
        txt /= torch.norm(txt, dim=-1, keepdim=True)
        txt = self.proj.encode_text(txt)
        return txt
    
    def temp(self):
        return self.proj.temperature
    
class ProjModel_2CBL(nn.Module):
    def __init__(self, input_size1, input_size2, output_size, temperature=0.07, dropout = 0.1, proj_type = 'mlp', device = 'cuda'):
        super().__init__()
        if proj_type == 'linear':
            # for Image
            self.fc1 = torch.nn.Linear(in_features=input_size1, out_features=output_size, bias=False).to(device)
            # for Text
            self.fc2 = torch.nn.Linear(in_features=input_size2, out_features=output_size, bias=False).to(device)
        elif proj_type == 'mlp':
            # for Image
            self.fc1 = MLP(input_dim=input_size1, concept_dim=output_size, dropout = dropout).to(device)
            # for Text
            self.fc2 = MLP(input_dim=input_size2, concept_dim=output_size, dropout = dropout).to(device)
        else:
            raise NotImplementedError

        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x1, x2):
        out1 = self.fc1(x1)
        out2 = self.fc2(x2)
        return out1, out2, self.temperature
    
    def encode_image(self, x1):
        out1 = self.fc1(x1)
        return out1
    
    def encode_text(self, x2):
        out2 = self.fc2(x2)
        return out2
        
class MLP(nn.Module):
    def __init__(self, input_dim, concept_dim, dropout):
        super().__init__()
        # hidden_dim = int(math.sqrt(input_dim * concept_dim))
        self.projection = nn.Linear(input_dim, concept_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(concept_dim, concept_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x


def loss_fn(outs_image, outs_text, img_target_features, txt_target_features, temp, weight, targets, nec=5, metric = 'similarity', chunk_size=32):
    if metric == 'similarity':
        lossfun = utils.cos_sim_cubed
        loss_image = F.binary_cross_entropy(torch.sigmoid(outs_image), img_target_features)
        # loss_image = - torch.mean(lossfun(outs_image, img_target_features))
        loss_text = - torch.mean(lossfun(outs_text, txt_target_features))
        # loss_text = F.binary_cross_entropy(torch.sigmoid(outs_text), txt_target_features)
    else:
        lossfun = nn.MSELoss()
        loss_image = lossfun(outs_image, img_target_features)
        loss_text = lossfun(outs_text, txt_target_features)
    
    outs_image = torch.clamp(outs_image, min=1e-6)
    outs_text = torch.clamp(outs_text, min=1e-6)
    I_e = F.normalize(outs_image, p=2, dim=-1)
    T_e = F.normalize(outs_text, p=2, dim=-1)

    I, C = I_e.shape
    T, _ = T_e.shape
    logits = torch.zeros(I, T, device=I_e.device)
    scaler = temp.exp()

    for i_start in range(0, I, chunk_size):

        i_end = min(i_start + chunk_size, I)
        I_chunk = I_e[i_start:i_end]  # [chunk, C]

        prod = I_chunk.unsqueeze(1) * T_e
        
        topk, _ = torch.topk(prod, nec, dim=2)
        logits[i_start:i_end] = torch.sum(topk, dim=2) * scaler

    loss_cls = F.cross_entropy(logits, targets)
    loss = (1-weight) * (loss_image + loss_text) + weight * loss_cls

    return loss

def loss_fn_woa(outs_image, outs_text, img_target_features, txt_target_features, temp, weight, targets, nec=5, metric = 'similarity', prior = None):
    if metric == 'similarity':
        lossfun = utils.cos_sim_cubed
        loss_image = F.binary_cross_entropy(torch.sigmoid(outs_image), img_target_features)
        # loss_image = - torch.mean(lossfun(outs_image, img_target_features))
        if prior is not None:
            loss_text = - torch.mean(lossfun(outs_text, txt_target_features) + lossfun(outs_text, prior))
        else:
            loss_text = - torch.mean(lossfun(outs_text, txt_target_features))
    else:
        lossfun = nn.MSELoss()
        loss_image = lossfun(outs_image, img_target_features)
        loss_text = lossfun(outs_text, txt_target_features)
    
    I_e = F.normalize(outs_image, p=2, dim=-1)
    T_e = F.normalize(outs_text, p=2, dim=-1)
    logits = torch.sum(torch.topk((I_e.unsqueeze(1) * T_e), nec)[0], dim=2) * torch.exp(temp)
    # logits = (I_e @ T_e.T) * torch.exp(temp)
    loss_cls = F.cross_entropy(logits, targets)
    loss = (1-weight) * (loss_image + loss_text) + weight * loss_cls

    return loss

def loss_fn_kd(outs_image, outs_text, img_target_features, txt_target_features, temp, weight, targets_KD, nec=5, targets = None, metric = 'similarity', prior = None):
    if metric == 'similarity':
        lossfun = utils.cos_sim_cubed
        # lossfun2 = similarity.cos_similarity
        loss_image = F.binary_cross_entropy(torch.sigmoid(outs_image), img_target_features)
        # loss_image = - torch.mean(lossfun2(outs_image.T, img_target_features.T))
        # loss_image = F.mse_loss(outs_image, img_target_features)
        loss_text = - torch.mean(lossfun(outs_text, txt_target_features))
    else:
        lossfun = nn.MSELoss()
        loss_image = lossfun(outs_image, img_target_features)
        loss_text = lossfun(outs_text, txt_target_features)
    
    outs_image = torch.clamp(outs_image, min=0)
    I_e = F.normalize(outs_image, p=2, dim=-1)

    outs_text = torch.clamp(outs_text, min=0)
    T_e = F.normalize(outs_text, p=2, dim=-1)

    logits = torch.sum(torch.topk((I_e.unsqueeze(1) * T_e), nec)[0], dim=2) * torch.exp(temp)

    loss_kd = 0.5 * (F.kl_div(torch.softmax(logits, dim=1).log(), torch.softmax(targets_KD, dim=1), reduction='batchmean') +
                      F.kl_div(torch.softmax(logits.T, dim=1).log(), torch.softmax(targets_KD.T, dim=1), reduction='batchmean'))
    
    loss_cls = F.cross_entropy(logits, torch.argmax(torch.softmax(targets_KD, dim=1), dim=1))

    loss = (1-weight) * (loss_image + loss_text) + weight * (0.8 * loss_kd + 0.2 * loss_cls)

    return loss

def loss_fn_cont(outs_image, outs_text, img_target_features, txt_target_features, temp, weight, nec=5, metric = 'similarity', chunk_size=16):
    if metric == 'similarity':
        lossfun = utils.cos_sim_cubed
        loss_image = F.binary_cross_entropy(torch.sigmoid(outs_image), img_target_features)
        # loss_image = - torch.mean(lossfun(outs_image, img_target_features))
        loss_text = - torch.mean(lossfun(outs_text, txt_target_features))
    else:
        lossfun = nn.MSELoss()
        loss_image = lossfun(outs_image, img_target_features)
        loss_text = lossfun(outs_text, txt_target_features)
    
    outs_image = torch.clamp(outs_image, min=1e-6)
    outs_text = torch.clamp(outs_text, min=1e-6)
    I_e = F.normalize(outs_image, p=2, dim=-1)
    T_e = F.normalize(outs_text, p=2, dim=-1)

    I, C = I_e.shape
    T, _ = T_e.shape
    logits = torch.zeros(I, T, device=I_e.device)
    scaler = temp.exp()

    for i_start in range(0, I, chunk_size):

        i_end = min(i_start + chunk_size, I)
        I_chunk = I_e[i_start:i_end]  # [chunk, C]

        prod = I_chunk.unsqueeze(1) * T_e
        
        topk, _ = torch.topk(prod, nec, dim=2)
        logits[i_start:i_end] = torch.sum(topk, dim=2) * scaler

    targets = torch.arange(len(outs_image)).cuda()

    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    loss_cls = (loss_i + loss_t) / 2
    loss = (1-weight) * (loss_image + loss_text) + weight * loss_cls

    return loss

def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    accuracies = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        accuracies.append(acc.item())
    return accuracies
