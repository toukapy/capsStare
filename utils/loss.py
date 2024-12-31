import torch.nn.functional as F
import torch
import torch.nn as nn

def contrastive_loss(z_i, z_j, temperature = 0.5):
    z_i = F.normalize(z_i, dim = 1)
    z_j = F.normalize(z_j, dim = 1)

    similarity_matrix = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(z_i.size[0]).to(z_i.device)
    return nn.CrossEntropyLoss()(similarity_matrix, labels)

def reconstruction_loss(original, reconstructed):
    return nn.MSELoss()(original, reconstructed)