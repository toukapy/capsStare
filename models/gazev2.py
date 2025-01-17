import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class FrozenEncoder(nn.Module):
    """ Frozen backbone fro feature extraction."""
    def __init__(self):
        super(FrozenEncoder, self).__init__()
        base_model = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.features(x)
        return features

class CapsuleFormation(nn.Module):
    """Transforms features into capsules."""
    def __init__(self, input_dim, num_capsules, capsule_dim):
        super(CapsuleFormation, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.linear = nn.Linear(input_dim, num_capsules * capsule_dim)

    def forward(self, features):
        B, C, H, W = features.size()
        flattened = features.view(B, C, -1).mean(dim=2)  # Global pooling
        capsules = self.linear(flattened)  # Shape: (B, num_capsules * capsule_dim)
        capsules = capsules.view(B, self.num_capsules, self.capsule_dim)  # Reshape
        return capsules
    

class SelfAttentionRouting(nn.Module):
    """Implements SAR for routing between capsules."""
    def __init__(self, num_capsules, capsule_dim, heads=4):
        super(SelfAttentionRouting, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=capsule_dim, num_heads=heads)

    def forward(self, capsules):
        capsules = capsules.transpose(0, 1)  # Required shape: (seq_len, batch, feature_dim)
        routed_capsules, _ = self.multihead_attn(capsules, capsules, capsules)
        return routed_capsules.transpose(0, 1)  # Shape: (B, num_capsules, capsule_dim)


class RegionDecoder(nn.Module):
    """Region-specific GRU decoder."""
    def __init__(self, capsule_dim, hidden_dim, output_dim):
        super(RegionDecoder, self).__init__()
        self.gru = nn.GRU(capsule_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, capsules):
        _, hidden = self.gru(capsules)  # Only take hidden state
        output = self.fc(hidden.squeeze(0))  # Shape: (B, output_dim)
        return output
    

class GazeFusion(nn.Module):
    """Fuses region-specific outputs."""
    def __init__(self, input_dim, output_dim):
        super(GazeFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, regions):
        fused = torch.cat(regions, dim=1)  # Concatenate region outputs
        return self.fc(fused)


class GazeEstimationModel(nn.Module):
    def __init__(self, encoder, capsule_dim=64, hidden_dim=128, output_dim=3):
        super(GazeEstimationModel, self).__init__()
        self.encoder = encoder
        self.capsule_formation = CapsuleFormation(input_dim=2048, num_capsules=8, capsule_dim=capsule_dim)
        self.routing = SelfAttentionRouting(num_capsules=8, capsule_dim=capsule_dim)

        # Region decoders
        self.eye_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)
        self.face_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)

        # Fusion module
        self.fusion = GazeFusion(output_dim * 2, output_dim)

    def forward(self, x):
        B, T, C, H, W = x.size()  # Batch size, Temporal length, Channels, Height, Width
        x = x.view(B * T, C, H, W)  # Flatten temporal dimension for encoder
        features = self.encoder(x)  # Process with frozen encoder
        features = features.view(B, T, -1)  # Reshape back to temporal sequences

        # Pass through capsule layers and GRU
        capsules = self.capsule_formation(features)
        routed_capsules = self.routing(capsules)
        output = self.eye_decoder(routed_capsules)
        return output

