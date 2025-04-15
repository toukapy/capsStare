import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class FrozenEncoder(nn.Module):
    """Frozen backbone for feature extraction using ConvNeXt-Base."""

    def __init__(self, trainable_layers=10):
        super(FrozenEncoder, self).__init__()
        # Load the ConvNeXt-Base model with pretrained weights.
        base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # Use the features from the ConvNeXt model.
        # In torchvision's implementation, `base_model.features` contains all the convolutional blocks.
        self.features = base_model.features

        # Freeze all parameters in the feature extractor.
        for param in self.features.parameters():
            param.requires_grad = False

        # Unfreeze the last few parameters (or blocks) as specified by trainable_layers.
        # Note: This simple approach unfreezes the last 'trainable_layers' parameters; depending on your needs,
        # you might want to unfreeze whole blocks instead.
        for param in list(self.features.parameters())[-trainable_layers:]:
            param.requires_grad = True

        # The output channels of convnext_base are 1024.
        self.norm = nn.BatchNorm2d(1024)

    def forward(self, x):
        features = self.features(x)
        features = self.norm(features)
        return features


class CapsuleFormation(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim):
        """
        Produces capsule maps while keeping spatial dimensions.
        Output shape: (B, T, num_capsules, capsule_dim, H_enc, W_enc)
        """
        super(CapsuleFormation, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size=1)
        self.dropout = nn.Dropout(p=0.3)
        self.norm = nn.BatchNorm2d(num_capsules * capsule_dim)
        self.activation = nn.GELU()

    def forward(self, features):
        # features: (B, T, C, H, W) or (B, C, H, W)
        if features.dim() == 5:
            B, T, C, H, W = features.size()
            features = features.view(B * T, C, H, W)
            out = self.conv(features)  # (B*T, num_capsules*capsule_dim, H, W)
            out = self.dropout(out)
            out = self.norm(out)
            out = self.activation(out)
            # Reshape to (B, T, num_capsules, capsule_dim, H, W)
            out = out.view(B, T, self.num_capsules, self.capsule_dim, H, W)
        elif features.dim() == 4:
            B, C, H, W = features.size()
            out = self.conv(features)
            out = self.dropout(out)
            out = self.norm(out)
            out = self.activation(out)
            out = out.view(B, self.num_capsules, self.capsule_dim, H, W)
        else:
            raise ValueError("Unexpected input shape: {}".format(features.shape))
        return out

class SelfAttentionRouting(nn.Module):
    def __init__(self, num_capsules, capsule_dim, heads=4):
        super(SelfAttentionRouting, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=capsule_dim, num_heads=heads)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, capsules, return_attention=False):
        # capsules: expected shape (B, N, D) or (B, T, N, D)
        if capsules.dim() == 3:
            B, N, D = capsules.size()
            capsules_trans = capsules.transpose(0, 1)  # (N, B, D)
            routed_capsules, attn_weights = self.multihead_attn(
                capsules_trans, capsules_trans, capsules_trans, need_weights=True, average_attn_weights=False
            )
            routed_capsules = self.dropout(routed_capsules)
            routed_capsules = routed_capsules.transpose(0, 1)  # (B, N, D)
        elif capsules.dim() == 4:
            B, T, N, D = capsules.size()
            capsules_reshape = capsules.view(B * T, N, D).transpose(0, 1)  # (N, B*T, D)
            routed_capsules, attn_weights = self.multihead_attn(
                capsules_reshape, capsules_reshape, capsules_reshape, need_weights=True, average_attn_weights=False
            )
            routed_capsules = self.dropout(routed_capsules)
            routed_capsules = routed_capsules.transpose(0, 1).view(B, T, N, D)
        else:
            raise ValueError("Expected capsules of shape (B, N, D) or (B, T, N, D), got {}".format(capsules.size()))
        if return_attention:
            return routed_capsules, attn_weights
        else:
            return routed_capsules

class RegionDecoder(nn.Module):
    """Region-specific GRU decoder."""
    def __init__(self, capsule_dim, hidden_dim, output_dim):
        super(RegionDecoder, self).__init__()
        self.gru = nn.GRU(capsule_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)  # Added Dropout

    def forward(self, capsules):
        if len(capsules.size()) == 3:
            B, N, D = capsules.size()
            T = 1
            capsules = capsules.unsqueeze(1)  # Add temporal dimension
        elif len(capsules.size()) == 4:
            B, T, N, D = capsules.size()
        else:
            raise ValueError("[RegionDecoder] Expected 3D or 4D input, got input with size {}".format(capsules.size()))

        capsules = capsules.reshape(B, T * N, D)
        _, hidden = self.gru(capsules)
        hidden = self.dropout(hidden)  # Apply Dropout
        output = self.fc(hidden.squeeze(0))
        return output

class GazeFusion(nn.Module):
    """Fuses region-specific outputs."""
    def __init__(self, input_dim, output_dim):
        super(GazeFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)  # Added Dropout

    def forward(self, regions):
        if isinstance(regions, list):
            fused = torch.cat(regions, dim=1)
        else:
            fused = regions
        fused = self.dropout(fused)  # Apply Dropout
        return self.fc(fused)

# Modified GazeEstimationModel returns both capsule maps and attention weights.
class GazeEstimationModel(nn.Module):
    def __init__(self, encoder, capsule_dim=64, hidden_dim=128, output_dim=2):
        super(GazeEstimationModel, self).__init__()
        self.encoder = encoder
        # in_channels=1024 from FrozenEncoder output
        self.capsule_formation = CapsuleFormation(in_channels=1024, num_capsules=8, capsule_dim=capsule_dim)
        self.routing = SelfAttentionRouting(num_capsules=8, capsule_dim=capsule_dim)
        self.eye_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)
        self.face_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)
        self.fusion = GazeFusion(output_dim * 2, output_dim)

    def forward(self, x, return_capsules=False, return_attention=False):
        # x: shape (B, T, C, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.encoder(x)  # (B*T, 1024, H_enc, W_enc)
        H_enc, W_enc = features.size(-2), features.size(-1)
        features = features.view(B, T, 1024, H_enc, W_enc)
        # Get capsule maps with spatial dimensions: (B, T, num_capsules, capsule_dim, H_enc, W_enc)
        capsules = self.capsule_formation(features)
        # For routing, pool spatially (e.g. by averaging over H_enc and W_enc)
        capsules_flat = capsules.mean(dim=(-2, -1))  # shape: (B, T, num_capsules, capsule_dim)
        # For T=1, squeeze temporal dimension:
        if T == 1:
            capsules_flat = capsules_flat.squeeze(1)  # (B, num_capsules, capsule_dim)
        if return_attention:
            routed_capsules, attn_weights = self.routing(capsules_flat, return_attention=True)
        else:
            routed_capsules = self.routing(capsules_flat)
        eye_output = self.eye_decoder(routed_capsules)
        face_output = self.face_decoder(routed_capsules)
        combined_output = torch.cat([eye_output, face_output], dim=1)
        output = self.fusion(combined_output)
        if return_capsules and return_attention:
            return output, capsules, attn_weights
        if return_capsules:
            return output, capsules
        if return_attention:
            return output, attn_weights
        return output


