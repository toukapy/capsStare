import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class FrozenEncoder(nn.Module):
    """Frozen backbone for feature extraction using EfficientNet-B4."""

    def __init__(self, trainable_layers=0):
        super(FrozenEncoder, self).__init__()
        # Load EfficientNet-B4 with pretrained weights
        base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Use the features from EfficientNet
        self.features = base_model.features

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # EfficientNet-B4 final features have 1792 channels
        self.norm = nn.BatchNorm2d(1792)

    def forward(self, x):
        features = self.features(x)
        features = self.norm(features)
        return features


class CapsuleFormation(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim):
        super(CapsuleFormation, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        # Calculate flattened dimension for EfficientNet-B4 features (7x7x1792)
        self.linear = nn.Linear(input_dim, num_capsules * capsule_dim)
        self.norm = nn.LayerNorm(capsule_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, features):
        B, T, C, H, W = features.size()
        features = features.view(B * T, C, H, W)
        features = features.flatten(1)  # Flatten spatial dimensions

        capsules = self.linear(features)
        capsules = self.dropout(capsules)
        capsules = capsules.view(B * T, self.num_capsules, self.capsule_dim)
        capsules = self.norm(capsules)
        capsules = self.activation(capsules)
        capsules = capsules.view(B, T, self.num_capsules, self.capsule_dim)
        return capsules


class SelfAttentionRouting(nn.Module):
    def __init__(self, num_capsules, capsule_dim, heads=4):
        super(SelfAttentionRouting, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=capsule_dim,
            num_heads=heads,
            batch_first=True  # Added for compatibility
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, capsules):
        B, T, N, D = capsules.size()

        # Merge batch and temporal dimensions
        capsules = capsules.reshape(B * T, N, D)

        # Self-attention with residual connection
        attn_output, _ = self.multihead_attn(capsules, capsules, capsules)
        attn_output = self.dropout(attn_output)

        # Restore original dimensions
        attn_output = attn_output.reshape(B, T, N, D)
        return attn_output


class RegionDecoder(nn.Module):
    def __init__(self, capsule_dim, hidden_dim, output_dim):
        super(RegionDecoder, self).__init__()
        self.gru = nn.GRU(capsule_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, capsules):
        B, T, N, D = capsules.size()

        # Process each capsule independently through GRU
        capsules = capsules.reshape(B, T * N, D)
        _, hidden = self.gru(capsules)
        hidden = self.dropout(hidden[-1])  # Take last hidden state
        output = self.fc(hidden)
        return output


class GazeFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GazeFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, fused_input):  # Expects pre-concatenated input
        fused = self.dropout(fused_input)
        return self.fc(fused)


class GazeEstimationModel(nn.Module):
    def __init__(self, encoder, capsule_dim=128, hidden_dim=256, output_dim=2):
        super(GazeEstimationModel, self).__init__()
        self.encoder = encoder

        # EfficientNet-B4 features are 7x7x1792 = 87,808 when flattened
        self.capsule_formation = CapsuleFormation(
            input_dim=7 * 7 * 1792,  # Adjusted for EfficientNet-B4
            num_capsules=4,
            capsule_dim=capsule_dim
        )

        self.routing = SelfAttentionRouting(
            num_capsules=4,
            capsule_dim=capsule_dim
        )

        self.eye_decoder = RegionDecoder(
            capsule_dim,
            hidden_dim,
            output_dim
        )

        self.face_decoder = RegionDecoder(
            capsule_dim,
            hidden_dim,
            output_dim
        )

        self.fusion = GazeFusion(
            output_dim * 2,
            output_dim
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        # Get features from EfficientNet
        features = self.encoder(x)  # [B*T, 1792, 7, 7]

        # Reshape for capsules
        features = features.view(B, T, 1792, 7, 7)

        # Form capsules
        capsules = self.capsule_formation(features)

        # Route capsules
        routed_capsules = self.routing(capsules)

        # Decode regions
        eye_output = self.eye_decoder(routed_capsules)
        face_output = self.face_decoder(routed_capsules)

        # Fuse outputs
        combined_output = torch.cat([eye_output, face_output], dim=1)
        output = self.fusion(combined_output)

        return output