import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class FrozenEncoder(nn.Module):
    """Frozen backbone for feature extraction using ConvNeXt-Base."""

    def __init__(self, trainable_layers=0):
        super(FrozenEncoder, self).__init__()
        base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.features = base_model.features

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.norm = nn.BatchNorm2d(1024)  # ConvNeXt-Base has 1024 output channels

    def forward(self, x):
        features = self.features(x)
        features = self.norm(features)
        return features


class SelfAttentionRouting(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(SelfAttentionRouting, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.5)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        # Input shape: (B, T, feature_dim)
        # Add self-attention with residual connection
        attn_output, _ = self.attention(features, features, features)
        attn_output = self.dropout(attn_output)
        output = self.norm(features + attn_output)  # Residual connection
        return output


class RegionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegionDecoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Input shape: (B, T, input_dim)
        _, hidden = self.gru(x)  # Process sequence
        hidden = self.dropout(hidden[-1])  # Take last hidden state
        return self.fc(hidden)


class GazeEstimationModel(nn.Module):
    def __init__(self, encoder, hidden_dim=256, output_dim=2):
        super(GazeEstimationModel, self).__init__()
        self.encoder = encoder
        self.feature_dim = 1024 * 7 * 7  # Original dimension

        # Rest of the model remains the same
        self.eye_decoder = RegionDecoder(self.feature_dim, hidden_dim, output_dim)
        self.face_decoder = RegionDecoder(self.feature_dim, hidden_dim, output_dim)
        self.fusion = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.encoder(x)  # [B*T, 1024, 7, 7]
        features = features.reshape(B, T, -1)  # [B, T, 1024*7*7]

        eye_output = self.eye_decoder(features)
        face_output = self.face_decoder(features)
        return self.fusion(torch.cat([eye_output, face_output], dim=1))