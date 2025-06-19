import torch
import torch.nn as nn
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

        self.norm = nn.BatchNorm2d(1024)  # Output channels of convnext_base

    def forward(self, x):
        features = self.features(x)
        features = self.norm(features)
        return features


class FeatureProjection(nn.Module):
    """Non-learnable projection by flattening and reshaping feature map into region tokens."""

    def __init__(self, input_dim, num_regions, region_dim):
        super(FeatureProjection, self).__init__()
        assert input_dim == num_regions * region_dim, \
            f"Input dim {input_dim} must equal num_regions × region_dim = {num_regions * region_dim}"
        self.num_regions = num_regions
        self.region_dim = region_dim

    def forward(self, x):
        # Input: (B, T, C, H, W) — usually (B, T, 1024, 7, 7)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C * H * W)  # (B*T, input_dim)
        elif x.dim() == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)  # (B*T, input_dim)

        # Reshape directly into region tokens
        x = x.view(-1, self.num_regions, self.region_dim)  # (B*T, N, D)

        # Reshape to (B, T, N, D)
        B_T = x.shape[0]
        B = B_T // T
        return x.contiguous().view(B, T, self.num_regions, self.region_dim)


class SelfAttentionRouting(nn.Module):
    """Self-attention routing with optional projection to desired embed_dim."""

    def __init__(self, num_regions, input_dim, embed_dim=128, heads=4):
        super(SelfAttentionRouting, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        if input_dim != embed_dim:
            self.proj = nn.Linear(input_dim, embed_dim, bias=False)  # match attention dims
        else:
            self.proj = nn.Identity()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, regions):
        # regions: (B, T, N, input_dim)
        B, T, N, D = regions.shape
        regions = regions.view(B * T, N, D)  # for batch_first=True

        regions = self.proj(regions)  # (B*T, N, embed_dim)
        routed, _ = self.multihead_attn(regions, regions, regions)
        routed = self.dropout(routed)
        return routed.contiguous().view(B, T, N, self.embed_dim)


class RegionDecoder(nn.Module):
    """Region-specific GRU decoder (unchanged)"""

    def __init__(self, region_dim, hidden_dim, output_dim):
        super(RegionDecoder, self).__init__()
        self.gru = nn.GRU(region_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, regions):
        if regions.dim() == 3:  # (B, N, D)
            regions = regions.unsqueeze(1)  # Add temporal dim
        B, T, N, D = regions.shape
        regions = regions.reshape(B, T * N, D)
        _, hidden = self.gru(regions)
        return self.fc(self.dropout(hidden.squeeze(0)))


class GazeFusion(nn.Module):
    """Fuses region-specific outputs (unchanged)"""

    def __init__(self, input_dim, output_dim):
        super(GazeFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, regions):
        fused = torch.cat(regions, dim=1) if isinstance(regions, list) else regions
        return self.fc(self.dropout(fused))


class GazeEstimationMLP(nn.Module):
    """Modified model keeping self-attention but replacing capsules with MLP"""

    def __init__(self, encoder, region_dim=128, hidden_dim=256, output_dim=2):
        super(GazeEstimationMLP, self).__init__()
        self.encoder = encoder
        self.feature_projection = FeatureProjection(input_dim=1024*7*7, num_regions=4, region_dim=12544)
        self.routing = SelfAttentionRouting(
            num_regions=4,
            input_dim=12544,  # input dim from non-trainable projection
            embed_dim=128  # attention output dimension
        )
        self.eye_decoder = RegionDecoder(region_dim, hidden_dim, output_dim)
        self.face_decoder = RegionDecoder(region_dim, hidden_dim, output_dim)
        self.fusion = GazeFusion(output_dim * 2, output_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Extract features
        x = x.view(B * T, C, H, W)
        features = self.encoder(x)  # (B*T, 1024, 7, 7)
        features = features.view(B, T, 1024, 7, 7)

        # Project to region features (replaces capsule formation)
        regions = self.feature_projection(features)  # (B, T, 4, region_dim)

        # Apply self-attention routing
        routed_regions = self.routing(regions)

        # Decode regions
        eye_output = self.eye_decoder(routed_regions)
        face_output = self.face_decoder(routed_regions)

        # Fuse outputs
        combined = torch.cat([eye_output, face_output], dim=1)
        output = self.fusion(combined)
        return output


