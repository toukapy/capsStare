import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class FrozenEncoder(nn.Module):
    """Frozen backbone for feature extraction using a Vision Transformer (ViT)."""

    def __init__(self, trainable_layers=0):
        super(FrozenEncoder, self).__init__()
        # Load the pretrained ViT model.
        base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        for param in base_model.parameters():
            param.requires_grad = False

        # Optionally unfreeze the last few transformer blocks.
        if trainable_layers > 0:
            for block in base_model.blocks[-trainable_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Remove the classification head.
        base_model.head = nn.Identity()
        self.transformer = base_model

        # Get the embedding dimension from the class token.
        embed_dim = self.transformer.class_token.shape[-1]

        # Use positional embeddings if available; otherwise, create our own.
        if hasattr(base_model, "pos_embed"):
            self.pos_embed = base_model.pos_embed
        else:
            # For vit_b_16: image_size=224, patch_size=16 -> 14x14 patches + 1 class token = 197 tokens.
            num_tokens = 1 + (224 // 16) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x):
        # Patch embedding
        x = self.transformer.conv_proj(x)  # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # Prepend the class token
        cls_tokens = self.transformer.class_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embeddings (using our custom pos_embed)
        x = x + self.pos_embed
        if hasattr(self.transformer, "pos_drop"):
            x = self.transformer.pos_drop(x)
        # Use the encoder layers (Torchvision ViT uses transformer.encoder.layers)
        for block in self.transformer.encoder.layers:
            x = block(x)
        # If a final normalization exists, use it; otherwise, skip.
        if hasattr(self.transformer, "norm"):
            x = self.transformer.norm(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)  # (B, num_tokens, D)
        # Remove the class token to get patch tokens.
        patch_tokens = features[:, 1:]
        B, N, D = patch_tokens.shape
        patch_dim = int(N ** 0.5)
        # Reshape tokens into a 2D feature map.
        patch_tokens = patch_tokens.transpose(1, 2).view(B, D, patch_dim, patch_dim)
        return patch_tokens


class CapsuleFormation(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim):
        super(CapsuleFormation, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.linear = nn.Linear(input_dim, num_capsules * capsule_dim)
        self.norm = nn.LayerNorm(capsule_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)  # Added Dropout

    def forward(self, features):
        if len(features.size()) == 3:  # Handle inputs already flattened
            B, T, flattened_dim = features.size()
            features = features.view(B * T, flattened_dim)
        elif len(features.size()) == 5:
            B, T, C, H, W = features.size()
            flattened_dim = C * H * W
            features = features.view(B * T, flattened_dim)
        else:
            raise ValueError("[CapsuleFormation] Expected 3D or 5D input, got input with size {}".format(features.size()))

        capsules = self.linear(features)
        capsules = self.dropout(capsules)  # Apply Dropout
        capsules = capsules.view(B * T, self.num_capsules, self.capsule_dim)
        capsules = self.norm(capsules)
        capsules = self.activation(capsules)
        capsules = capsules.view(B, T, self.num_capsules, self.capsule_dim)  # Restore batch and temporal structure
        return capsules

class SelfAttentionRouting(nn.Module):
    def __init__(self, num_capsules, capsule_dim, heads=4):
        super(SelfAttentionRouting, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=capsule_dim, num_heads=heads)
        self.dropout = nn.Dropout(p=0.5)  # Added Dropout

    def forward(self, capsules):
        if len(capsules.size()) == 3:
            B, N, D = capsules.size()
            T = 1
            capsules = capsules.unsqueeze(1)  # Add temporal dimension
        elif len(capsules.size()) == 4:
            B, T, N, D = capsules.size()
        else:
            raise ValueError("[SelfAttentionRouting] Expected 3D or 4D input, got input with size {}".format(capsules.size()))

        capsules = capsules.view(B * T * N, D).unsqueeze(0)  # Merge batch and temporal dimensions
        routed_capsules, _ = self.multihead_attn(capsules, capsules, capsules)
        routed_capsules = self.dropout(routed_capsules)  # Apply Dropout
        routed_capsules = routed_capsules.squeeze(0).view(B, T, N, D)  # Restore original dimensions
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

        capsules = capsules.view(B, T * N, D)
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

class GazeEstimationModel(nn.Module):
    def __init__(self, encoder, capsule_dim=64, hidden_dim=128, output_dim=2):
        super(GazeEstimationModel, self).__init__()
        self.encoder = encoder
        self.capsule_formation = CapsuleFormation(input_dim=150528, num_capsules=8, capsule_dim=capsule_dim)
        self.routing = SelfAttentionRouting(num_capsules=8, capsule_dim=capsule_dim)
        self.eye_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)
        self.face_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)
        self.fusion = GazeFusion(output_dim * 2, output_dim)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.encoder(x)
        features = features.reshape(B, T, -1)
        capsules = self.capsule_formation(features)
        routed_capsules = self.routing(capsules)
        eye_output = self.eye_decoder(routed_capsules)
        face_output = self.face_decoder(routed_capsules)
        combined_output = torch.cat([eye_output, face_output], dim=1)
        output = self.fusion(combined_output)
        return output


