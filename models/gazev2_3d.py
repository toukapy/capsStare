import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class FrozenEncoder(nn.Module):
    """Extractor de características con ConvNeXt-Base congelado."""

    def __init__(self, trainable_layers=10):
        super(FrozenEncoder, self).__init__()
        base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.features = base_model.features
        for param in self.features.parameters():
            param.requires_grad = False
        for param in list(self.features.parameters())[-trainable_layers:]:
            param.requires_grad = True
        self.norm = nn.BatchNorm2d(1024)

    def forward(self, x):
        features = self.features(x)
        features = self.norm(features)
        return features


class CapsuleFormation(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim):
        super(CapsuleFormation, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.linear = nn.Linear(input_dim, num_capsules * capsule_dim)
        self.norm = nn.LayerNorm(capsule_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, features):
        if len(features.size()) == 3:
            B, T, flattened_dim = features.size()
            features = features.view(B * T, flattened_dim)
        elif len(features.size()) == 5:
            B, T, C, H, W = features.size()
            flattened_dim = C * H * W
            features = features.view(B * T, flattened_dim)
        else:
            raise ValueError("[CapsuleFormation] Se esperaba entrada 3D o 5D, se recibió: {}".format(features.size()))
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
        self.multihead_attn = nn.MultiheadAttention(embed_dim=capsule_dim, num_heads=heads)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, capsules):
        if len(capsules.size()) == 3:
            B, N, D = capsules.size()
            T = 1
            capsules = capsules.unsqueeze(1)
        elif len(capsules.size()) == 4:
            B, T, N, D = capsules.size()
        else:
            raise ValueError(
                "[SelfAttentionRouting] Se esperaba entrada 3D o 4D, se recibió: {}".format(capsules.size()))
        capsules = capsules.view(B * T * N, D).unsqueeze(0)
        routed_capsules, _ = self.multihead_attn(capsules, capsules, capsules)
        routed_capsules = self.dropout(routed_capsules)
        routed_capsules = routed_capsules.squeeze(0).view(B, T, N, D)
        return routed_capsules


class RegionDecoder(nn.Module):
    """Decodificador GRU para cada región."""

    def __init__(self, capsule_dim, hidden_dim, output_dim):
        super(RegionDecoder, self).__init__()
        self.gru = nn.GRU(capsule_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, capsules):
        if len(capsules.size()) == 3:
            B, N, D = capsules.size()
            T = 1
            capsules = capsules.unsqueeze(1)
        elif len(capsules.size()) == 4:
            B, T, N, D = capsules.size()
        else:
            raise ValueError("[RegionDecoder] Se esperaba entrada 3D o 4D, se recibió: {}".format(capsules.size()))
        capsules = capsules.view(B, T * N, D)
        _, hidden = self.gru(capsules)
        hidden = self.dropout(hidden)
        output = self.fc(hidden.squeeze(0))
        return output


class GazeFusion(nn.Module):
    """Fusión de salidas de regiones visuales."""

    def __init__(self, input_dim, output_dim):
        super(GazeFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, regions):
        if isinstance(regions, list):
            fused = torch.cat(regions, dim=1)
        else:
            fused = regions
        fused = self.dropout(fused)
        return self.fc(fused)


class GazeEstimationModel(nn.Module):
    def __init__(self, encoder, capsule_dim=64, hidden_dim=128, output_dim=2, head_feature_dim=64, input_dim=50176):
        """
        Modelo de estimación de mirada mejorado con fusión temprana del head pose.

        Parámetros:
          - encoder: extractor de características (por ejemplo, FrozenEncoder).
          - capsule_dim: dimensión de cada cápsula.
          - hidden_dim: dimensión oculta del decodificador GRU.
          - output_dim: dimensión de salida (2 para [pitch, yaw]).
          - head_feature_dim: dimensión de la representación del head pose.
          - input_dim: dimensión original de la salida aplanada del encoder (para imágenes de 224×224 suele ser 50176).
        """
        super(GazeEstimationModel, self).__init__()
        self.encoder = encoder
        # MLP para procesar el head pose por frame (2 dimensiones -> head_feature_dim)
        self.head_pose_mlp = nn.Sequential(
            nn.Linear(2, head_feature_dim),
            nn.ReLU(),
            nn.Linear(head_feature_dim, head_feature_dim)
        )
        # La fusión temprana consiste en concatenar la representación visual y la del head pose.
        self.capsule_formation = CapsuleFormation(input_dim=input_dim + head_feature_dim, num_capsules=8,
                                                  capsule_dim=capsule_dim)
        self.routing = SelfAttentionRouting(num_capsules=8, capsule_dim=capsule_dim)
        self.eye_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)
        self.face_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)
        self.fusion = GazeFusion(output_dim * 2, output_dim)

    def forward(self, x, head_pose):
        """
        x: tensor de imágenes (B, T, C, H, W)
        head_pose: tensor de head pose (B, T, 2)
        """
        B, T, C, H, W = x.size()
        # Extraer características visuales
        x = x.view(B * T, C, H, W)
        features = self.encoder(x)  # (B*T, channels, H', W')
        features = features.reshape(B, T, -1)  # (B, T, input_dim)
        # Procesar head pose para cada frame
        head_features = self.head_pose_mlp(head_pose)  # (B, T, head_feature_dim)
        # Fusión temprana: concatenar a nivel de cada frame
        fused_features = torch.cat([features, head_features], dim=-1)  # (B, T, input_dim + head_feature_dim)
        # Continuar con la formación de cápsulas y routing
        capsules = self.capsule_formation(fused_features)
        routed_capsules = self.routing(capsules)
        eye_output = self.eye_decoder(routed_capsules)
        face_output = self.face_decoder(routed_capsules)
        combined_output = torch.cat([eye_output, face_output], dim=1)
        output = self.fusion(combined_output)
        return output



