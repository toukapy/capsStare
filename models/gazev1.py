import torch

import torch.nn as nn

import torch.nn.functional as F


from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights



class FrozenEncoder(nn.Module):


    """ Frozen backbone for feature extraction using EfficientNet-B4."""


    def __init__(self, trainable_layers=5):

        super(FrozenEncoder, self).__init__()


        base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        self.features = nn.Sequential(*list(base_model.children())[:-2])

        for param in self.features.parameters():

            param.requires_grad = False


        for param in list(self.features.parameters())[-trainable_layers:]:


            param.requires_grad = True


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

        self.linear = nn.Linear(input_dim, num_capsules * capsule_dim)


        self.norm = nn.LayerNorm(capsule_dim)


        self.activation = nn.GELU()


        self.dropout = nn.Dropout(p=0.5)  # Added Dropout



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


        self.dropout = nn.Dropout(p=0.5)  # Added Dropout



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


        self.dropout = nn.Dropout(p=0.5)  # Added Dropout



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


        self.capsule_formation = CapsuleFormation(input_dim=87808, num_capsules=8, capsule_dim=capsule_dim)

        self.routing = SelfAttentionRouting(num_capsules=4, capsule_dim=capsule_dim)



        self.eye_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)

        self.face_decoder = RegionDecoder(capsule_dim, hidden_dim, output_dim)



        self.fusion = GazeFusion(output_dim * 2, output_dim)



    def forward(self, x):


        B, T, C, H, W = x.size()


        x = x.view(B * T, C, H, W)


        features = self.encoder(x)


        features = features.view(B, T, -1)



        capsules = self.capsule_formation(features)

        routed_capsules = self.routing(capsules)


        eye_output = self.eye_decoder(routed_capsules)


        face_output = self.face_decoder(routed_capsules)


        combined_output = torch.cat([eye_output, face_output], dim=1)


        output = self.fusion(combined_output)

        return output