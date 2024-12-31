import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# SimCLR Projection Head
class SimCLRHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimCLRHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Gaze Heatmap Prediction Head
class GazeHeatMapHead(nn.Module):
    def __init__(self, input_dim, heatmap_dim):
        super(GazeHeatMapHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, heatmap_dim, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x
    

# Masked Reconstruction Head
class MaskedReconstructor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MaskedReconstructor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.fc(x)
        return x
    
# Full Encoder with SSL Heads
class GazeSelfSupervisionEncoder(nn.Module):
    def __init__(self, feature_dim, projection_dim, heatmap_dim):
        super(GazeSelfSupervisionEncoder, self).__init__()

        # Pretrained backbone
        self.encoder = models.resnet50(pretrained = True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # SSL heads
        self.simclr_head = SimCLRHead(feature_dim, projection_dim)
        self.heatmap_head = GazeHeatMapHead(feature_dim, heatmap_dim)
        self.reconstructor = MaskedReconstructor(feature_dim, feature_dim)

    def forward(self, x):
        features = self.encoder(x)
        contrastive_output = self.simclr_head(features.mean(dim=[2, 3]))
        heatmap_output = self.heatmap_head(features)
        reconstructor_output = self.reconstructor(features.mean(dim=[2, 3]))
        return features, contrastive_output, heatmap_output, reconstructor_output
    
class AdaptiveEncoder(nn.Module):
    def __init__(self, feature_dim = 2048):
        super(AdaptiveEncoder, self).__init__()

        base_model = models.resnet50(pretrained = True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim

    def forward(self, x):
        features = self.encoder(x)
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        return pooled_features
        