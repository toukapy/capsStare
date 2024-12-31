import torch.nn as nn
import torch

class GazePredictionBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GazePredictionBranch, self).__init__()
        self.adapter = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, bach_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.adapter(x).unsqueeze(1)
        _, hidden = self.gru(x)
        gaze_pred = self.fc(hidden.squeeze(0))
        return gaze_pred
    

class GazePredictionPipeline(nn.Module):
    def __init__(self, encoder, feature_dim, hidden_dim, output_dim):
        super(GazePredictionPipeline, self).__init__()
        self.encoder = encoder

        self.eye_branch = GazePredictionBranch(feature_dim, hidden_dim, output_dim)
        self.face_branch = GazePredictionBranch(feature_dim, hidden_dim, output_dim)
        self.body_branch = GazePredictionBranch(feature_dim, hidden_dim, output_dim)

        self.fusion_layer = nn.Linear(3 * output_dim, output_dim)
    
    def forward(self, x, region_masks):
        # Extract features
        features = self.encoder(x)

        # Branch predictions
        eye_gaze = self.eye_branch(features)
        face_gaze = self.face_branch(features)
        body_gaze = self.body_branch(features)

        # Apply region masks (1 if present, 0 if missing)
        if region_masks is not None:
            eye_gaze = region_masks[:, 0].unsqueeze(1) * eye_gaze
            face_gaze = region_masks[:, 1].unsqueeze(1) * face_gaze
            body_gaze = region_masks[:, 2].unsqueeze(1) * body_gaze

            # Normalize fusion weights based on available regions
            valid_regions = region_masks.sum(dim=1, keepdim = True)
            valid_regions[valid_regions == 0] = 1
            fusion_weights = region_masks / valid_regions
        else:
            fusion_weights = torch.ones(x.size(0), 3).to(x.device) / 3
        
        # Fuse predictions
        fused_gaze = self.fusion_layer(
            torch.cat([
                fusion_weights[:, 0].unsqueeze(1) * eye_gaze,
                fusion_weights[:, 1].unsqueeze(1) * face_gaze,
                fusion_weights[:, 2].unsqueeze(1) * body_gaze
            ], dim = 1)
        )

        return fused_gaze