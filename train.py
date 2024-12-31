from models import AdaptiveEncoder, GazePredictionPipeline
import torch
from torch import nn
from tqdm import tqdm
from utils.utils import infer_region_masks


# Initialize components
feature_dim = 2048
hidden_dim = 512
output_dim = 3
encoder = AdaptiveEncoder(feature_dim)
model = GazePredictionPipeline(encoder, feature_dim, hidden_dim, output_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()
epochs = 100

# Training loop
# Example integration in the training loop
for epoch in range(10):
    model.train()
    for images, targets in train_loader:  # Images: (B, C, H, W), Targets: (B, OutputDim)
        images, targets = images.cuda(), targets.cuda()

        # Infer region masks dynamically
        region_masks = infer_region_masks(images).cuda()

        # Forward pass
        predictions = model(images, region_masks)

        # Compute loss
        loss = nn.MSELoss()(predictions, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
