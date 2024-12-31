import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
from models.backbone import AdaptiveEncoder

# Define transformations for self-supervised learning
transform = transforms.Compose([
    transforms.RandomResizedCrop(224), 
    transforms.RandomHorizontalFlip(), 
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = ImageFolder(root="path_to_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the encoder
feature_dim = 2048
projection_dim = 128
encoder = AdaptiveEncoder(feature_dim)
projection_head = nn.Sequential(
    nn.Linear(feature_dim, projection_dim),
    nn.ReLU(),
    nn.Linear(projection_dim, projection_dim)
)
encoder = encoder.cuda()
projection_head = projection_head.cuda()

# Define optimizer
optimizer = optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=0.001)

# Contrastive loss function
def contrastive_loss(z_i, z_j, temperature=0.5):
    """Computes contrastive loss for a batch."""
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # Combine positive pairs
    z = nn.functional.normalize(z, dim=1)  # Normalize feature vectors
    similarity_matrix = torch.matmul(z, z.T)  # Cosine similarity

    # Create labels for contrastive loss
    labels = torch.arange(batch_size).repeat(2).cuda()
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # Mask self-similarity
    mask = torch.eye(labels.size(0), device=labels.device).bool()
    labels = labels.masked_fill(mask, 0)

    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature

    # Compute loss
    exp_sim = torch.exp(similarity_matrix)
    loss = -torch.log(exp_sim / exp_sim.sum(dim=1)).mean()
    return loss

# Training loop
epochs = 10
encoder.train()
projection_head.train()

for epoch in range(epochs):
    total_loss = 0
    for images, _ in dataloader:
        images = images.cuda()

        # Generate two augmentations for each image
        augment_1 = transform(images)
        augment_2 = transform(images)

        # Forward pass through encoder and projection head
        features_1 = encoder(augment_1)
        features_2 = encoder(augment_2)

        projections_1 = projection_head(features_1)
        projections_2 = projection_head(features_2)

        # Compute contrastive loss
        loss = contrastive_loss(projections_1, projections_2)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")