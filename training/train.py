import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from training.dataset import DayNightDataset
from models.generator import GeneratorUNet
from models.discriminator import Discriminator

# -----------------------------
# Hyperparameters
# -----------------------------
epochs = 10
batch_size = 1
lr = 2e-4
lambda_l1 = 100
sample_interval = 200  # save sample every 200 batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Create models
# -----------------------------
G = GeneratorUNet().to(device)
D = Discriminator().to(device)

# -----------------------------
# Optimizers
# -----------------------------
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# -----------------------------
# Loss functions
# -----------------------------
criterion_GAN = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()

# -----------------------------
# Dataset & loader
# -----------------------------
dataset = DayNightDataset('.', transform=None)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

os.makedirs("outputs/samples", exist_ok=True)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

    for i, batch in loop:
        real_A = batch["day"].to(device)
        real_B = batch["night"].to(device)

        # ----------------------
        # Dynamic valid / fake labels (Fixes shape mismatch)
        # ----------------------
        with torch.no_grad():
            pred_shape = D(real_B, real_A).shape

        valid = torch.ones(pred_shape, device=device)
        fake  = torch.zeros(pred_shape, device=device)

        # ----------------------
        # Train Generator
        # ----------------------
        optimizer_G.zero_grad()

        fake_B = G(real_A)
        pred_fake = D(fake_B, real_A)

        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1  = criterion_L1(fake_B, real_B)

        loss_G = loss_GAN + lambda_l1 * loss_L1
        loss_G.backward()
        optimizer_G.step()

        # ----------------------
        # Train Discriminator
        # ----------------------
        optimizer_D.zero_grad()

        pred_real = D(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = D(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        loop.set_postfix({
            "Loss_G": round(loss_G.item(), 4),
            "Loss_D": round(loss_D.item(), 4)
        })

        # ----------------------
        # Save sample outputs
        # ----------------------
        if i % sample_interval == 0:
            imgs = torch.cat((real_A, fake_B, real_B), dim=3)
            save_image(imgs, f"outputs/samples/{epoch}_{i}.png", normalize=True)

    # ----------------------
    # Save model checkpoints
    # ----------------------
    torch.save(G.state_dict(), f"outputs/G_epoch_{epoch+1}.pth")
    torch.save(D.state_dict(), f"outputs/D_epoch_{epoch+1}.pth")

print("Training finished!")
