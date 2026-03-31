import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 20

TRAIN_FILES = [
    "F:\\jet0run0\\run_0_chunk_0.pt",
]

print("DEVICE:", DEVICE)

# =========================
# DATASET
# =========================
class JetDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file, weights_only=True)
        self.lr = data["lr"].float()
        self.hr = data["hr"].float()

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = self.lr[idx]
        hr = self.hr[idx]

        # 🔥 per-sample normalization (IMPORTANT)
        scale = hr.max()
        lr = lr / (scale + 1e-8)
        hr = hr / (scale + 1e-8)

        return lr, hr


def get_loader(file):
    return DataLoader(JetDataset(file), batch_size=BATCH_SIZE, shuffle=True)

# =========================
# UNET GENERATOR (FIXED)
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)

        # Decoder with skip connections
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, x):
        x1 = F.leaky_relu(self.enc1(x), 0.2)
        x2 = F.leaky_relu(self.enc2(x1), 0.2)
        x3 = F.leaky_relu(self.enc3(x2), 0.2)

        x = F.relu(self.dec1(x3))
        x = torch.cat([x, x2], dim=1)

        x = F.relu(self.dec2(x))
        x = torch.cat([x, x1], dim=1)

        x = self.dec3(x)

        x = F.interpolate(x, size=(125,125))
        return x

# =========================
# DISCRIMINATOR
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, lr, hr):
        lr = F.interpolate(lr, size=(125,125))
        x = torch.cat([lr, hr], dim=1)
        return self.model(x)

# =========================
# INIT
# =========================
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# 🔥 weaker discriminator
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

L1 = nn.L1Loss()
BCE = nn.BCEWithLogitsLoss()

# =========================
# VISUALIZATION
# =========================
def visualize(G, loader, epoch):
    lr, hr = next(iter(loader))
    lr, hr = lr.to(DEVICE), hr.to(DEVICE)

    with torch.no_grad():
        fake = G(lr)

    lr = lr[0].cpu().permute(1,2,0).numpy()
    fake = fake[0].cpu().permute(1,2,0).numpy()
    hr = hr[0].cpu().permute(1,2,0).numpy()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(lr[:,:,0], cmap='inferno')
    plt.title("LR")

    plt.subplot(1,3,2)
    plt.imshow(fake[:,:,0], cmap='inferno')
    plt.title("Generated")

    plt.subplot(1,3,3)
    plt.imshow(hr[:,:,0], cmap='inferno')
    plt.title("HR")

    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f"epoch_{epoch}.png")
    plt.close()

# =========================
# TRAIN STEP
# =========================
def train_step(lr, hr):
    lr, hr = lr.to(DEVICE), hr.to(DEVICE)

    # ---- Train D ----
    fake = G(lr).detach()
    real_pred = D(lr, hr)
    fake_pred = D(lr, fake)

    loss_D = (
        BCE(real_pred, torch.ones_like(real_pred)) +
        BCE(fake_pred, torch.zeros_like(fake_pred))
    )

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # ---- Train G (twice) ----
    for _ in range(2):
        fake = G(lr)
        fake_pred = D(lr, fake)

        loss_G = (
            BCE(fake_pred, torch.ones_like(fake_pred)) +
            100 * L1(fake, hr)
        )

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    return loss_G.item(), loss_D.item()

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")

    for file in TRAIN_FILES:
        loader = get_loader(file)

        for i, (lr, hr) in enumerate(loader):
            g_loss, d_loss = train_step(lr, hr)

            if i % 100 == 0:
                print(f"[{i}] G: {g_loss:.3f} | D: {d_loss:.3f}")

    visualize(G, loader, epoch+1)

# =========================
# SAVE
# =========================
torch.save(G.state_dict(), "generator_fixed.pth")
print("DONE")