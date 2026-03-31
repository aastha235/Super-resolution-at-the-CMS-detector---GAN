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
EPOCHS = 10   # 5–10 recommended

TRAIN_FILES = [
    "F:\\jet0run0\\run_1_chunk_0.pt",
    "F:\\jet0run0\\run_1_chunk_1.pt",
    "F:\\jet0run0\\run_1_chunk_2.pt",
    "F:\\jet0run0\\run_1_chunk_3.pt",
    "F:\\jet0run0\\run_1_chunk_4.pt"
]

print("DEVICE:", DEVICE)

# =========================
# DATASET
# =========================
class JetDataset(Dataset):
    def __init__(self, file):
        data = torch.load(file, weights_only=True)
        self.lr = data["lr"].float()
        self.hr = data["hr"].float()

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = self.lr[idx]
        hr = self.hr[idx]

        # per-sample normalization
        scale = hr.max()
        lr = lr / (scale + 1e-8)
        hr = hr / (scale + 1e-8)

        return lr, hr


def get_loader(file):
    return DataLoader(JetDataset(file), batch_size=BATCH_SIZE, shuffle=True)

# =========================
# GENERATOR (UNet)
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)

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
        return self.model(torch.cat([lr, hr], dim=1))

# =========================
# INIT
# =========================
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))

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
    plt.subplot(1,3,1); plt.imshow(lr[:,:,0], cmap='inferno'); plt.title("LR")
    plt.subplot(1,3,2); plt.imshow(fake[:,:,0], cmap='inferno'); plt.title("Generated")
    plt.subplot(1,3,3); plt.imshow(hr[:,:,0], cmap='inferno'); plt.title("HR")
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f"epoch_3_{epoch}.png")
    plt.close()

# =========================
# TRAIN STEP
# =========================
def train_step(lr, hr):
    lr, hr = lr.to(DEVICE), hr.to(DEVICE)

    # ---- Train D ----
    fake = G(lr).detach()
    loss_D = (
        BCE(D(lr, hr), torch.ones_like(D(lr, hr))) +
        BCE(D(lr, fake), torch.zeros_like(D(lr, fake)))
    )

    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # ---- Train G twice ----
    for _ in range(2):
        fake = G(lr)
        loss_G = (
            BCE(D(lr, fake), torch.ones_like(D(lr, fake))) +
            50 * L1(fake, hr)
        )

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    return loss_G.item(), loss_D.item()

# =========================
# TRAIN LOOP + SAVE
# =========================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")
    start_time=time.time()
    for file in TRAIN_FILES:
        loader = get_loader(file)

        for i, (lr, hr) in enumerate(loader):
            g_loss, d_loss = train_step(lr, hr)

            if i % 250 == 0:
                print(f"[{i}] G: {g_loss:.3f} | D: {d_loss:.3f}")
    
    visualize(G, loader, epoch+1)

    # 🔥 SAVE EACH EPOCH
    torch.save(G.state_dict(), f"generator_3_epoch_{epoch+1}.pth")
    end_time=time.time()
    print(f"Total time : {end_time-start_time}")

print("Training done!")

# =========================
# EVALUATION
# =========================
print("\nRunning evaluation...")

loader = get_loader(TRAIN_FILES[0])
lr, hr = next(iter(loader))
lr, hr = lr.to(DEVICE), hr.to(DEVICE)

with torch.no_grad():
    fake = G(lr)

real_energy = hr.sum(dim=[1,2,3]).cpu()
fake_energy = fake.sum(dim=[1,2,3]).cpu()

# Scatter plot
plt.scatter(real_energy, fake_energy)
plt.xlabel("Real Energy")
plt.ylabel("Generated Energy")
plt.title("Energy Correlation")
plt.savefig(f"energy_corr_3.png")
plt.close()

print("Evaluation done!")

# =========================
# MULTIPLE SAMPLE VISUALIZATION
# =========================
print("\nGenerating multiple samples...")

def visualize_multiple(G, loader, num_samples=5):
    G.eval()

    lr_batch, hr_batch = next(iter(loader))
    lr_batch = lr_batch.to(DEVICE)
    hr_batch = hr_batch.to(DEVICE)

    with torch.no_grad():
        fake_batch = G(lr_batch)

    for i in range(num_samples):
        lr = lr_batch[i].cpu().permute(1,2,0).numpy()
        fake = fake_batch[i].cpu().permute(1,2,0).numpy()
        hr = hr_batch[i].cpu().permute(1,2,0).numpy()

        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(lr[:,:,0], cmap='inferno')
        plt.title(f"LR (sample {i})")

        plt.subplot(1,3,2)
        plt.imshow(fake[:,:,0], cmap='inferno')
        plt.title("Generated")

        plt.subplot(1,3,3)
        plt.imshow(hr[:,:,0], cmap='inferno')
        plt.title("HR")

        plt.savefig(f"eval_3_{i}.png")
        plt.close()

# Run it
visualize_multiple(G, loader, num_samples=5)