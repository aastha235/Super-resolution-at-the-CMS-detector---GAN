import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

data = torch.load("F:\\jet0run0\\run_0_chunk_0.pt",weights_only=True)

lr = data["lr"]   # (N, 3, 64, 64)
hr = data["hr"]   # (N, 3, 125, 125)
y  = data["y"]

print(lr.shape, hr.shape, y.shape)

idx = 0

lr_sample = lr[idx]   # (3,64,64)
hr_sample = hr[idx]   # (3,125,125)
label = y[idx]

print("Label:", label)
print("LR shape:", lr_sample.shape)
print("HR shape:", hr_sample.shape)

import numpy as np

lr_vis = lr_sample.permute(1,2,0).numpy()
hr_vis = hr_sample.permute(1,2,0).numpy()

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(lr_vis[:,:,0], cmap='inferno')
plt.title("LR Channel 0")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(hr_vis[:,:,0], cmap='inferno')
plt.title("HR Channel 0")
plt.colorbar()

plt.show()

for i in range(3):
    plt.imshow(hr_sample[i].numpy(), cmap='inferno')
    plt.title(f"HR Channel {i}")
    plt.colorbar()
    plt.show()

print("LR min/max:", lr_sample.min().item(), lr_sample.max().item())
print("HR min/max:", hr_sample.min().item(), hr_sample.max().item())

print("Zero ratio HR:", (hr_sample == 0).sum().item() / hr_sample.numel())
print("LR energy:", lr_sample.sum().item())
print("HR energy:", hr_sample.sum().item())


# find one of each quark vs gluon
q_idx = (y == 0).nonzero(as_tuple=True)[0][0].item()
g_idx = (y == 1).nonzero(as_tuple=True)[0][0].item()

q_img = hr[q_idx].permute(1,2,0).numpy()
g_img = hr[g_idx].permute(1,2,0).numpy()

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(q_img[:,:,0])
plt.title("Quark")

plt.subplot(1,2,2)
plt.imshow(g_img[:,:,0])
plt.title("Gluon")

plt.show()