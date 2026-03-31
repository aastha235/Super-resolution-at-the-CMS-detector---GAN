import numpy as np
import pandas as pd
import numpy as np
df = pd.read_parquet("F:\\jet0run0\\run_2_chunk_5.parquet")

row = df.iloc[0]
def convert(x):
    return np.stack([np.stack(channel) for channel in x]).astype(np.float32)


lr = convert(row["X_jets_LR"])   # (3, 64, 64)
hr = convert(row["X_jets"])      # (3, 125, 125)

print(lr.shape, hr.shape)

lr_vis = np.transpose(lr, (1,2,0))   # (64,64,3)
hr_vis = np.transpose(hr, (1,2,0))   # (125,125,3)

import matplotlib.pyplot as plt

plt.imshow(lr_vis[:,:,0], cmap='inferno')
plt.title("Low Resolution Jet")
plt.colorbar()
plt.show()

plt.imshow(hr_vis[:,:,0], cmap='inferno')
plt.title("High Resolution Jet")
plt.colorbar()
plt.show()