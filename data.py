import pandas as pd
import numpy as np
df = pd.read_parquet("F:\\jet0run0\\run_0_chunk_0.parquet")

row = df.iloc[0]

print(type(row["X_jets_LR"]))
print(type(row["X_jets_LR"][0]))
print(type(row["X_jets_LR"][0][0]))

lr = row["X_jets_LR"]

print("Level 1:", len(lr))             # should be 3
print("Level 2:", len(lr[0]))          # you saw 64
print("Level 3 type:", type(lr[0][0]))

hr = row["X_jets"]
print(type(row["X_jets"]))
print(type(row["X_jets"][0]))
print(type(row["X_jets"][0][0]))

hr = row["X_jets"]

print("Level 1:", len(hr))             # should be 3
print("Level 2:", len(hr[0]))          # you saw 64
print("Level 3 type:", type(hr[0][0]))



print("Final level shape:", np.array(lr[0][0]).shape)
print("Final level shape:", np.array(hr[0][0]).shape)
# print(row)

# print("Label (y):", row["y"])
# print("pt:", row["pt"])
# print("m0:", row["m0"])

# print("\nLR shape:", len(row["X_jets_LR"]), "channels")
# print("LR channel length:", len(row["X_jets_LR"][0]))

# print("\nHR shape:", len(row["X_jets"]), "channels")
# print("HR channel length:", len(row["X_jets"][0]))
# print(df.shape)
# print(df.columns)
# # print(df.head())

# # print(type(df["X_jets_LR"].iloc[0]))
# # print(type(df["X_jets"].iloc[0]))

# # df.head()
# # df.info()
# sample = df.iloc[1]

# lr = np.array(sample["X_jets_LR"])
# hr = np.array(sample["X_jets"])

# print(lr.shape)
# print(type(lr[0]))
# print(np.array(lr[0]).shape)
# print(np.array(lr[1]).shape)
# print(np.array(lr[2]).shape)

# print(hr.shape)
# print(type(hr[0]))
# print(np.array(hr[0]).shape)
# print(np.array(hr[1]).shape)
# print(np.array(hr[2]).shape)

# sample = df.iloc[0]["X_jets"]

# # print(sample)
# print(len(sample[0]))
# lr = sample["X_jets_LR"]
# hr = sample["X_jets"]
# print(lr.shape)
# print(hr.shape)
# print(lr.min(), lr.max())
# print(hr.min(), hr.max())
# print(len(lr))
# print(len(hr))


# import matplotlib.pyplot as plt

# plt.imshow(lr[:,:,0], cmap='inferno')
# plt.title("Low Res Channel 1")
# plt.colorbar()
# plt.show()

# print(lr.min(), lr.max(), lr.mean())

# plt.hist(lr.flatten(), bins=50)
# plt.title("Pixel Value Distribution")
# plt.show()

# print(df["y"].value_counts())

# plt.subplot(1,2,1)
# plt.imshow(lr[:,:,0])
# plt.title("LR")

# plt.subplot(1,2,2)
# plt.imshow(hr[:,:,0])
# plt.title("HR")

# plt.show()
# print(lr.sum(), hr.sum())

# import numpy as np
# np.isnan(lr).sum()
# (lr == 0).all()
# print(lr.max())




