import pandas as pd
import torch
import numpy as np
import os

def convert(x):
    return np.stack([np.stack(channel) for channel in x]).astype(np.float32)

def process_file(parquet_path, save_path):
    df = pd.read_parquet(parquet_path)

    lr_list = []
    hr_list = []
    y_list = []

    for _, row in df.iterrows():
        lr = convert(row["X_jets_LR"])   # (3,64,64)
        hr = convert(row["X_jets"])      # (3,125,125)

        lr_list.append(lr)
        hr_list.append(hr)
        y_list.append(row["y"])

    # Convert to tensors
    lr_tensor = torch.tensor(np.array(lr_list))   # (N,3,64,64)
    hr_tensor = torch.tensor(np.array(hr_list))   # (N,3,125,125)
    y_tensor  = torch.tensor(y_list)

    torch.save({
        "lr": lr_tensor,
        "hr": hr_tensor,
        "y": y_tensor
    }, save_path)

    print(f"Saved: {save_path}")

parquet_files = [
    "F:\\jet0run0\\run_0_chunk_0.parquet", "F:\\jet0run0\\run_0_chunk_1.parquet","F:\\jet0run0\\run_0_chunk_2.parquet","F:\\jet0run0\\run_0_chunk_3.parquet",
    "F:\\jet0run0\\run_1_chunk_0.parquet","F:\\jet0run0\\run_1_chunk_1.parquet","F:\\jet0run0\\run_1_chunk_2.parquet","F:\\jet0run0\\run_1_chunk_3.parquet",
    "F:\\jet0run0\\run_1_chunk_4.parquet","F:\\jet0run0\\run_2_chunk_0.parquet","F:\\jet0run0\\run_2_chunk_1.parquet","F:\\jet0run0\\run_2_chunk_2.parquet",
    "F:\\jet0run0\\run_2_chunk_3.parquet","F:\\jet0run0\\run_2_chunk_4.parquet","F:\\jet0run0\\run_2_chunk_5.parquet"
]

for file in parquet_files:
    save_name = file.replace(".parquet", ".pt")
    process_file(file, save_name)