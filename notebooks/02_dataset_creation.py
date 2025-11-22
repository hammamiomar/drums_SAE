# %%
import os

import pandas as pd

from drums_SAE import dataset_prepare

os.chdir("..")
# %% Create Dataset Manifest
df = dataset_prepare.create_dataset_manifest("data")
df.head()
# %%
df.to_csv("data/dataset_manifest.csv", index=True)
