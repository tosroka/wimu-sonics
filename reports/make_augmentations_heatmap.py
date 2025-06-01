import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key

# Ustawienia
base_file = "no_augment_results.csv"

blacklist = [base_file] + ["bit_crush_10_results.csv", "vibrato_16_001_results.csv", "speed_120percent_results.csv", "time_mask_15percent_results.csv"]
folder = "./results"
threshold = 0.1  # próg różnicy średnich

# Wczytaj średnie z pliku bazowego
def get_column_means(file_path):
    df = pd.read_csv(file_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.mean(skipna=True)

def cmp_with_numbers(x1: str, x2: str):
    a = x1.split("_")
    b = x2.split("_")

    if a[0]=="mask" and b[0]=="mask":
        if int(a[1][:-2]) < int(b[1][:-2]):
            return 1
        else:
            return -1
    
    if x1<x2:
        return 1
    else:
        return -1


base_means = get_column_means(os.path.join(folder, base_file))

# Zbierz różnice przekraczające threshold
differences = {}

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".csv") and filename not in blacklist:
        file_path = os.path.join(folder, filename)
        means = get_column_means(file_path)

        diffs = {}
        for col in base_means.index:
            base_val = base_means[col]
            current_val = means.get(col, np.nan)
            diff = base_val - current_val
            if abs(diff) > threshold:
                diffs[col] = -diff

        if diffs:
            clean_name = filename.replace("_results.csv", "")
            differences[clean_name] = diffs

# Utwórz ramkę danych z różnicami
all_models = base_means.index.tolist()
all_augments = sorted(differences.keys(), key=cmp_to_key(cmp_with_numbers), reverse=True)



data = []
for model in all_models:
    row = []
    for augment in all_augments:
        row.append(differences.get(augment, {}).get(model, np.nan))
    data.append(row)

df = pd.DataFrame(data, index=all_models, columns=all_augments)

df = df.reindex(index=["real", "suno", "udio", "YuE", "musicgen"])

# Podział augmentacji (kolumn) na dwie połowy
mid = len(all_augments) // 2
augment_top = all_augments[:mid]
augment_bottom = all_augments[mid:]

fig, axes = plt.subplots(2, 1,figsize=(12, 7),sharex=False, sharey=True)
# Górny subplot
im1 = axes[0].imshow(df[augment_top], cmap='RdBu_r', aspect='equal')
axes[0].set_xticks(np.arange(len(augment_top)))
axes[0].set_xticklabels(augment_top, rotation=45, ha='right')
axes[0].set_yticks(np.arange(len(all_models)))
axes[0].set_yticklabels(df.index)
axes[0].set_title("Augmentations (part 1)")

# Dolny subplot
im2 = axes[1].imshow(df[augment_bottom], cmap='RdBu_r', aspect='equal')
axes[1].set_xticks(np.arange(len(augment_bottom)))
axes[1].set_xticklabels(augment_bottom, rotation=45, ha='right')
axes[1].set_yticks(np.arange(len(all_models)))
axes[1].set_yticklabels(df.index)
axes[1].set_title("Augmentations (part 2)")

# Wspólny pasek kolorów (po prawej stronie)
fig.colorbar(im1, ax=axes, label='Impact on predictions (higher is more fake)')

#plt.tight_layout()
plt.show()
plt.savefig("reports/figures/heatmap.png", bbox_inches='tight')