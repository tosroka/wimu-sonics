import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a
sns.set_theme(style='whitegrid', context='talk', palette='muted')
csv_folder = 'results'

all_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# remove not plottable
ignore_keywords = ['EQ', 'no_augment', 'compress_ogg']
files = [f for f in all_files if not any(kw in f for kw in ignore_keywords)]

# Group files by prefix
grouped_files = {}
for f in files:
    match = re.match(r'([a-z_\d]+?)(_[-\d\.a-z]+)?_results\.csv', f)
    print(match)
    if match:
        group = match.group(1)
        param = match.group(2) if match.group(2) else ''
        grouped_files.setdefault(group, []).append((param, f))

for group, param_files in grouped_files.items():
    data = []
    labels = []
    params = []

    # Sort by numeric parameter
    def extract_number(param_str):
        nums = re.findall(r'[-+]?\d*\.?\d+', param_str)
        return float(nums[0]) if nums else 0

    param_files.sort(key=lambda x: extract_number(x[0]))

    for param, filename in param_files:
        df = pd.read_csv(os.path.join(csv_folder, filename))
        mean_vals = df.mean()
        data.append(mean_vals.values)
        labels = df.columns.tolist()
        params.append(param.strip('_'))

    plot_data = pd.DataFrame(data, columns=labels)
    plot_data['param'] = params
    plot_data = plot_data.melt(id_vars='param', var_name='Metric', value_name='Mean Value')

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot_data, x='param', y='Mean Value', hue='Metric', marker='o')
    plt.title(f'{group} effects on results')
    plt.xlabel('Parameter')
    plt.ylabel('Fakeness')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Metric')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.savefig(f'reports/figures/{group}_lineplot.png')
    plt.show()
