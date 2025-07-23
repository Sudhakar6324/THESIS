import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = 'siren_experiments_v1.xlsx'
df = pd.read_excel(file_path)

# Filter for batch size = 512 and learning rate = 5e-5
fdf = df[(df['batch_size'] == 512) & (df['learning_rate'] == 5e-05)]

# 1) PSNR vs Number of Neurons (fixed n_layers_total)
plt.figure(figsize=(8, 6))
for layers in sorted(fdf['n_layers_total'].unique()):
    subset = fdf[fdf['n_layers_total'] == layers].sort_values('n_neurons')
    plt.plot(subset['n_neurons'], subset['avg_psnr'], marker='o', label=f'n_layers_total={layers}')
plt.xlabel('Number of Neurons')
plt.ylabel('Average PSNR')
plt.title('PSNR vs Number of Neurons (batch=512, lr=5e-5)')
plt.legend()
plt.grid(True)
plt.savefig("plot_psnr_vs_n_neurons.png", dpi=300)

# 2) PSNR vs Number of Layers (fixed n_neurons)
plt.figure(figsize=(8, 6))
for neurons in sorted(fdf['n_neurons'].unique()):
    subset = fdf[fdf['n_neurons'] == neurons].sort_values('n_layers_total')
    plt.plot(subset['n_layers_total'], subset['avg_psnr'], marker='o', label=f'n_neurons={neurons}')
plt.xlabel('Number of Layers')
plt.ylabel('Average PSNR')
plt.title('PSNR vs Number of Layers (batch=512, lr=5e-5)')
plt.legend()
plt.grid(True)
plt.savefig("plot_psnr_vs_n_layers.png", dpi=300)

# 3) PSNR vs Disk Space
plt.figure(figsize=(8, 6))
subset = fdf.sort_values('Disk Space')
plt.plot(subset['Disk Space'], subset['avg_psnr'], marker='o')
# Annotate initial size
initial_size=10.49
plt.annotate(
    f'Initial file size: {initial_size:.2f} MB',
    xy=(0.05, 0.95),
    xycoords='axes fraction',
    fontsize=12,
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8),
    verticalalignment='top'
)
plt.xlabel('Disk Space')
plt.ylabel('Average PSNR')
plt.title('PSNR vs Disk Space (batch=512, lr=5e-5)')
plt.grid(True)
plt.savefig("plot_psnr_vs_diskspace.png", dpi=300)

# Show all plots
plt.tight_layout()
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. Load your data (Excel or CSV)
# df = pd.read_excel("siren_experiments_v1.xlsx")   # If your data is in Excel
# # df = pd.read_csv("experiment_results.csv")      # If your data is in CSV

# # -----------------------------------------------------------------------------------
# # 2. Plot: PSNR vs. n_neurons
# # -----------------------------------------------------------------------------------
# # Group by n_neurons, compute mean PSNR
# group_n_neurons = df.groupby("n_neurons")["avg_psnr"].mean().reset_index()

# plt.figure(figsize=(8, 6))
# plt.plot(group_n_neurons["n_neurons"], group_n_neurons["avg_psnr"], 
#          marker='o', linewidth=2, markersize=6)
# plt.title("PSNR vs. n_neurons", fontsize=14, pad=15)
# plt.xlabel("Number of Neurons (n_neurons)", fontsize=12)
# plt.ylabel("PSNR", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True, which='both', ls='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("plot_n_neurons.png", dpi=300)  # Save the figure
# plt.show()

# # -----------------------------------------------------------------------------------
# # 3. Plot: PSNR vs. n_layers_total
# # -----------------------------------------------------------------------------------
# group_n_layers = df.groupby("n_layers_total")["avg_psnr"].mean().reset_index()

# plt.figure(figsize=(8, 6))
# plt.plot(group_n_layers["n_layers_total"], group_n_layers["avg_psnr"], 
#          marker='o', linewidth=2, markersize=6)
# plt.title("PSNR vs. n_layers_total", fontsize=14, pad=15)
# plt.xlabel("Total Layers (n_layers_total)", fontsize=12)
# plt.ylabel("PSNR", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True, which='both', ls='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("plot_n_layers_total.png", dpi=300)
# plt.show()

# # -----------------------------------------------------------------------------------
# # 4. Plot: PSNR vs. batch_size
# # -----------------------------------------------------------------------------------
# group_batch = df.groupby("batch_size")["avg_psnr"].mean().reset_index()

# plt.figure(figsize=(8, 6))
# plt.plot(group_batch["batch_size"], group_batch["avg_psnr"], 
#          marker='o', linewidth=2, markersize=6)
# plt.title("PSNR vs. batch_size", fontsize=14, pad=15)
# plt.xlabel("Batch Size", fontsize=12)
# plt.ylabel("PSNR", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True, which='both', ls='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("plot_batch_size.png", dpi=300)
# plt.show()

# # -----------------------------------------------------------------------------------
# # 5. Plot: PSNR vs. learning_rate
# # -----------------------------------------------------------------------------------
# group_lr = df.groupby("learning_rate")["avg_psnr"].mean().reset_index()

# plt.figure(figsize=(8, 6))
# plt.plot(group_lr["learning_rate"], group_lr["avg_psnr"], 
#          marker='o', linewidth=2, markersize=6)
# plt.title("PSNR vs. learning_rate", fontsize=14, pad=15)
# plt.xlabel("Learning Rate", fontsize=12)
# plt.ylabel("PSNR", fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True, which='both', ls='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("plot_learning_rate.png", dpi=300)
# plt.show()