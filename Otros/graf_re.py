#%%

import random
import matplotlib.pyplot as plt
import seaborn as sns

# Establece el estilo de los gráficos
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

sf_rec = []
sf_masa = []

while len(sf_rec) < 100000:
    masa = round(random.uniform(0.01, 0.95), 2)
    rec = round(random.uniform(0.01, 0.95), 2)
    if masa == 0:  # prevent division by zero
        continue
    if rec / masa > 10:
        sf_rec.append(rec)
        sf_masa.append(masa)

# Crear un gráfico bonito
plt.figure(figsize=(8, 6))
scatter = plt.scatter(sf_masa, sf_rec, c=sf_rec, cmap='viridis', alpha=0.7, s=40)
plt.title("Relación Rec/Masa donde Rec/Masa > 10", fontsize=16, fontweight='bold')
plt.xlabel("Masa", fontsize=14)
plt.ylabel("Rec", fontsize=14)
plt.colorbar(scatter, label="Rec")
sns.despine()
plt.tight_layout()
plt.show()