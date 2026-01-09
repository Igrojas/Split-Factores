#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def graficar_curvas_ley_recuperacion(
        archivo_excel='Test Dilucion.xlsx',
        hoja=0,
        col_id='id',
        col_ley='Ley acumulada, Cu%',
        col_rec='Recuperación, Cu%',
        col_ley_unica='Ley Cu',
        nombre_figura='curvas_ley_recuperacion.png',
        mostrar_columnas=True
    ):
    """
    Lee un archivo Excel con datos de ley y recuperación, y grafica las curvas Ley-Recuperación para cada id.

    Returns:
        df: DataFrame con los datos leídos.
    """
    df = pd.read_excel(archivo_excel, sheet_name=hoja)
    if mostrar_columnas:
        print("Columnas en el archivo:", df.columns.tolist())
    print(f"\nRegistros cargados: {len(df)}")
    print(f"IDs únicos: {df[col_id].unique()}")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

    for i, id_val in enumerate(sorted(df[col_id].unique())):
        subset = df[df[col_id] == id_val].sort_values(by=col_rec)
        subset_no_100 = subset[subset[col_rec] != 100]

        # Ley (si existe)
        if col_ley_unica in subset.columns:
            unique_ley = subset[col_ley_unica].iloc[0]
            ley_str = f"{unique_ley:.2f}"
        else:
            ley_str = "?"

        # Graficar datos reales
        ax.plot(
            subset_no_100[col_rec], subset_no_100[col_ley],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2, markersize=6,
            label=f'ID {id_val} (Ley={ley_str}%)'
        )

    ax.set_xlabel('Recuperación Cu (%)', fontsize=12)
    ax.set_ylabel('Ley Acumulada Cu (%)', fontsize=12)
    ax.set_title('Curvas Ley-Recuperación', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(nombre_figura, dpi=150, bbox_inches='tight')
    plt.show()

    return df

# Para usar la función:
df = graficar_curvas_ley_recuperacion()

# %%
