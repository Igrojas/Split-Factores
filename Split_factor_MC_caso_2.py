#%%
import random
from dataclasses import dataclass
import math
import pandas as pd
from typing import Dict, Tuple


class circuito():
    def __init__(self):
        self.name='padre'
        self.flow_in=[]
        self.flow_out=[]
        self.split_factor=[] # masa, elementos_i

    def calcula(self,flujos):
        print(self.name)

class celda(circuito):
    def __init__(self):
        super().__init__()
        self.volumen_celda=0
        self.factor_celda=0.85

    def calcula(self,flujos):
        #print(self.name)
        flujos[self.flow_in[0]].cuf=flujos[self.flow_in[0]].masa*flujos[self.flow_in[0]].cut/100

        flujos[self.flow_out[0]].masa=flujos[self.flow_in[0]].masa*self.split_factor[0]
        flujos[self.flow_out[1]].masa=flujos[self.flow_in[0]].masa*(1-self.split_factor[0])

        flujos[self.flow_out[0]].cuf=flujos[self.flow_in[0]].cuf*self.split_factor[1]
        flujos[self.flow_out[1]].cuf=flujos[self.flow_in[0]].cuf*(1-self.split_factor[1])

        flujos[self.flow_out[0]].cut=flujos[self.flow_out[0]].cuf/flujos[self.flow_out[0]].masa*100
        flujos[self.flow_out[1]].cut=flujos[self.flow_out[1]].cuf/flujos[self.flow_out[1]].masa*100

class suma(circuito):
    def __init__(self):
        # self.flow_in=[]
        # self.flow_out=[]
        super().__init__()

    def calcula(self,flujos):
        #print(self.name)
        flujos[self.flow_out[0]].masa=0
        flujos[self.flow_out[0]].cuf=0

        for flow in self.flow_in:
            #print(flow)
            flujos[self.flow_out[0]].masa += flujos[flow].masa
            flujos[self.flow_out[0]].cuf += flujos[flow].cuf

        flujos[self.flow_out[0]].cut = flujos[self.flow_out[0]].cuf/flujos[self.flow_out[0]].masa*100

class flujo():
    def __init__(self,id):
        self.name='sn'
        self.flujo=id
        self.masa=0
        self.cut=0 #Esto es la Ley de Cobre
        self.cuf=0
        self.cp=0
        self.ros=1
        self.caudal=0

# =============================================================================


def flujos_globales(lista_equipos):

    fin = set()
    fout = set()

    for eq in lista_equipos.items():
        fin.update(eq[1].flow_in)
        fout.update(eq[1].flow_out)

    print(fin)
    print(fout)

    flujos_entrada = fin - fout
    flujos_salida   = fout - fin
    flujos_salida_conc = flujos_salida - {11}
    flujos_internos = fin & fout

    return flujos_entrada, flujos_salida, flujos_salida_conc, flujos_internos

# =============================================================================



df = pd.read_excel('Simulaciones_v2.xlsx')
df = df[df["Simulacion"] == 2]

num_simulaciones=df["Simulacion"].nunique()
print(f"Total de simulaciones: {num_simulaciones}")

lista_equipos={}
flujos={}
for sim in range(1, num_simulaciones+1):
    print(f"{'='*50}")
    print(f"Simulación: {sim}")
    df_sim = df[df["Simulacion"] == 2]

    lista_equipos[sim]={}
    for idx, row in df_sim.iterrows():
        # print(row["tipo"])

        if row["tipo"] == "celda":
            equipo=celda()
        elif row["tipo"] == "suma":
            equipo=suma()

        # Solo considerar las columnas después de 'sp cuf'
        columnas_flujos = list(df.columns)
        idx_sp_cuf = columnas_flujos.index("sp cuf")
        columnas_flujos = columnas_flujos[idx_sp_cuf+1:]

        # Para los flujos de entrada (=1), nómbralos como 'Alim equipo'
        equipo.flow_in = []
        for col in columnas_flujos:
            if row[col] == 1:
                # Asumimos que nombres tipo "Alim Equipo"
                flujos[col] = flujo(col)
                flujos[col].name = f"Alim {row['Equipo']}"
                equipo.flow_in.append(col)
        # Para los flujos de salida (= -1), el primero es Conc, el segundo es Rel
        equipo.flow_out = []
        out_count = 0
        for col in columnas_flujos:
            if row[col] == -1:
                flujos[col] = flujo(col)
                if out_count == 0:
                    flujos[col].name = f"Conc {row['Equipo']}"
                elif out_count == 1:
                    flujos[col].name = f"Rel {row['Equipo']}"
                equipo.flow_out.append(col)
                out_count += 1
        
        # Aca agregamos todos los datos, nombre, flujos, split factor.
        equipo.name = row["Equipo"]
        equipo.split_factor = [row["sp masa"], row["sp cuf"]]
        lista_equipos[sim][row["Equipo"]] = equipo

print(f"{'='*50}")
# print(lista_equipos)
for sim in lista_equipos:
    print(f"{'='*50}")
    print(f"Simulación: {sim}")
    print(f"{'='*50}")
    for equipo in lista_equipos[sim].items():

        # print(f"Equipo: {equipo[0]}")
        print(f"Nombre: {equipo[1].name}")
        print(f"Split Factor: {equipo[1].split_factor}")
        print(f"Flow In: {equipo[1].flow_in}")
        print(f"Flow Out: {equipo[1].flow_out}")

fe, fs, fs_conc, fi = flujos_globales(lista_equipos[1])

print(fs_conc)
#%%
import pandas as pd
import openpyxl

flujos[4].name='Alim. 1ra Limpieza'
flujos[4].masa= 24.515
flujos[4].cut= 3.63
Max_iter= 100
#%%

iter_mc = 100000

# Crear lista para almacenar resultados de cada iteración
resultados = []

for i in range(iter_mc):
    for sim in lista_equipos:

        sp_jms_1    =  [round(random.uniform(0.01, 0.95), 2), round(random.uniform(0.01, 0.95), 2)]
        sp_scv    =  [round(random.uniform(0.01, 0.95), 2), round(random.uniform(0.01, 0.95), 2)]
        sp_cl_scv =  [round(random.uniform(0.01, 0.95), 2), round(random.uniform(0.01, 0.95), 2)]

        lista_equipos[1]["Jameson 1"].split_factor = sp_jms_1
        lista_equipos[1]["Scavenger"].split_factor = sp_scv
        lista_equipos[1]["1ra Limp. Scv."].split_factor = sp_cl_scv

        # Resolvemos el circuito iterativamente por simulación
        for iter in range(Max_iter):
            for nombre, equipo in lista_equipos[sim].items():
                equipo.calcula(flujos)

    Recuperacion = sum([flujos[i].cuf for i in fs_conc]) / flujos[4].cuf * 100
    MassPull = sum([flujos[i].masa for i in fs_conc]) / flujos[4].masa * 100
    RazonEnriquecimiento = Recuperacion / MassPull
    Ley_Conc_Final = sum([flujos[i].cuf for i in fs_conc]) / sum([flujos[i].masa for i in fs_conc])*100

    # Guardamos split factors y resultados en cada fila
    fila = {
        'sp_Jameson_1_masa': sp_jms_1[0],
        'sp_Jameson_1_cuf': sp_jms_1[1],
        'sp_Scavenger_masa': sp_scv[0],
        'sp_Scavenger_cuf': sp_scv[1],
        'sp_Cl_Scavenger_masa': sp_cl_scv[0],
        'sp_Cl_Scavenger_cuf': sp_cl_scv[1],
        'er_jameson_1': sp_jms_1[1]/sp_jms_1[0],
        'er_scavenger': sp_scv[1]/sp_scv[0],
        'er_cl_scavenger': sp_cl_scv[1]/sp_cl_scv[0],
        'Recuperacion': Recuperacion,
        'MassPull': MassPull,
        'RazonEnriquecimiento': RazonEnriquecimiento,
        'Ley_Conc_Final': Ley_Conc_Final,
        # Generar automáticamente los resultados para todos los flujos presentes
        **{f'Flujo {k} Masa': v.masa for k,v in flujos.items()},
        **{f'Flujo {k} Cut': v.cut for k,v in flujos.items()},

    }
    resultados.append(fila)

# Convertimos la lista de resultados a un DataFrame de pandas
df_resultados = pd.DataFrame(resultados)

# Si quieres guardar el DataFrame a un Excel:
import os

# Crear la carpeta 'results' si no existe
os.makedirs('results', exist_ok=True)
df_resultados.to_excel(os.path.join('results', 'results_caso_2.xlsx'), index=False)

# Si quieres ver los primeros resultados:
display(df_resultados.head())

#%%
import seaborn as sns
import matplotlib.pyplot as plt

def graficar_resultados(df, rec_min=None, rec_max=None, ley_min=None, ley_max=None):

    df_filtrado = df.copy()
    
    # Filtro FIJO: todos 'Flujo X Cut' < 30
    cut_cols = [col for col in df_filtrado.columns if col.startswith('Flujo') and col.endswith('Cut')]
    for col in cut_cols:
        df_filtrado = df_filtrado[df_filtrado[col] < 30]
    
    # También filtra todas las er (razón de enriquecimiento) mayores a 7
    df_filtrado = df_filtrado[df_filtrado['er_jameson_1'] <= 8]
    df_filtrado = df_filtrado[df_filtrado['er_scavenger'] <= 2]
    df_filtrado = df_filtrado[df_filtrado['er_cl_scavenger'] <= 4.8]
    # Filtros variables
    if rec_min is not None:
        df_filtrado = df_filtrado[df_filtrado['Recuperacion'] >= rec_min]
    if rec_max is not None:
        df_filtrado = df_filtrado[df_filtrado['Recuperacion'] <= rec_max]
    if ley_min is not None:
        df_filtrado = df_filtrado[df_filtrado['Ley_Conc_Final'] >= ley_min]
    if ley_max is not None:
        df_filtrado = df_filtrado[df_filtrado['Ley_Conc_Final'] <= ley_max]

    # Ya no es subplot. Creamos una sola figura y eje.
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = sns.scatterplot(
        x='Recuperacion',
        y='Ley_Conc_Final',
        hue='RazonEnriquecimiento',
        size='MassPull',
        palette='viridis',
        data=df_filtrado,
        legend='brief',
        ax=ax
    )
    ax.grid(True,alpha=0.5)
    ax.set_title('Recuperacion vs Ley_Conc_Final')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()

    return df_filtrado

# Ejemplo de uso:
df_filtrado = graficar_resultados(df_resultados.round(2), rec_min=97, ley_min=10, ley_max=30)
def guardar_resultados_avanzado(df_filtrado, archivo='results/results_caso_2_filtered.xlsx'):
    """
    Guarda el DataFrame filtrado en un archivo Excel.
    Además, en una segunda hoja, agrega las primeras concentraciones más altas,
    ordenadas con las columnas específicas:
    Recuperación, MassPull, Razón Enriquicimiento, Ley de Conc. Final,
    Flujo 11 Masa, Flujo 11 Cut
    """
    import pandas as pd

    # Guardar df_filtrado en la primera hoja
    with pd.ExcelWriter(archivo) as writer:
        df_filtrado.to_excel(writer, index=False, sheet_name='Resultados Filtrados')

        # Obtener los N resultados con las concentraciones finales más altas. 
        # Por defecto, toma los 10 más altos o menos si hay menos filas.
        top_n = min(10, len(df_filtrado))
        df_top = df_filtrado.nlargest(top_n, 'Ley_Conc_Final')

        # Columnas requeridas
        columnas = [
            'Recuperacion',
            'MassPull',
            'RazonEnriquecimiento',
            'Ley_Conc_Final',
            'Flujo 11 Masa',
            'Flujo 11 Cut'
        ]
        # Filtrar por las columnas que existen en el dataframe (en caso de que falte alguna)
        columnas_existentes = [col for col in columnas if col in df_top.columns]

        df_resumen = df_top.loc[:, columnas_existentes].copy()
        # Cambiamos los nombres a los pedidos
        renombres = {
            'Recuperacion': 'Recuperación',
            'MassPull': 'MassPull',
            'RazonEnriquicimiento': 'Razón Enriquicimiento',
            'RazonEnriquecimiento': 'Razón Enriquicimiento',
            'Ley_Conc_Final': 'Ley de Conc. Final',
            'Flujo 11 Masa': 'Flujo 11 Masa',
            'Flujo 11 Cut': 'Flujo 11 Cut'
        }
        df_resumen.rename(columns=renombres, inplace=True)
        df_resumen.to_excel(writer, index=False, sheet_name='Top Concentraciones')

guardar_resultados_avanzado(df_filtrado)
# %%