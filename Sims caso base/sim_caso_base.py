#%%
import random
from dataclasses import dataclass
import math
import pandas as pd
from typing import Dict, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy


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
    flujos_salida_conc = flujos_salida - {9}
    flujos_internos = fin & fout

    return flujos_entrada, flujos_salida, flujos_salida_conc, flujos_internos


import numpy as np

def cargar_datos_equipos(path_excel, sheet_name=None):
    """
    Carga los equipos y flujos desde el archivo de configuración.
    
    Args:
        path_excel: Ruta al archivo Excel
        sheet_name: Nombre de la hoja a leer. Si es None, lee la hoja principal (por defecto)
    
    Retorna:
        lista_equipos: diccionario con la info de cada equipo por simulación
        flujos: diccionario global de flujos (mismo para todas las simulaciones)
    """
    if sheet_name is None:
        df = pd.read_excel(path_excel)
        print(f"Cargando datos desde hoja principal de: {path_excel}")
    else:
        df = pd.read_excel(path_excel, sheet_name=sheet_name)
        print(f"Cargando datos desde hoja '{sheet_name}' de: {path_excel}")
    num_simulaciones = df["Simulacion"].nunique()
    print(f"Total de simulaciones: {num_simulaciones}")

    lista_equipos = {}
    flujos = {}
    for sim in range(1, num_simulaciones + 1):
        print(f"{'='*50}")
        print(f"Simulación: {sim}")
        df_sim = df[df["Simulacion"] == sim]

        lista_equipos[sim] = {}
        for idx, row in df_sim.iterrows():

            # Crear equipo según tipo
            if row["tipo"] == "celda":
                equipo = celda()
            elif row["tipo"] == "suma":
                equipo = suma()

            # Definir flujos de entrada y salida
            columnas_flujos = list(df.columns)
            idx_sp_cuf = columnas_flujos.index("sp cuf")
            columnas_flujos = columnas_flujos[idx_sp_cuf + 1 :]

            equipo.flow_in = []
            for col in columnas_flujos:
                if row[col] == 1:
                    flujos[col] = flujo(col)
                    flujos[col].name = f"Alim {row['Equipo']}"
                    equipo.flow_in.append(col)

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

            equipo.name = row["Equipo"]
            equipo.split_factor = [row["sp masa"], row["sp cuf"]]
            lista_equipos[sim][row["Equipo"]] = equipo

    print(f"{'='*50}")
    for sim in lista_equipos:
        print(f"{'='*50}")
        print(f"Simulación: {sim}")
        print(f"{'='*50}")
        for equipo in lista_equipos[sim].items():
            print(f"Nombre: {equipo[1].name}")
            print(f"Split Factor: {equipo[1].split_factor}")
            print(f"Flow In: {equipo[1].flow_in}")
            print(f"Flow Out: {equipo[1].flow_out}")

    return lista_equipos, flujos

def correr_simulacion_normal(lista_equipos, flujos):
    """
    Corre la simulación normal con los equipos y flujos entregados,
    guarda los resultados y los retorna como dataframe.
    """
    # Identificar flujos globales internamente
    fe, fs, fs_conc, fi = flujos_globales(lista_equipos[1])
    print(fs_conc)

    # (Ajusta aquí la semilla inicial "manual" de masivo/flujos según corresponda a tus datos)
    flujos[4].name = 'Alim. 1ra Limpieza'
    flujos[4].masa = 24.515
    flujos[4].cut = 2.40
    iter_sim = 1

    resultados = []
    for i in range(iter_sim):
        for sim in lista_equipos:
            for nombre, equipo in lista_equipos[sim].items():
                equipo.calcula(flujos)

            Recuperacion = sum([flujos[i].cuf for i in fs_conc]) / flujos[4].cuf * 100
            MassPull = sum([flujos[i].masa for i in fs_conc]) / flujos[4].masa * 100
            RazonEnriquecimiento = Recuperacion / MassPull if MassPull != 0 else 0
            Ley_Conc_Final = (
                sum([flujos[i].cuf for i in fs_conc]) / sum([flujos[i].masa for i in fs_conc]) * 100 
                if sum([flujos[i].masa for i in fs_conc]) != 0 else 0
            )

            fila = {
                'Simulacion': sim,
                'Recuperacion': Recuperacion,
                'MassPull': MassPull,
                'RazonEnriquecimiento': RazonEnriquecimiento,
                'Ley_Conc_Final': Ley_Conc_Final,
                **{f'Flujo {k} Masa': v.masa for k, v in flujos.items()},
                **{f'Flujo {k} Cut': v.cut for k, v in flujos.items()},
            }
            resultados.append(fila)

    df_resultados = pd.DataFrame(resultados)
    return df_resultados

def correr_simulacion_montecarlo(
    archivo_base,
    equipos_objetivo=None,
    semilla_inicial_flujos=None,
    n_sim=1_000_000,
    sheet_name=None
):
    """
    Ejecuta una simulación Monte Carlo para CADA 'simulación' encontrada en la hoja base.
    
    Para cada simulación:
        - Se cargan los equipos y flujos UNA VEZ (fuera del loop MC)
        - Para cada iteración de Monte Carlo:
            - Se hace copia profunda de equipos y flujos
            - Se sortean split_factor aleatorios SOLO para equipos_objetivo
            - Se calculan los flujos/resultados
        - Se guardan los resultados en Excel separados por hoja según simulación
    
    Args:
        archivo_base: Ruta al archivo Excel con las simulaciones
        equipos_objetivo: Lista de nombres de equipos a modificar con split factors aleatorios
        semilla_inicial_flujos: Dict con valores iniciales para flujos {id_flujo: {'masa': val, 'cut': val}}
        n_sim: Número de iteraciones Monte Carlo
        sheet_name: Nombre de la hoja Excel a leer. Si es None, lee la hoja principal (por defecto)
    
    Returns:
        Dict donde cada llave es el id de simulación y el valor es un DataFrame con resultados MC
    """
    import tqdm
    import os
    
    # Cargar equipos y flujos UNA VEZ para todas las simulaciones
    lista_equipos_base, flujos_base = cargar_datos_equipos(archivo_base, sheet_name=sheet_name)
    simulaciones_ids = list(lista_equipos_base.keys())
    
    # Prepara dict salida
    results_por_sim = {}

    if equipos_objetivo is None:
        equipos_objetivo = ["Jameson 1"]  # Valor por defecto

    # Procesar cada simulación
    for sim_id in simulaciones_ids:
        print(f"\n{'='*60}")
        print(f"Iniciando Monte Carlo para Simulación {sim_id}")
        print(f"Equipos objetivo: {equipos_objetivo}")
        print(f"Iteraciones: {n_sim:,}")
        print(f"{'='*60}")
        
        resultados_mc = []
        
        # Obtener equipos base para esta simulación (fuera del loop MC)
        equipos_base = lista_equipos_base[sim_id]
        
        # Calcular flujos globales una vez (no cambian entre iteraciones MC)
        fe, fs, fs_conc, fi = flujos_globales(equipos_base)
        
        # Loop de Monte Carlo
        for sim_number in tqdm.trange(n_sim, desc=f"MC Sim {sim_id}", leave=False):
            try:
                # Hacer copia profunda de equipos y flujos para esta iteración
                equipos = copy.deepcopy(equipos_base)
                flujos = copy.deepcopy(flujos_base)
                
                # Aplicar semilla inicial si corresponde
                if semilla_inicial_flujos is not None:
                    for k, vals in semilla_inicial_flujos.items():
                        if k in flujos:
                            flujos[k].masa = vals['masa']
                            flujos[k].cut = vals['cut']
                
                # Modificar split factors SOLO para equipos_objetivo
                for eq_name, eq in equipos.items():
                    if eq.name in equipos_objetivo:
                        s1 = np.random.uniform(0.1, 0.85)
                        s2 = np.random.uniform(0.1, 0.85)
                        eq.split_factor = [s1, s2]
                
                # Calcular todos los equipos de esta simulación
                for nombre, equipo in equipos.items():
                    equipo.calcula(flujos)
                
                # Calcular resultados principales
                try:
                    # Verificar que el flujo base existe
                    if 4 not in flujos or flujos[4].cuf == 0:
                        continue
                    
                    Recuperacion = sum([flujos[i].cuf for i in fs_conc]) / flujos[4].cuf * 100
                    MassPull = sum([flujos[i].masa for i in fs_conc]) / flujos[4].masa * 100
                    RazonEnriquecimiento = Recuperacion / MassPull if MassPull != 0 else 0
                    Ley_Conc_Final = (
                        sum([flujos[i].cuf for i in fs_conc]) / sum([flujos[i].masa for i in fs_conc]) * 100
                        if sum([flujos[i].masa for i in fs_conc]) != 0 else 0
                    )
                except (ZeroDivisionError, KeyError):
                    continue
                
                # Recopilar información de split factors para equipos objetivo
                split_info = {}
                for eq_name, eq in equipos.items():
                    if eq.name in equipos_objetivo:
                        split_info[f"{eq.name}_split_masa"] = eq.split_factor[0]
                        split_info[f"{eq.name}_split_cuf"] = eq.split_factor[1]
                
                # Crear fila de resultados
                fila = {
                    'MonteCarlo_Iter': sim_number,
                    'Recuperacion': Recuperacion,
                    'MassPull': MassPull,
                    'RazonEnriquecimiento': RazonEnriquecimiento,
                    'Ley_Conc_Final': Ley_Conc_Final,
                    **split_info,
                    **{f'Flujo {k} Masa': v.masa for k, v in flujos.items()},
                    **{f'Flujo {k} Cut': v.cut for k, v in flujos.items()},
                }
                resultados_mc.append(fila)
                
            except Exception as e:
                # Silenciar errores para no interrumpir el proceso
                continue
        
        # Crear DataFrame con resultados de esta simulación
        df_resultados_mc = pd.DataFrame(resultados_mc)
        results_por_sim[sim_id] = df_resultados_mc
        
        print(f"Simulación {sim_id} completada: {len(resultados_mc):,} iteraciones exitosas")

    # Guardar resultados en Excel separados por hoja según simulación
    os.makedirs('results', exist_ok=True)
    archivo_salida = os.path.join('results', 'results_montecarlo.xlsx')
    
    with pd.ExcelWriter(archivo_salida) as writer:
        for sim_id, df in results_por_sim.items():
            df.to_excel(writer, sheet_name=f'Simulacion_{sim_id}', index=False)
    
    print(f"\n{'='*60}")
    print(f"Resultados guardados en: {archivo_salida}")
    print(f"{'='*60}")

    return results_por_sim

# === MAIN ===
def main():
    """
    Ejecuta la simulación y retorna todas las salidas relevantes:
    - lista_equipos: diccionario de equipos para cada simulación
    - flujos: diccionario de flujos
    - df_resultados: DataFrame con resultados principales de simulación normal
    - df_resultados_mc: Diccionario con resultados del Monte Carlo por simulación {sim_id: DataFrame}
    """
    # 1. Cargar equipos y flujos desde archivo base
    lista_equipos, flujos = cargar_datos_equipos('../Simulacion_caso_base.xlsx')

    # 2. Simulación "normal"
    df_resultados = correr_simulacion_normal(lista_equipos, flujos)

    # 3. Simulación Monte Carlo (1e6 ejecuciones, split_factor aleatorio para equipos deseados)
    archivo_base = '../Simulacion_caso_base.xlsx'
    semilla = {4: {'masa': 24.515, 'cut': 2.40}}
    equipos_a_cambiar = ["Jameson 1"] # Modificar según el problema
    hoja_mc = "Sim MC"  # Nombre de la hoja para Monte Carlo (None para usar hoja principal)
    df_resultados_mc = correr_simulacion_montecarlo(
        archivo_base=archivo_base,
        equipos_objetivo=equipos_a_cambiar,
        semilla_inicial_flujos=semilla,
        n_sim=1000000,
        sheet_name=hoja_mc
    )

    # 4. Guardar resultados normales
    import os
    os.makedirs('results', exist_ok=True)
    with pd.ExcelWriter(os.path.join('results', 'results_caso_4.xlsx')) as writer:
        for sim_id in df_resultados['Simulacion'].unique():
            df_sim = df_resultados[df_resultados['Simulacion'] == sim_id]
            df_sim.to_excel(writer, sheet_name=f'Simulacion_{sim_id}', index=False)
    
    # Los resultados de Monte Carlo ya se guardan dentro de correr_simulacion_montecarlo
    # No es necesario guardarlos aquí nuevamente

    # Mostrar resultados principales (opcional si usas Jupyter)
    try:
        from IPython.display import display
        display(df_resultados.head())
        if df_resultados_mc is not None:
            # df_resultados_mc es un diccionario, mostrar el primer DataFrame
            if df_resultados_mc:
                first_sim_id = list(df_resultados_mc.keys())[0]
                display(df_resultados_mc[first_sim_id].head())
    except ImportError:
        print(df_resultados.head())
        if df_resultados_mc is not None:
            # df_resultados_mc es un diccionario, mostrar el primer DataFrame
            if df_resultados_mc:
                first_sim_id = list(df_resultados_mc.keys())[0]
                print(f"\nResultados MC - Simulación {first_sim_id}:")
                print(df_resultados_mc[first_sim_id].head())

    # Retornar todas las salidas solicitadas
    return lista_equipos, flujos, df_resultados, df_resultados_mc
    
# Ejecutar si es programa principal

if __name__ == "__main__":
    lista_equipos, flujos, df_resultados, df_resultados_mc = main()

#%%
def graficar_resultados(df, rec_min=None, rec_max=None, ley_min=None, ley_max=None, solo_flujo9_cut=False):
    """
    Filtra el DataFrame según los parámetros ingresados. 
    Si solo_flujo9_cut=True, sólo filtra por 'Flujo 9 Cut' > 0.2 y < 30 y no por las otras columnas de cut.
    """
    df_filtrado = df.copy()
    
    df_filtrado = df_filtrado[df_filtrado['Flujo 10 Cut'] < 30]
    

    # Filtro FIJO: todos 'Flujo X Cut' < 30
    cut_cols = [col for col in df_filtrado.columns if col.startswith('Flujo') and col.endswith('Cut')]
    for col in cut_cols:
        df_filtrado = df_filtrado[df_filtrado[col] < 27]

    # Filtros variables
    if rec_min is not None:
        df_filtrado = df_filtrado[df_filtrado['Recuperacion'] >= rec_min]
    if rec_max is not None:
        df_filtrado = df_filtrado[df_filtrado['Recuperacion'] <= rec_max]
    if ley_min is not None:
        df_filtrado = df_filtrado[df_filtrado['Ley_Conc_Final'] >= ley_min]
    if ley_max is not None:
        df_filtrado = df_filtrado[df_filtrado['Ley_Conc_Final'] <= ley_max]

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

# Ejemplo de uso con los filtros originales:
df_filtrado = graficar_resultados(df_resultados.round(2), rec_min=90, ley_min=1, ley_max=30)
# Ejemplo de uso SOLO filtrando por Flujo 9 Cut:

def guardar_resultados_avanzado(df_filtrado, archivo='results/results_caso_4_filtered.xlsx'):
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
            'Flujo 9 Masa',
            'Flujo 9 Cut'
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
            'Flujo 9 Masa': 'Flujo 9 Masa',
            'Flujo 9 Cut': 'Flujo 9 Cut'
        }
        df_resumen.rename(columns=renombres, inplace=True)
        df_resumen.to_excel(writer, index=False, sheet_name='Top Concentraciones')

guardar_resultados_avanzado(df_filtrado)
# %%

file_test_dil = '../Test Dilucion.xlsx'
df_test_dil = pd.read_excel(file_test_dil)

# Solo id 5 para test de dilución
df_test_id5 = df_test_dil[df_test_dil['id'] == 5]

# Elimina fila(s) donde la recuperación es 100 (asumiendo columna Recuperación, Cu%)
if 'Recuperación, Cu%' in df_test_id5.columns:
    df_test_id5 = df_test_id5[df_test_id5['Recuperación, Cu%'] != 100]
# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(8,6))

# Graficar TODOS los puntos en una sola gráfica:
# 1. Resultados de simulación (nube Monte Carlo) con etiqueta según columna 'Simulacion'
for idx, row in df_resultados.iloc[:2].iterrows():
    x = int(round(row['Recuperacion']))
    y = int(round(row['Ley_Conc_Final']))
    label = f"Sim {row['Simulacion']}"  # Etiqueta para cada punto
    ax.scatter(
        x, 
        y,
        color='black', 
        alpha=1, 
        edgecolor='w', 
        s=500,
        marker='*',
        #label=label
    )
    # Mostramos la etiqueta un poco más arriba del punto en la gráfica
    ax.text(
        x, 
        y + 1,  # Solo enteros, desplazamiento de 1 entero hacia arriba
        label, 
        fontsize=10, 
        ha='center', 
        va='bottom',  # Anclar la parte inferior del texto para que quede "sobre" el punto
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
    )
# Agrega un punto verde estrella en (94.41, 18.36) convertido a enteros
ax.scatter(
    int(round(94.41)),
    int(round(18.36)),
    color='green',
    edgecolor='w',
    s=500,
    marker='*',
    label="Pilotaje Caso Base"
)

# 2. Resultados de test de dilución (id 5) como línea, pero SOBRE LA MISMA GRÁFICA
# Nos aseguramos de convertir los ejes de df_test_id5 a enteros antes de graficar
df_test_id5_entero = df_test_id5.copy()
df_test_id5_entero['Recuperación, Cu%'] = df_test_id5_entero['Recuperación, Cu%'].round().astype(int)
df_test_id5_entero['Ley acumulada, Cu%'] = df_test_id5_entero['Ley acumulada, Cu%'].round().astype(int)

sns.lineplot(
    x='Recuperación, Cu%', 
    y='Ley acumulada, Cu%', 
    data=df_test_id5_entero,
    color='royalblue',
    alpha=1,
    marker='D',
    markersize=8,
    linestyle='--',
    linewidth=2,
    label=f"Test de Dilución Ley Alimentación = {int(round(df_test_id5['Ley Cu'].iloc[0]))}",
    zorder=10,
    ax=ax
)

# Modificar los ticks de x y y según pauta: x de 5 en 5 hasta 100, y de 2 en 2
# Eje x: de mínimo múltiplo de 5 mayor o igual al límite inferior, hasta 100 de 5 en 5
x_start = max(0, 5 * int(np.floor(ax.get_xlim()[0] / 5)))
x_end = 100
xticks = np.arange(x_start, x_end + 1, 5)
ax.set_xticks(xticks)
ax.set_xlim(x_start, x_end)

# Eje y: de mínimo múltiplo de 2 mayor o igual al límite inferior, hasta el mayor múltiplo de 2 mayor o igual al límite superior
ylim = ax.get_ylim()
y_start = 2 * int(np.floor(ylim[0] / 2))
y_end = 2 * int(np.ceil(ylim[1] / 2))
yticks = np.arange(y_start, y_end + 1, 2)
ax.set_yticks(yticks)
ax.set_ylim(y_start, y_end)

ax.set_xlabel('Recuperación (%)', fontsize=12)
ax.set_ylabel('Ley de Conc. Final (%)', fontsize=12)
ax.set_title('Nube de Simulación y Test de Dilución', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()
#%%