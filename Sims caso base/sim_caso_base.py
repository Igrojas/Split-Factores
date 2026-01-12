#%%
import random
from dataclasses import asdict, dataclass
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


def flujos_globales(lista_equipos,salidas_relave={9}):

    fin = set()
    fout = set()

    for eq in lista_equipos.items():
        fin.update(eq[1].flow_in)
        fout.update(eq[1].flow_out)

    flujos_entrada = fin - fout
    flujos_salida   = fout - fin
    flujos_salida_conc = flujos_salida-salidas_relave
    flujos_internos = fin & fout

    print(f"Flujos entrada: {flujos_entrada}")
    print(f"Flujos salida: {flujos_salida}")
    print(f"Flujos salida conc: {flujos_salida_conc}")
    print(f"Flujos internos: {flujos_internos}")

    return flujos_entrada, flujos_salida, flujos_salida_conc, flujos_internos

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
    flujos[4].masa = 23.84
    flujos[4].cut = 2.5167
    iter_sim = 100

    resultados = []

    for sim in lista_equipos:
        for i in range(iter_sim):
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
            'Error Masa': flujos[4].masa - sum(flujos[i].masa for i in fs),
            'Error CuF': flujos[4].cuf - sum(flujos[i].cuf for i in fs),
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
    sheet_name=None,
    ley_conc_final_min=None,
    ley_conc_final_max=None,
    nombre_archivo_salida=None
):
    """
    Ejecuta una simulación Monte Carlo para CADA 'simulación' encontrada en la hoja base.
    
    Para cada simulación:
        - Se cargan los equipos y flujos UNA VEZ (fuera del loop MC)
        - Para cada iteración de Monte Carlo:
            - Se hace copia profunda de equipos y flujos
            - Se sortean split_factor aleatorios SOLO para equipos_objetivo
            - Se calculan los flujos/resultados
        - Se filtran los resultados según Ley_Conc_Final (si se especifican filtros)
        - Se guardan los resultados en Excel separados por hoja según simulación
    
    Args:
        archivo_base: Ruta al archivo Excel con las simulaciones
        equipos_objetivo: Lista de nombres de equipos a modificar con split factors aleatorios
        semilla_inicial_flujos: Dict con valores iniciales para flujos {id_flujo: {'masa': val, 'cut': val}}
        n_sim: Número de iteraciones Monte Carlo
        sheet_name: Nombre de la hoja Excel a leer. Si es None, lee la hoja principal (por defecto)
        ley_conc_final_min: Valor mínimo para filtrar por Ley_Conc_Final (inclusive). Si es None, no filtra por mínimo
        ley_conc_final_max: Valor máximo para filtrar por Ley_Conc_Final (inclusive). Si es None, no filtra por máximo
        nombre_archivo_salida: Nombre del archivo de salida. Si es None, usa 'results_montecarlo.xlsx'
    
    Returns:
        Dict donde cada llave es el id de simulación y el valor es un DataFrame con resultados MC filtrados
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
                            s1 = np.random.uniform(0.02, 0.7) #masa
                            s2 = np.random.uniform(0.02, 0.9) #cuf
                            eq.split_factor = [s1, s2]
                    
                    for i in range(100):
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
        
        cut_cols = [col for col in df_resultados_mc.columns if col.startswith('Flujo') and col.endswith('Cut')]
        for col in cut_cols:
            df_resultados_mc = df_resultados_mc[df_resultados_mc[col] < 36]

        # Aplicar filtros por Ley_Conc_Final si se especificaron
        if ley_conc_final_min is not None:
            df_resultados_mc = df_resultados_mc[df_resultados_mc['Ley_Conc_Final'] >= ley_conc_final_min]
        if ley_conc_final_max is not None:
            df_resultados_mc = df_resultados_mc[df_resultados_mc['Ley_Conc_Final'] <= ley_conc_final_max]
        
        results_por_sim[sim_id] = df_resultados_mc
        
        total_iter = len(resultados_mc)
        iter_filtradas = len(df_resultados_mc)
        print(f"Simulación {sim_id} completada: {total_iter:,} iteraciones exitosas, {iter_filtradas:,} después del filtro")

    # Guardar resultados en Excel separados por hoja según simulación
    os.makedirs('results', exist_ok=True)
    if nombre_archivo_salida is None:
        nombre_archivo_salida = 'results_montecarlo.xlsx'
    archivo_salida = os.path.join('results', nombre_archivo_salida)
    
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
    Ejecuta las simulaciones normales y Monte Carlo para cada hoja:
    - Simulaciones normales: "Sim Dia", "Sim Noche", "Sim Promedio"
    - Simulaciones Monte Carlo: "Sim MC Dia", "Sim MC Noche", "Sim MC Promedio"
    
    Retorna:
    - resultados_normales: Diccionario {nombre_hoja: DataFrame} con resultados de simulación normal
    - resultados_mc: Diccionario {nombre_hoja: {sim_id: DataFrame}} con resultados de Monte Carlo
    """
    import os
    
    # Configuración
    archivo_base = '../Simulacion_caso_base.xlsx'
    semilla = {4: {'masa': 23.84, 'cut': 2.5167}}
    equipos_a_cambiar = ["Jameson 1"]
    ley_min = 10
    ley_max = 26
    n_sim_mc = 10000
    
    # Hojas de simulación normal
    hojas_sim_normal = ["Sim Dia", "Sim Noche", "Sim Promedio"]
    
    # Hojas de simulación Monte Carlo
    hojas_sim_mc = ["Sim MC Dia", "Sim MC Noche", "Sim MC Promedio"]
    
    # Diccionarios para almacenar resultados
    resultados_normales = {}
    resultados_mc = {}
    
    # Crear directorio de resultados
    os.makedirs('results', exist_ok=True)
    
    # ===== SIMULACIONES NORMALES =====
    print(f"\n{'='*60}")
    print("EJECUTANDO SIMULACIONES NORMALES")
    print(f"{'='*60}\n")
    
    for hoja in hojas_sim_normal:
        print(f"\n{'='*60}")
        print(f"Procesando hoja: {hoja}")
        print(f"{'='*60}")
        
        # Cargar equipos y flujos desde la hoja específica
        lista_equipos, flujos = cargar_datos_equipos(archivo_base, sheet_name=hoja)
        
        # Ejecutar simulación normal
        df_resultados = correr_simulacion_normal(lista_equipos, flujos)
        
        # Guardar resultados
        resultados_normales[hoja] = df_resultados
        
        # Guardar en Excel
        nombre_archivo = f"results/results_{hoja.lower().replace(' ', '_')}.xlsx"
        with pd.ExcelWriter(nombre_archivo) as writer:
            for sim_id in df_resultados['Simulacion'].unique():
                df_sim = df_resultados[df_resultados['Simulacion'] == sim_id]
                df_sim.to_excel(writer, sheet_name=f'Simulacion_{sim_id}', index=False)
        
        print(f"Resultados guardados en: {nombre_archivo}")
    
    # ===== SIMULACIONES MONTE CARLO =====
    print(f"\n{'='*60}")
    print("EJECUTANDO SIMULACIONES MONTE CARLO")
    print(f"{'='*60}\n")
    
    for hoja_mc in hojas_sim_mc:
        print(f"\n{'='*60}")
        print(f"Procesando hoja MC: {hoja_mc}")
        print(f"{'='*60}")
        
        # Ejecutar simulación Monte Carlo
        nombre_archivo_mc = f"results_montecarlo_{hoja_mc.lower().replace(' ', '_')}.xlsx"
        df_resultados_mc = correr_simulacion_montecarlo(
            archivo_base=archivo_base,
            equipos_objetivo=equipos_a_cambiar,
            semilla_inicial_flujos=semilla,
            n_sim=n_sim_mc,
            sheet_name=hoja_mc,
            ley_conc_final_min=ley_min,
            ley_conc_final_max=ley_max,
            nombre_archivo_salida=nombre_archivo_mc
        )
        
        # Guardar resultados
        resultados_mc[hoja_mc] = df_resultados_mc
    
    # Mostrar resumen de resultados
    print(f"\n{'='*60}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    
    try:
        from IPython.display import display
        for hoja, df in resultados_normales.items():
            print(f"\nResultados normales - {hoja}:")
            display(df.head())
        
        for hoja_mc, dict_mc in resultados_mc.items():
            if dict_mc:
                first_sim_id = list(dict_mc.keys())[0]
                print(f"\nResultados MC - {hoja_mc} (Simulación {first_sim_id}):")
                display(dict_mc[first_sim_id].head())
    except ImportError:
        for hoja, df in resultados_normales.items():
            print(f"\nResultados normales - {hoja}:")
            print(df.head())
        
        for hoja_mc, dict_mc in resultados_mc.items():
            if dict_mc:
                first_sim_id = list(dict_mc.keys())[0]
                print(f"\nResultados MC - {hoja_mc} (Simulación {first_sim_id}):")
                print(dict_mc[first_sim_id].head())
    
    # Retornar todas las salidas
    return resultados_normales, resultados_mc
    
# Ejecutar si es programa principal

if __name__ == "__main__":
    resultados_normales, resultados_mc = main()

# ===== SECCIÓN DE GRÁFICAS =====
# Crear gráficas para las 3 simulaciones: Día, Noche, Promedio
# Cada simulación tendrá 2 subplots: Split Masa vs Split CuF, y Nube MC + Test de Dilución
#%%
# Definir las hojas a procesar (en orden: Día, Noche)
hojas_mc = ["Sim MC Dia", "Sim MC Noche"]
hojas_normal = ["Sim Dia", "Sim Noche"]
turnos = ["Día", "Noche"]

# Cargar datos de Monte Carlo para cada hoja
dict_sims_mc_por_turno = {}
for hoja_mc, turno in zip(hojas_mc, turnos):
    if hoja_mc in resultados_mc and resultados_mc[hoja_mc]:
        dict_sims_mc_por_turno[turno] = resultados_mc[hoja_mc]
    else:
        # Fallback: intentar cargar desde archivo
        try:
            archivo_mc = f'results/results_montecarlo_{hoja_mc.lower().replace(" ", "_")}.xlsx'
            df_mc = pd.read_excel(archivo_mc)
            # Si hay múltiples hojas en el Excel, usar la primera
            if isinstance(df_mc, dict):
                dict_sims_mc_por_turno[turno] = df_mc
            else:
                dict_sims_mc_por_turno[turno] = {1: df_mc}
        except FileNotFoundError:
            print(f"Advertencia: No se encontraron resultados MC para {hoja_mc}")
            dict_sims_mc_por_turno[turno] = {}

# Cargar datos de simulación normal para cada hoja
dict_resultados_normal_por_turno = {}
for hoja_normal, turno in zip(hojas_normal, turnos):
    if hoja_normal in resultados_normales:
        dict_resultados_normal_por_turno[turno] = resultados_normales[hoja_normal]
    else:
        dict_resultados_normal_por_turno[turno] = None

# Verificar que tenemos al menos una simulación MC
n_simulaciones = len([t for t in turnos if dict_sims_mc_por_turno.get(t)])

if n_simulaciones > 0:
    file_test_dil = '../Test Dilucion.xlsx'
    df_test_dil = pd.read_excel(file_test_dil)

    # Datos "pilotaje", se usará para comparar
    data = {
        'Rec Cuf': [94.56,  95.31],
        'Rec Masa': [22.26 , 14.46],
        'Turno': ['Día', 'Noche']
    }
    df_pilotaje = pd.DataFrame(data)

    # Solo id 5 para test de dilución
    df_test_id5 = df_test_dil[df_test_dil['id'] == 5]
    if 'Recuperación, Cu%' in df_test_id5.columns:
        df_test_id5 = df_test_id5[df_test_id5['Recuperación, Cu%'] != 100]
    df_test_id5_entero = df_test_id5.copy()
    df_test_id5_entero['Recuperación, Cu%'] = df_test_id5_entero['Recuperación, Cu%'].round().astype(int)
    df_test_id5_entero['Ley acumulada, Cu%'] = df_test_id5_entero['Ley acumulada, Cu%'].round().astype(int)

    # Definir colores y marcadores (solo Día y Noche)
    colores = {'Día': 'orange', 'Noche': 'blue'}
    marcadores = {'Día': 'D', 'Noche': 'D'}

    ##############
    # 1. Grafica SOLO test de dilucion id 5
    ##############
    plt.figure(figsize=(8,6))
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
        label=f"Test de Dilución Ley Alimentación = {float(round(df_test_id5['Ley Cu'].iloc[0],2)) if not df_test_id5.empty else ''}",
        zorder=10,
    )
    plt.xlabel('Recuperación (%)')
    plt.ylabel('Ley de Conc. Final (%)')
    plt.title('Test de Dilución Ley Alimentación = 2.25% Cu')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    ##############
    # 2. Graficar datos de pilotaje (Día y Noche) + puntos de Simulación 1, 2, 3 (si existen)
    ##############
    plt.figure(figsize=(8,6))

    # Agregar la línea del test de dilución primero (atrás del resto)
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
        label=f"Test de Dilución Ley Alimentación = {float(round(df_test_id5['Ley Cu'].iloc[0],2)) if not df_test_id5.empty else ''}",
        zorder=10,
    )

    # Puntos de Simulación 1, 2, 3 (para cada turno si existen) - deben ir SOBRE los de pilotaje
    colores_sim_turno = {'Día': 'orange', 'Noche': 'blue'}
    marcadores_sim = {1: "o", 2: "s", 3: "v"}
    sim_labels = {1: "Simulación 1", 2: "Simulación 2", 3: "Simulación Pilotaje"}

    # Guardar cada punto para evitar duplicados en la leyenda
    handles_for_legend = []
    labels_for_legend = []

    for turno in turnos:
        df_result = dict_resultados_normal_por_turno.get(turno)
        if df_result is not None and not df_result.empty:
            for sim_number in [1,2,3]:
                df_sim = df_result[df_result['Simulacion']==sim_number]
                if not df_sim.empty:
                    x = df_sim.iloc[0]['Recuperacion']
                    y = df_sim.iloc[0]['Ley_Conc_Final']
                    sc = plt.scatter(
                        x, y,
                        color=colores_sim_turno.get(turno, "gray"),
                        marker=marcadores_sim.get(sim_number, "o"),
                        s=120,
                        edgecolor='black',
                        linewidth=1.8,
                        zorder=16,  # Más arriba que pilotaje
                        label=f"{sim_labels.get(sim_number,'Simulación')} ({turno})"
                    )
                    # Etiqueta a la derecha
                    plt.annotate(f" {sim_number}",
                                (x, y),
                                textcoords="offset points",
                                xytext=(12,0),
                                ha='left',
                                va='center',
                                fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                                )
                    handles_for_legend.append(sc)
                    labels_for_legend.append(f"{sim_labels.get(sim_number,'Simulación')} ({turno})")

    # Graficar puntos de pilotaje Día y Noche
    pilot_labels = ['Día', 'Noche']
    color_dia = 'orange'
    color_noche = 'blue'
    colores_pilot = {'Día': color_dia, 'Noche': color_noche}
    marcadores_pilot = {'Día': 'D', 'Noche': 'D'}
    for i, turno in enumerate(pilot_labels):
        x_pilot = df_pilotaje.loc[i, 'Rec Cuf']
        y_pilot = df_pilotaje.loc[i, 'Rec Masa']
        sc = plt.scatter(
            x_pilot, y_pilot,
            color=colores_pilot[turno],
            marker=marcadores_pilot[turno],
            s=180,
            edgecolor='black',
            linewidth=2,
            zorder=12,  # Debajo de los puntos de simulación
            label=f"Pilotaje {turno}"
        )
        # Etiqueta a la izquierda
        plt.annotate(f"{turno}",
                    (x_pilot, y_pilot),
                    textcoords="offset points",
                    xytext=(-16,0),
                    ha='right',
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    color=colores_pilot[turno])
        handles_for_legend.append(sc)
        labels_for_legend.append(f"Pilotaje {turno}")

    plt.xlabel('Recuperación (%)')
    plt.ylabel('Ley de Conc. Final (%)')
    plt.title('Pilotaje Día/Noche + Simulaciones 1, 2, 3 + Test de Dilución')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Filtrar duplicados en las etiquetas de la leyenda manteniendo la preferencia por los primeros
    from collections import OrderedDict
    legend_dict = OrderedDict()
    # Primero los puntos de simulación, luego los puntos de pilotaje, luego otros
    for lbl, hdl in zip(labels_for_legend, handles_for_legend):
        if lbl not in legend_dict:
            legend_dict[lbl] = hdl
    handles, labels = plt.gca().get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l not in legend_dict:
            legend_dict[l] = h

    plt.legend(legend_dict.values(), legend_dict.keys())
    plt.tight_layout()
    plt.show()

    # Gráfico de solo los puntos de simulación 1, 2, 3 en colores de turno, con etiquetas a la derecha
    for turno in turnos:
        df_result = dict_resultados_normal_por_turno.get(turno)
        if df_result is not None and not df_result.empty:
            for sim_number in [1,2,3]:
                df_sim = df_result[df_result['Simulacion']==sim_number]
                if not df_sim.empty:
                    x = int(round(df_sim.iloc[0]['Recuperacion']))
                    y = int(round(df_sim.iloc[0]['Ley_Conc_Final']))
                    plt.scatter(
                        x, y,
                        color=colores_sim_turno.get(turno, "gray"),
                        marker=marcadores_sim.get(sim_number, "o"),
                        s=120,
                        edgecolor='black',
                        linewidth=1.8,
                        zorder=12,
                        label=f"{sim_labels.get(sim_number,'Simulación')} ({turno})"
                    )
                    # Etiqueta a la derecha
                    plt.annotate(f" {sim_number}",
                                (x, y),
                                textcoords="offset points",
                                xytext=(12,0),
                                ha='left',
                                va='center',
                                fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                                )
    plt.xlabel('Recuperación (%)')
    plt.ylabel('Ley de Conc. Final (%)')
    plt.title('Test de Dilución (id=5) + Simulaciones 1, 2, 3')
    plt.grid(True, linestyle='--', alpha=0.6)
    handles, labels = plt.gca().get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

    ##############
    # 3. El gráfico original (Split vs Split y Nube MC+Test) POR TURNO
    ##############

    def plot_mc_and_split_per_turno(
        dict_sims_mc_por_turno,
        dict_resultados_normal_por_turno,
        df_pilotaje,
        colores,
        marcadores,
        df_test_id5,
        df_test_id5_entero,
        turnos,
        er_min=6,
        er_max=11,
        fig_width=17,
        fig_height_per_row=6,
        jameson_1_split_masa_col="Jameson 1_split_masa",
        jameson_1_split_cuf_col="Jameson 1_split_cuf"
    ):
        """
        Genera los subplots Split vs Split y Nube MC + Test por turno.
        Filtra er_jameson_1 una sola vez, y utiliza los mismos datos filtrados para ambos subplots.
        La leyenda para 'Simulación Normal' aparece una sola vez como 'Simulación'.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        ncols = 2
        nrows = len(turnos)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height_per_row * nrows))

        if nrows == 1:
            axes = axes[np.newaxis, :]  # Garantiza que axes se indexe por turno

        for idx_turno, turno in enumerate(turnos):
            # Ejes para turno
            ax0 = axes[idx_turno, 0]
            ax1 = axes[idx_turno, 1]

            # Obtener datos MC
            dict_sims_mc_turno = dict_sims_mc_por_turno.get(turno, {})

            if not dict_sims_mc_turno:
                ax0.text(0.5, 0.5, f"No hay datos MC para {turno}", ha='center', va='center', transform=ax0.transAxes)
                ax0.axis('off')
                ax1.text(0.5, 0.5, f"No hay datos MC para {turno}", ha='center', va='center', transform=ax1.transAxes)
                ax1.axis('off')
                continue

            # Primer DataFrame de MC para este turno
            sim_id = list(dict_sims_mc_turno.keys())[0]
            df_mc = dict_sims_mc_turno[sim_id].copy()

            # Calcular er_jameson_1 si es posible
            if jameson_1_split_masa_col in df_mc.columns and jameson_1_split_cuf_col in df_mc.columns:
                df_mc["er_jameson_1"] = df_mc[jameson_1_split_cuf_col] / df_mc[jameson_1_split_masa_col]
            else:
                df_mc["er_jameson_1"] = np.nan  # Para que la variable siempre exista

            # Aplicar filtro ER UNA SOLA VEZ
            use_er_filter = df_mc["er_jameson_1"].notnull().any()
            if use_er_filter:
                df_mc_er = df_mc[(df_mc["er_jameson_1"] >= er_min) & (df_mc["er_jameson_1"] <= er_max)].copy()
            else:
                df_mc_er = df_mc.copy()  # Si no puede filtrar, sigue con el original (vacío si corresponde)

            # --- SUBPLOT 1: Split Masa vs Split CuF de Jameson 1 ---
            if use_er_filter and not df_mc_er.empty:
                im0 = ax0.scatter(
                    df_mc_er[jameson_1_split_masa_col], df_mc_er[jameson_1_split_cuf_col],
                    alpha=0.7, s=18, c=df_mc_er["er_jameson_1"], cmap="RdYlBu"
                )
                ax0.set_xlabel("Split Masa Jameson 1", fontsize=11)
                ax0.set_ylabel("Split CuF Jameson 1", fontsize=11)
                ax0.set_title(f"Jameson 1: Split Masa vs Split CuF\n(color = ER) - {turno}", fontsize=12)
                cbar0 = fig.colorbar(im0, ax=ax0)
                cbar0.set_label("ER Jameson 1", fontsize=10)
                ax0.grid(True, alpha=0.2)
            elif not use_er_filter:
                ax0.text(0.5, 0.5, "Faltan datos para graficar splits", ha='center', va='center', transform=ax0.transAxes)
                ax0.axis('off')
            else:
                ax0.text(0.5, 0.5, "No hay datos después del filtro ER", ha='center', va='center', transform=ax0.transAxes)
                ax0.axis('off')

            # --- SUBPLOT 2: Nube MC y Test de Dilución ---
            # Nube de simulación MC
            if "Recuperacion" in df_mc_er.columns and "Ley_Conc_Final" in df_mc_er.columns and not df_mc_er.empty:
                sc1 = ax1.scatter(
                    df_mc_er['Recuperacion'],
                    df_mc_er['Ley_Conc_Final'],
                    c=df_mc_er['er_jameson_1'] if use_er_filter else None,
                    cmap='RdYlBu' if use_er_filter else None,
                    alpha=1,
                    s=15,
                    label=f"Simulación Monte Carlo ({turno})"
                )
                if use_er_filter:
                    cbar1 = fig.colorbar(sc1, ax=ax1, label="ER Jameson 1", pad=0.02)

            # Pilotaje: mostrar solo el punto correspondiente a este turno
            df_grupo_pilotaje = df_pilotaje[df_pilotaje['Turno'] == turno]
            if not df_grupo_pilotaje.empty:
                ax1.scatter(
                    df_grupo_pilotaje['Rec Cuf'],
                    df_grupo_pilotaje['Rec Masa'],
                    color=colores[turno],
                    marker=marcadores[turno],
                    s=150,
                    label=f"Pilotaje {turno}",
                    edgecolor='black',
                    linewidth=2,
                    zorder=5
                )

            # Simulación normal para este turno
            df_resultados_turno = dict_resultados_normal_por_turno.get(turno)
            simulacion_normal_handle = None
            label_added = False
            if df_resultados_turno is not None and not df_resultados_turno.empty:
                for idx, row in df_resultados_turno.iterrows():
                    x = int(round(row['Recuperacion']))
                    y = int(round(row['Ley_Conc_Final']))
                    # Solo el primer punto lleva la etiqueta "Simulación", los siguientes quedan sin etiqueta
                    label = "Simulación" if not label_added else None
                    h = ax1.scatter(
                        x, y,
                        color=colores[turno],
                        marker='*',
                        s=150,
                        edgecolor='black',
                        linewidth=1.5,
                        label=label,
                        zorder=6
                    )
                    if not label_added:
                        simulacion_normal_handle = h
                        label_added = True
                    ax1.text(
                        x,
                        y + 0.5,
                        f"Sim {int(round(row['Simulacion']))}",
                        fontsize=10,
                        ha='center',
                        va='bottom',
                        fontweight='bold',
                        color='black',
                        zorder=7
                    )

            # Resultados de test de dilución (misma gráfica)
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
                label=f"Test de Dilución Ley Alimentación = {float(round(df_test_id5['Ley Cu'].iloc[0],2))}",
                zorder=10,
                ax=ax1
            )

            # Configurar ticks y etiquetas para subplot 2
            x_start = max(0, 5 * int(np.floor(ax1.get_xlim()[0] / 5)))
            x_end = 100
            xticks = np.arange(x_start, x_end + 1, 5)
            ax1.set_xticks(xticks)
            ax1.set_xlim(x_start, x_end)
            ylim = ax1.get_ylim()
            y_start = 2 * int(np.floor(ylim[0] / 2))
            y_end = 2 * int(np.ceil(ylim[1] / 2))
            yticks = np.arange(y_start, y_end + 1, 2)
            ax1.set_yticks(yticks)
            ax1.set_ylim(y_start, y_end)
            ax1.set_xlabel('Recuperación (%)', fontsize=11)
            ax1.set_ylabel('Ley de Conc. Final (%)', fontsize=11)
            ax1.set_title(f'Nube de Simulación y Test de Dilución - {turno}', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.6)
            # Colocar leyenda fuera, a la derecha, centrada verticalmente
            handles, labels = ax1.get_legend_handles_labels()
            seen = set()
            new_handles = []
            new_labels = []
            for h, l in zip(handles, labels):
                if l not in seen:
                    new_handles.append(h)
                    new_labels.append(l)
                    seen.add(l)
            ax1.legend(new_handles, new_labels, bbox_to_anchor=(1.1, 0.81), loc='center left', fontsize=12, frameon=True)

        plt.tight_layout(rect=[0, 0, 1, 1])  # Dejar más espacio a la derecha
        plt.show()

    # Uso de la función
    if dict_sims_mc_por_turno and dict_resultados_normal_por_turno:
        plot_mc_and_split_per_turno(
            dict_sims_mc_por_turno=dict_sims_mc_por_turno,
            dict_resultados_normal_por_turno=dict_resultados_normal_por_turno,
            df_pilotaje=df_pilotaje,
            colores=colores,
            marcadores=marcadores,
            df_test_id5=df_test_id5,
            df_test_id5_entero=df_test_id5_entero,
            turnos=turnos,
            er_min=6,
            er_max=11
        )
    else:
        print("No se pueden generar las gráficas: faltan datos de Monte Carlo")


 ##############
    # 4. El gráfico original (Split vs Split y Nube MC+Test) POR TURNO TOP 10 SIMULACIONES
    ##############

    def plot_top_simulaciones_by_ley_per_turno(
        dict_sims_mc_por_turno,
        dict_resultados_normal_por_turno,
        df_pilotaje,
        colores,
        marcadores,
        df_test_id5,
        df_test_id5_entero,
        turnos,
        er_min=6,
        er_max=11,
        fig_width=17,
        fig_height_per_row=6,
        jameson_1_split_masa_col="Jameson 1_split_masa",
        jameson_1_split_cuf_col="Jameson 1_split_cuf",
        excel_filename='top10_simulaciones.xlsx'
    ):
        """
        Genera los subplots Split vs Split y Nube MC + Test por turno.
        Filtra er_jameson_1 una sola vez, y utiliza los mismos datos filtrados para ambos subplots.
        Adicionalmente, identifica y destaca las top 10 simulaciones con mayor Ley_Conc_Final en el gráfico de la izquierda.
        Devuelve un dataframe con estas simulaciones y guarda un excel.
        Los puntos que no son top, se muestran poco visibles (gris y alpha muy bajo) para destacarlos menos.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        ncols = 2
        nrows = len(turnos)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height_per_row * nrows))

        # Para almacenar todas las top 10 de cada turno
        lista_top10_df = []

        # Color palette for the top 10
        colores_top10 = sns.color_palette("tab10", 10)

        if nrows == 1:
            axes = axes[np.newaxis, :]  # Garantiza que axes se indexe por turno

        for idx_turno, turno in enumerate(turnos):
            # Ejes para turno
            ax0 = axes[idx_turno, 0]
            ax1 = axes[idx_turno, 1]

            # Obtener datos MC
            dict_sims_mc_turno = dict_sims_mc_por_turno.get(turno, {})

            if not dict_sims_mc_turno:
                ax0.text(0.5, 0.5, f"No hay datos MC para {turno}", ha='center', va='center', transform=ax0.transAxes)
                ax0.axis('off')
                ax1.text(0.5, 0.5, f"No hay datos MC para {turno}", ha='center', va='center', transform=ax1.transAxes)
                ax1.axis('off')
                continue

            # Primer DataFrame de MC para este turno
            sim_id = list(dict_sims_mc_turno.keys())[0]
            df_mc = dict_sims_mc_turno[sim_id].copy()

            # Calcular er_jameson_1 si es posible
            if jameson_1_split_masa_col in df_mc.columns and jameson_1_split_cuf_col in df_mc.columns:
                df_mc["er_jameson_1"] = df_mc[jameson_1_split_cuf_col] / df_mc[jameson_1_split_masa_col]
            else:
                df_mc["er_jameson_1"] = np.nan  # Para que la variable siempre exista

            # Aplicar filtro ER UNA SOLA VEZ
            use_er_filter = df_mc["er_jameson_1"].notnull().any()
            if use_er_filter:
                df_mc_er = df_mc[(df_mc["er_jameson_1"] >= er_min) & (df_mc["er_jameson_1"] <= er_max)].copy()
            else:
                df_mc_er = df_mc.copy()  # Si no puede filtrar, sigue con el original (vacío si corresponde)

            # --- Encontrar Top 10 simulaciones con mayor Ley_Conc_Final ---
            if "Ley_Conc_Final" in df_mc_er.columns and not df_mc_er.empty:
                df_top10 = df_mc_er.nlargest(10, 'Ley_Conc_Final').copy()
            else:
                df_top10 = pd.DataFrame()

            df_top10['Turno'] = turno
            lista_top10_df.append(df_top10)

            # Añadir ID a las top 10 para mostrar en el plot
            if not df_top10.empty:
                df_top10 = df_top10.reset_index(drop=True)
                df_top10['Top10_ID'] = ['TOP{:02d}'.format(i+1) for i in range(len(df_top10))]
                # Para identificar rápido quién pertenece al top10
                mask_top10 = df_mc_er.index.isin(df_top10.index)
            else:
                # En caso de estar vacío, ningún top
                mask_top10 = np.zeros(len(df_mc_er), dtype=bool)

            # --- SUBPLOT 1: Split Masa vs Split CuF de Jameson 1, con TOP 10 destacado ---
            if use_er_filter and not df_mc_er.empty:
                # 1. Dibujar primero los puntos que NO son top10: color gris y alpha bajo
                idx_nontop = ~mask_top10
                if idx_nontop.any():
                    ax0.scatter(
                        df_mc_er.loc[idx_nontop, jameson_1_split_masa_col],
                        df_mc_er.loc[idx_nontop, jameson_1_split_cuf_col],
                        color="lightgray",
                        alpha=1,
                        s=18,
                        marker='o',
                        label=None
                    )
                # 2. Si desea también mostrar el mapa de color ER en el top10, lo podría hacer, aunque aquí destacamos coloreados
                # 3. Ahora destacar TOP10 en color
                if not df_top10.empty:
                    for i, row in df_top10.iterrows():
                        ax0.scatter(
                            row[jameson_1_split_masa_col],
                            row[jameson_1_split_cuf_col],
                            color=colores_top10[i % 10],
                            s=110,
                            marker='o',
                            edgecolor='black',
                            linewidth=2,
                            label=row['Top10_ID'] if i == 0 else None,  # Solo el primer TOP10 aparece en leyenda
                            zorder=10
                        )
  
                ax0.set_xlabel("Split Masa Jameson 1", fontsize=11)
                ax0.set_ylabel("Split CuF Jameson 1", fontsize=11)
                ax0.set_title(f"Jameson 1: Split Masa vs Split CuF\n(TOP10 color, resto gris; ER no visible) - {turno}", fontsize=12)
                # No se hace colorbar porque los no-top son grises (no hay mapa de color)
                ax0.grid(True, alpha=0.2)
            elif not use_er_filter:
                ax0.text(0.5, 0.5, "Faltan datos para graficar splits", ha='center', va='center', transform=ax0.transAxes)
                ax0.axis('off')
            else:
                ax0.text(0.5, 0.5, "No hay datos después del filtro ER", ha='center', va='center', transform=ax0.transAxes)
                ax0.axis('off')

            # --- SUBPLOT 2: Nube MC y Test de Dilución ---
            # Nube de simulación MC
            if "Recuperacion" in df_mc_er.columns and "Ley_Conc_Final" in df_mc_er.columns and not df_mc_er.empty:
                # 1. Dibujar los que NO son top en gris y apagado
                idx_nontop2 = ~mask_top10
                if idx_nontop2.any():
                    ax1.scatter(
                        df_mc_er.loc[idx_nontop2, 'Recuperacion'],
                        df_mc_er.loc[idx_nontop2, 'Ley_Conc_Final'],
                        color='lightgray',
                        alpha=1,
                        s=15,
                        marker='o',
                        label=None
                    )
                # 2. TOP10 destacados
                if not df_top10.empty:
                    for i, row in df_top10.iterrows():
                        ax1.scatter(
                            row['Recuperacion'],
                            row['Ley_Conc_Final'],
                            color=colores_top10[i % 10],
                            s=110,
                            marker='o',
                            edgecolor='black',
                            linewidth=2,
                            label=row['Top10_ID'] if i == 0 else None,  # Solo el primer TOP10 aparece en leyenda
                            zorder=10
                        )

                # No colorbar, ya que los no-top son grises y los top10 van en color
                # (El usuario puede descomentar lo siguiente si quiere seguir mostrando ER de top10, pero por claridad: no)
                # if use_er_filter:
                #     cbar1 = fig.colorbar(sc1, ax=ax1, label="ER Jameson 1", pad=0.02)

            # Pilotaje: mostrar solo el punto correspondiente a este turno
            df_grupo_pilotaje = df_pilotaje[df_pilotaje['Turno'] == turno]
            if not df_grupo_pilotaje.empty:
                ax1.scatter(
                    df_grupo_pilotaje['Rec Cuf'],
                    df_grupo_pilotaje['Rec Masa'],
                    color=colores[turno],
                    marker=marcadores[turno],
                    s=150,
                    label=f"Pilotaje {turno}",
                    edgecolor='black',
                    linewidth=2,
                    zorder=5
                )

            # Simulación normal para este turno
            df_resultados_turno = dict_resultados_normal_por_turno.get(turno)
            simulacion_normal_handle = None
            label_added = False
            if df_resultados_turno is not None and not df_resultados_turno.empty:
                for idx, row in df_resultados_turno.iterrows():
                    x = int(round(row['Recuperacion']))
                    y = int(round(row['Ley_Conc_Final']))
                    # Solo el primer punto lleva la etiqueta "Simulación", los siguientes quedan sin etiqueta
                    label = "Simulación" if not label_added else None
                    h = ax1.scatter(
                        x, y,
                        color=colores[turno],
                        marker='*',
                        s=150,
                        edgecolor='black',
                        linewidth=1.5,
                        label=label,
                        zorder=6
                    )
                    if not label_added:
                        simulacion_normal_handle = h
                        label_added = True
                    ax1.text(
                        x,
                        y + 0.5,
                        f"Sim {int(round(row['Simulacion']))}",
                        fontsize=10,
                        ha='center',
                        va='bottom',
                        fontweight='bold',
                        color='black',
                        zorder=7
                    )

            # Resultados de test de dilución (misma gráfica)
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
                label=f"Test de Dilución Ley Alimentación = {float(round(df_test_id5['Ley Cu'].iloc[0],2))}",
                zorder=10,
                ax=ax1
            )

            # Configurar ticks y etiquetas para subplot 2
            x_start = max(0, 5 * int(np.floor(ax1.get_xlim()[0] / 5)))
            x_end = 100
            xticks = np.arange(x_start, x_end + 1, 5)
            ax1.set_xticks(xticks)
            ax1.set_xlim(x_start, x_end)
            ylim = ax1.get_ylim()
            y_start = 2 * int(np.floor(ylim[0] / 2))
            y_end = 2 * int(np.ceil(ylim[1] / 2))
            yticks = np.arange(y_start, y_end + 1, 2)
            ax1.set_yticks(yticks)
            ax1.set_ylim(y_start, y_end)
            ax1.set_xlabel('Recuperación (%)', fontsize=11)
            ax1.set_ylabel('Ley de Conc. Final (%)', fontsize=11)
            ax1.set_title(f'Nube de Simulación y Test de Dilución - {turno}', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.6)
            # Colocar leyenda fuera, a la derecha, centrada verticalmente
            handles, labels_ = ax1.get_legend_handles_labels()
            # Quitar los labels de las top 10 (generalmente son los colores_top10 o similares)
            # Solo mantener en la leyenda los que no sean top 10
            # Normalmente, los labels de top 10 no tienen que mostrarse, suelen ser "Top 1", "Top 2", ..., etc. 
            # Si quieres ser más exhaustivo, podrías filtrar por estos nombres; 
            # pero aquí simplemente filtramos los labels vacíos o no repetidos sin incluir los de top 10.
            top10_labels = [f"Top {i+1}" for i in range(10)]
            new_handles = []
            new_labels = []
            seen = set()
            for h, l in zip(handles, labels_):
                # Quitar etiquetas de los top 10 de la leyenda
                if l not in seen and l not in [None, ""] and l not in top10_labels:
                    new_handles.append(h)
                    new_labels.append(l)
                    seen.add(l)
            ax1.legend(new_handles, new_labels, bbox_to_anchor=(1.1, 0.81), loc='center left', fontsize=12, frameon=True)

        plt.tight_layout(rect=[0, 0, 1, 1])  # Dejar más espacio a la derecha
        plt.show()

        # Concatenar y guardar todas las top 10 en un Excel
        df_top10_total = pd.concat(lista_top10_df, axis=0, ignore_index=True)
        df_top10_total.to_excel(excel_filename, index=False)
        return df_top10_total

    # Uso de la función
    if dict_sims_mc_por_turno and dict_resultados_normal_por_turno:
        df_top10 = plot_top_simulaciones_by_ley_per_turno(
            dict_sims_mc_por_turno=dict_sims_mc_por_turno,
            dict_resultados_normal_por_turno=dict_resultados_normal_por_turno,
            df_pilotaje=df_pilotaje,
            colores=colores,
            marcadores=marcadores,
            df_test_id5=df_test_id5,
            df_test_id5_entero=df_test_id5_entero,
            turnos=turnos,
            er_min=6,
            er_max=11,
            excel_filename="top10_simulaciones.xlsx"
        )
    else:
        print("No se pueden generar las gráficas: faltan datos de Monte Carlo")