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


def flujos_globales(lista_equipos):

    fin = set()
    fout = set()

    for eq in lista_equipos.items():
        fin.update(eq[1].flow_in)
        fout.update(eq[1].flow_out)

    flujos_entrada = fin - fout
    flujos_salida   = fout - fin
    flujos_salida_conc = flujos_salida - {9}
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
    semilla = {4: {'masa': 24.515, 'cut': 2.40}}
    equipos_a_cambiar = ["Jameson 1"]
    ley_min = 12
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
# Definir las hojas a procesar (en orden: Día, Noche, Promedio)
hojas_mc = ["Sim MC Dia", "Sim MC Noche", "Sim MC Promedio"]
hojas_normal = ["Sim Dia", "Sim Noche", "Sim Promedio"]
turnos = ["Día", "Noche", "Promedio"]

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
        'Rec Cuf': [94.53,  95.20,  94.43],
        'Rec Masa': [22.94 , 14.46, 18.22],
        'Turno': ['Día', 'Noche', 'Promedio']
    }
    df_pilotaje = pd.DataFrame(data)

    # Solo id 5 para test de dilución
    df_test_id5 = df_test_dil[df_test_dil['id'] == 5]
    if 'Recuperación, Cu%' in df_test_id5.columns:
        df_test_id5 = df_test_id5[df_test_id5['Recuperación, Cu%'] != 100]
    df_test_id5_entero = df_test_id5.copy()
    df_test_id5_entero['Recuperación, Cu%'] = df_test_id5_entero['Recuperación, Cu%'].round().astype(int)
    df_test_id5_entero['Ley acumulada, Cu%'] = df_test_id5_entero['Ley acumulada, Cu%'].round().astype(int)

    # Definir colores y marcadores
    colores = {'Día': 'orange', 'Noche': 'blue', 'Promedio': 'green'}
    marcadores = {'Día': 'D', 'Noche': 'D', 'Promedio': 'D'}

    # --- SUBPLOTS ---
    # Crear subplots: 3 filas (Día, Noche, Promedio) x 2 columnas (Split, Nube+Test)
    ncols = 2
    nrows = 3  # Siempre 3 filas para Día, Noche, Promedio
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(17, 6 * nrows))
    
    # Para cada turno (Día, Noche, Promedio)
    for idx_turno, turno in enumerate(turnos):
        # Obtener los ejes para esta fila
        ax0 = axes[idx_turno, 0]  # Subplot izquierdo: Split Masa vs Split CuF
        ax1 = axes[idx_turno, 1]  # Subplot derecho: Nube MC + Test de Dilución
        
        # Obtener datos MC para este turno
        dict_sims_mc_turno = dict_sims_mc_por_turno.get(turno, {})
        
        if not dict_sims_mc_turno:
            # Si no hay datos MC para este turno, mostrar mensaje
            ax0.text(0.5, 0.5, f"No hay datos MC para {turno}", ha='center', va='center', transform=ax0.transAxes)
            ax0.axis('off')
            ax1.text(0.5, 0.5, f"No hay datos MC para {turno}", ha='center', va='center', transform=ax1.transAxes)
            ax1.axis('off')
            continue
        
        # Obtener el primer DataFrame de MC para este turno
        sim_id = list(dict_sims_mc_turno.keys())[0]
        df_mc = dict_sims_mc_turno[sim_id].copy()
        
        # Obtener datos de simulación normal para este turno
        df_resultados_turno = dict_resultados_normal_por_turno.get(turno)
        
        # Calcular er_jameson_1 si es necesario (para ambos subplots)
        if "Jameson 1_split_masa" in df_mc.columns and "Jameson 1_split_cuf" in df_mc.columns:
            df_mc["er_jameson_1"] = df_mc["Jameson 1_split_cuf"] / df_mc["Jameson 1_split_masa"]
        
        # --- SUBPLOT 1: Split Masa vs Split CuF de Jameson 1 ---
        if "Jameson 1_split_masa" in df_mc.columns and "Jameson 1_split_cuf" in df_mc.columns:
            # Filtrar solo para este subplot
            df_mc_filtrado = df_mc[(df_mc["er_jameson_1"] >= 6) & (df_mc["er_jameson_1"] <= 11)].copy()
            
            if not df_mc_filtrado.empty:
                im0 = ax0.scatter(
                    df_mc_filtrado["Jameson 1_split_masa"], df_mc_filtrado["Jameson 1_split_cuf"],
                    alpha=0.7, s=18, c=df_mc_filtrado["er_jameson_1"], cmap="RdYlBu"
                )
                ax0.set_xlabel("Split Masa Jameson 1", fontsize=11)
                ax0.set_ylabel("Split CuF Jameson 1", fontsize=11)
                ax0.set_title(f"Jameson 1: Split Masa vs Split CuF\n(color = ER) - {turno}", fontsize=12)
                cbar0 = fig.colorbar(im0, ax=ax0)
                cbar0.set_label("ER Jameson 1", fontsize=10)
                ax0.grid(True, alpha=0.2)
            else:
                ax0.text(0.5, 0.5, "No hay datos después del filtro ER", ha='center', va='center', transform=ax0.transAxes)
                ax0.axis('off')
        else:
            ax0.text(0.5, 0.5, "Faltan datos para graficar splits", ha='center', va='center', transform=ax0.transAxes)
            ax0.axis('off')
        
        # --- SUBPLOT 2: Nube MC y Test de Dilución ---
        # Filtrar df_mc para el subplot 2 también (si tiene er_jameson_1)
        df_mc_plot2 = df_mc.copy()
        if "er_jameson_1" in df_mc_plot2.columns:
            df_mc_plot2 = df_mc_plot2[(df_mc_plot2["er_jameson_1"] >= 6) & (df_mc_plot2["er_jameson_1"] <= 11)]
        
        # Nube de simulación MC
        if "Recuperacion" in df_mc_plot2.columns and "Ley_Conc_Final" in df_mc_plot2.columns and not df_mc_plot2.empty:
            sc1 = ax1.scatter(
                df_mc_plot2['Recuperacion'],
                df_mc_plot2['Ley_Conc_Final'],
                c=df_mc_plot2['er_jameson_1'] if "er_jameson_1" in df_mc_plot2.columns else None,
                cmap='RdYlBu',
                alpha=1,
                s=15,
                label=f"Simulación Monte Carlo ({turno})"
            )
            if "er_jameson_1" in df_mc_plot2.columns:
                cbar1 = fig.colorbar(sc1, ax=ax1, label="ER Jameson 1", pad=0.02)
        
        # Pilotaje: mostrar solo el punto correspondiente a este turno
        df_grupo_pilotaje = df_pilotaje[df_pilotaje['Turno'] == turno]
        if not df_grupo_pilotaje.empty:
            ax1.scatter(
                df_grupo_pilotaje['Rec Cuf'],
                df_grupo_pilotaje['Rec Masa'],
                color=colores[turno],
                marker=marcadores[turno],
                s=250 if turno == 'Promedio' else 150,
                label=f"Pilotaje {turno}",
                edgecolor='black',
                linewidth=2,
                zorder=5
            )
        
        # Simulación normal (simpre) para este turno
        if df_resultados_turno is not None and not df_resultados_turno.empty:
            # Obtener la primera fila de resultados normales para este turno
            for idx, row in df_resultados_turno.iterrows():
                x = int(round(row['Recuperacion']))
                y = int(round(row['Ley_Conc_Final']))
                ax1.scatter(
                    x, y,
                    color=colores[turno],
                    marker='*',
                    s=350 if turno == 'Promedio' else 250,
                    edgecolor='black',
                    linewidth=1.5,
                    label=f"Simulación Normal ({turno})",
                    zorder=6
                )
                ax1.text(
                    x,
                    y + 0.5,
                    f"Sim {row['Simulacion']}",
                    fontsize=10,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'),
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
            label=f"Test de Dilución Ley Alimentación = {int(round(df_test_id5['Ley Cu'].iloc[0]))}",
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
        ax1.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=8, frameon=True)

    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Dejar más espacio a la derecha para leyenda y colorbar
    plt.show()
else:
    print("No se pueden generar las gráficas: faltan datos de Monte Carlo")
#%%
