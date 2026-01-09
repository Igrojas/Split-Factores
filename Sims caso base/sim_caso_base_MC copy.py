#%%
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import pandas as pd
import os

# Para Jupyter/IPython
try:
    from IPython.display import display
except ImportError:
    display = print


@dataclass
class ConfigMonteCarlo:
    """Configuración para un equipo que usará Monte Carlo."""
    nombre_equipo: str
    rango_masa: Tuple[float, float] = (0.01, 0.95)
    rango_cuf: Tuple[float, float] = (0.01, 0.95)
    decimales: int = 2


@dataclass
class ConfigSimulacion:
    """Configuración centralizada de la simulación."""
    # Archivos
    archivo_excel: str = 'Simulacion_caso_base.xlsx'
    hoja_equipos: str = None  # None = primera hoja
    hoja_alimentacion: str = 'Alim'
    carpeta_resultados: str = 'results'
    
    # Simulación
    num_simulacion: int = 1
    flujo_alimentacion_id: int = 4
    masa_alimentacion: float = 100.0
    max_iteraciones: int = 100
    
    # Monte Carlo
    usar_monte_carlo: bool = True
    iteraciones_mc: int = 100000
    equipos_monte_carlo: List[ConfigMonteCarlo] = field(default_factory=list)
    
    # Flujos de salida
    flujos_salida_excluir: List[int] = field(default_factory=lambda: [9])  # Flujos a excluir de concentrado
    
    def __post_init__(self):
        """Inicializa configuración por defecto si no se especifica."""
        if not self.equipos_monte_carlo and self.usar_monte_carlo:
            # Configuración por defecto basada en el código original
            self.equipos_monte_carlo = [
                ConfigMonteCarlo('1ra Cl Ro', (0.01, 0.95), (0.01, 0.95)),
                ConfigMonteCarlo('Scavenger', (0.01, 0.95), (0.01, 0.95)),
                ConfigMonteCarlo('1ra Cl Scv', (0.01, 0.95), (0.01, 0.95)),
            ]


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
        super().__init__()

    def calcula(self,flujos):
        #print(self.name)
        flujos[self.flow_out[0]].masa=0
        flujos[self.flow_out[0]].cuf=0

        for flow in self.flow_in:
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


def flujos_globales(lista_equipos: Dict, flujos_excluir: List[int] = None) -> Tuple[set, set, set, set]:
    """
    Calcula los flujos globales del circuito.
    
    Args:
        lista_equipos: Diccionario con equipos de la simulación
        flujos_excluir: Lista de IDs de flujos a excluir de concentrado
        
    Returns:
        Tupla con (flujos_entrada, flujos_salida, flujos_salida_conc, flujos_internos)
    """
    if flujos_excluir is None:
        flujos_excluir = []
    
    fin = set()
    fout = set()

    for eq in lista_equipos.items():
        fin.update(eq[1].flow_in)
        fout.update(eq[1].flow_out)

    print(f"Flujos entrada: {fin}")
    print(f"Flujos salida: {fout}")

    flujos_entrada = fin - fout
    flujos_salida = fout - fin
    flujos_salida_conc = flujos_salida - set(flujos_excluir)
    flujos_internos = fin & fout

    return flujos_entrada, flujos_salida, flujos_salida_conc, flujos_internos


def cargar_equipos_desde_excel(config: ConfigSimulacion) -> Tuple[Dict, Dict]:
    """
    Carga equipos y flujos desde archivo Excel.
    
    Args:
        config: Configuración de la simulación
        
    Returns:
        Tupla con (lista_equipos, flujos)
    """
    # Cargar datos del Excel
    if config.hoja_equipos:
        df = pd.read_excel(config.archivo_excel, sheet_name=config.hoja_equipos)
    else:
        df = pd.read_excel(config.archivo_excel)
    
    df = df[df["Simulacion"] == config.num_simulacion]
    
    num_simulaciones = df["Simulacion"].nunique()
    print(f"Total de simulaciones encontradas: {num_simulaciones}")
    print(f"Usando simulación: {config.num_simulacion}")

    lista_equipos = {}
    flujos = {}
    
    lista_equipos[config.num_simulacion] = {}
    
    for idx, row in df.iterrows():
        # Crear equipo según tipo
        if row["tipo"] == "celda":
            equipo = celda()
        elif row["tipo"] == "suma":
            equipo = suma()
        else:
            print(f"Tipo de equipo desconocido: {row['tipo']}, saltando...")
            continue

        # Obtener columnas de flujos (después de 'sp cuf')
        columnas_flujos = list(df.columns)
        idx_sp_cuf = columnas_flujos.index("sp cuf")
        columnas_flujos = columnas_flujos[idx_sp_cuf+1:]

        # Procesar flujos de entrada (=1)
        equipo.flow_in = []
        for col in columnas_flujos:
            if row[col] == 1:
                if col not in flujos:
                    flujos[col] = flujo(col)
                    flujos[col].name = f"Alim {row['Equipo']}"
                equipo.flow_in.append(col)
        
        # Procesar flujos de salida (= -1)
        equipo.flow_out = []
        out_count = 0
        for col in columnas_flujos:
            if row[col] == -1:
                if col not in flujos:
                    flujos[col] = flujo(col)
                    if out_count == 0:
                        flujos[col].name = f"Conc {row['Equipo']}"
                    elif out_count == 1:
                        flujos[col].name = f"Rel {row['Equipo']}"
                equipo.flow_out.append(col)
                out_count += 1
        
        # Asignar nombre y split factor
        equipo.name = row["Equipo"]
        equipo.split_factor = [row["sp masa"], row["sp cuf"]]
        lista_equipos[config.num_simulacion][row["Equipo"]] = equipo

    return lista_equipos, flujos


def cargar_alimentacion(config: ConfigSimulacion, flujos: Dict) -> None:
    """
    Carga datos de alimentación desde Excel y los asigna al flujo correspondiente.
    
    Args:
        config: Configuración de la simulación
        flujos: Diccionario de flujos
    """
    df_alim = pd.read_excel(config.archivo_excel, sheet_name=config.hoja_alimentacion)
    
    if config.flujo_alimentacion_id not in flujos:
        raise ValueError(f"Flujo de alimentación {config.flujo_alimentacion_id} no existe")
    
    flujo_alim = flujos[config.flujo_alimentacion_id]
    flujo_alim.name = 'Alim. 1ra Limpieza'
    flujo_alim.masa = config.masa_alimentacion
    
    # Obtener ley de alimentación del Excel
    ley_alim = float(df_alim[df_alim['Sim'] == config.num_simulacion]['Alim'].iloc[0])
    flujo_alim.cut = ley_alim


def generar_split_factor_mc(config_mc: ConfigMonteCarlo) -> List[float]:
    """
    Genera un split factor aleatorio según configuración de Monte Carlo.
    
    Args:
        config_mc: Configuración de Monte Carlo para el equipo
        
    Returns:
        Lista con [split_factor_masa, split_factor_cuf]
    """
    masa = round(random.uniform(*config_mc.rango_masa), config_mc.decimales)
    cuf = round(random.uniform(*config_mc.rango_cuf), config_mc.decimales)
    return [masa, cuf]


def ejecutar_simulacion(lista_equipos: Dict, flujos: Dict, config: ConfigSimulacion) -> List[Dict]:
    """
    Ejecuta la simulación con o sin Monte Carlo.
    
    Args:
        lista_equipos: Diccionario con equipos
        flujos: Diccionario con flujos
        config: Configuración de la simulación
        
    Returns:
        Lista de diccionarios con resultados
    """
    # Obtener flujos globales
    fe, fs, fs_conc, fi = flujos_globales(
        lista_equipos[config.num_simulacion], 
        config.flujos_salida_excluir
    )
    
    # Cargar alimentación
    cargar_alimentacion(config, flujos)
    
    resultados = []
    num_iteraciones = config.iteraciones_mc if config.usar_monte_carlo else 1
    
    for i in range(num_iteraciones):
        if i % 1000 == 0 and config.usar_monte_carlo:
            print(f"Iteración MC: {i}")
        
        # Aplicar Monte Carlo a equipos seleccionados
        split_factors_mc = {}
        if config.usar_monte_carlo:
            for config_mc in config.equipos_monte_carlo:
                nombre = config_mc.nombre_equipo
                if nombre in lista_equipos[config.num_simulacion]:
                    sf = generar_split_factor_mc(config_mc)
                    lista_equipos[config.num_simulacion][nombre].split_factor = sf
                    split_factors_mc[nombre] = sf
                else:
                    print(f"Advertencia: Equipo '{nombre}' no encontrado para Monte Carlo")
        
        # Reinicializar alimentación en cada iteración
        flujos[config.flujo_alimentacion_id].masa = config.masa_alimentacion
        df_alim = pd.read_excel(config.archivo_excel, sheet_name=config.hoja_alimentacion)
        flujos[config.flujo_alimentacion_id].cut = float(
            df_alim[df_alim['Sim'] == config.num_simulacion]['Alim'].iloc[0]
        )
        
        # Ejecutar iteraciones de convergencia
        for _ in range(config.max_iteraciones):
            for nombre, equipo in lista_equipos[config.num_simulacion].items():
                equipo.calcula(flujos)
        
        # Calcular métricas
        flujo_alim = flujos[config.flujo_alimentacion_id]
        Recuperacion = sum([flujos[i].cuf for i in fs_conc]) / flujo_alim.cuf * 100
        MassPull = sum([flujos[i].masa for i in fs_conc]) / flujo_alim.masa * 100
        RazonEnriquecimiento = Recuperacion / MassPull if MassPull > 0 else 0
        masa_conc = sum([flujos[i].masa for i in fs_conc])
        Ley_Conc_Final = sum([flujos[i].cuf for i in fs_conc]) / masa_conc * 100 if masa_conc > 0 else 0
        error_masa = flujo_alim.masa - sum([flujos[i].masa for i in fs])
        error_ley = flujo_alim.masa * flujo_alim.cut - sum([flujos[i].masa * flujos[i].cut for i in fs])
        
        # Construir fila de resultados
        fila = {
            'Recuperacion': Recuperacion,
            'MassPull': MassPull,
            'RazonEnriquecimiento': RazonEnriquecimiento,
            'Ley_Conc_Final': Ley_Conc_Final,
            'error_masa': error_masa,
            'error_ley': error_ley,
        }
        
        # Agregar split factors de Monte Carlo
        if config.usar_monte_carlo:
            for nombre, sf in split_factors_mc.items():
                nombre_safe = nombre.replace(' ', '_')
                fila[f'sp_{nombre_safe}_masa'] = sf[0]
                fila[f'sp_{nombre_safe}_cuf'] = sf[1]
                fila[f'er_{nombre_safe}'] = sf[1] / sf[0] if sf[0] > 0 else 0
        
        # Agregar datos de todos los flujos
        for k, v in flujos.items():
            fila[f'Flujo {k} Masa'] = v.masa
            fila[f'Flujo {k} Cut'] = v.cut
        
        resultados.append(fila)
    
    return resultados


# =============================================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# =============================================================================

# Crear configuración
config = ConfigSimulacion(
    archivo_excel='Simulacion_caso_base.xlsx',
    num_simulacion=1,
    flujo_alimentacion_id=4,
    masa_alimentacion=100.0,
    max_iteraciones=100,
    usar_monte_carlo=True,
    iteraciones_mc=100000,
    # Especificar equipos para Monte Carlo (puedes agregar/quitar fácilmente)
    equipos_monte_carlo=[
        ConfigMonteCarlo('1ra Cl Ro', (0.01, 0.95), (0.01, 0.95)),
        ConfigMonteCarlo('Scavenger', (0.01, 0.95), (0.01, 0.95)),
        ConfigMonteCarlo('1ra Cl Scv', (0.01, 0.95), (0.01, 0.95)),
        # Agregar más equipos aquí fácilmente:
        # ConfigMonteCarlo('Otro Equipo', (0.05, 0.90), (0.05, 0.90)),
    ],
    flujos_salida_excluir=[9],
)

# =============================================================================
# EJECUCIÓN
# =============================================================================

print(f"{'='*50}")
print("Cargando equipos desde Excel...")
lista_equipos, flujos = cargar_equipos_desde_excel(config)

print(f"{'='*50}")
print("Equipos cargados:")
for nombre, equipo in lista_equipos[config.num_simulacion].items():
    print(f"  - {equipo.name}: Split Factor = {equipo.split_factor}")

print(f"{'='*50}")
print("Ejecutando simulación...")
resultados = ejecutar_simulacion(lista_equipos, flujos, config)

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados)

# Guardar resultados
os.makedirs(config.carpeta_resultados, exist_ok=True)
archivo_resultados = os.path.join(config.carpeta_resultados, 'results_caso_base_MC.xlsx')
df_resultados.round(2).to_excel(archivo_resultados, index=False)
print(f"Resultados guardados en: {archivo_resultados}")

# Mostrar resumen
print(f"\nTotal de iteraciones: {len(resultados)}")
print(f"Columnas en resultados: {len(df_resultados.columns)}")
print("\nPrimeras 10 filas:")
display(df_resultados.round(2).head(10))

#%%
import seaborn as sns
import matplotlib.pyplot as plt

def graficar_resultados(df, rec_min=None, rec_max=None, ley_min=None, ley_max=None, solo_flujo9_cut=False):
    """
    Filtra el DataFrame según los parámetros ingresados.
    Si solo_flujo9_cut=True, sólo filtra por 'Flujo 9 Cut' > 0.2 y < 30 y no por las otras columnas de cut.
    Muestra un subplot 2x2 con gráficos relevantes.
    """
    df_filtrado = df.copy()

    # Filtro FIJO: todos 'Flujo X Cut' < 30
    cut_cols = [col for col in df_filtrado.columns if col.startswith('Flujo') and col.endswith('Cut')]
    for col in cut_cols:
        df_filtrado = df_filtrado[df_filtrado[col] < 30]

    # Filtros variables
    if rec_min is not None:
        df_filtrado = df_filtrado[df_filtrado['Recuperacion'] >= rec_min]
    if rec_max is not None:
        df_filtrado = df_filtrado[df_filtrado['Recuperacion'] <= rec_max]
    if ley_min is not None:
        df_filtrado = df_filtrado[df_filtrado['Ley_Conc_Final'] >= ley_min]
    if ley_max is not None:
        df_filtrado = df_filtrado[df_filtrado['Ley_Conc_Final'] <= ley_max]

    # Detectar automáticamente equipos con split factors en los resultados
    equipos_split = []
    for col in df_filtrado.columns:
        if col.startswith('sp_') and col.endswith('_masa'):
            nombre_base = col.replace('sp_', '').replace('_masa', '')
            col_cuf = f'sp_{nombre_base}_cuf'
            if col_cuf in df_filtrado.columns:
                equipos_split.append((col, col_cuf, nombre_base.replace('_', ' ')))

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    # Primer gráfico: Original scatterplot Recuperacion vs Ley_Conc_Final
    scatter = sns.scatterplot(
        x='Recuperacion',
        y='Ley_Conc_Final',
        hue='RazonEnriquecimiento',
        size='MassPull',
        palette='viridis',
        data=df_filtrado,
        legend='brief',
        ax=axs[0]
    )
    axs[0].grid(True, alpha=0.5)
    axs[0].set_title('Recuperacion vs Ley_Conc_Final')
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Siguientes gráficos: Split factors
    for i, (col_masa, col_cuf, equipo_name) in enumerate(equipos_split[:3]):  # Máximo 3 gráficos
        ax_idx = i+1
        if ax_idx < 4:
            sns.scatterplot(
                x=col_masa,
                y=col_cuf,
                hue='RazonEnriquecimiento',
                size='MassPull',
                palette='viridis',
                data=df_filtrado,
                legend=False,
                ax=axs[ax_idx]
            )
            axs[ax_idx].grid(True, alpha=0.5)
            axs[ax_idx].set_xlabel(f'Split Factor Masa ({equipo_name})')
            axs[ax_idx].set_ylabel(f'Split Factor Cuf ({equipo_name})')
            axs[ax_idx].set_title(f'Split Factor {equipo_name}')

    # Si queda un subplot extra sin datos
    for j in range(len(equipos_split) + 1, 4):
        if j < len(axs):
            axs[j].axis('off')

    plt.tight_layout()
    plt.show()

    return df_filtrado

# Ejemplo de uso con los filtros originales:
df_filtrado = graficar_resultados(df_resultados.round(2), rec_min=97, ley_min=10, ley_max=30)

def guardar_resultados_avanzado(df_filtrado, archivo='results/results_caso_base_MC_filtered.xlsx'):
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
            'RazonEnriquecimiento': 'Razón Enriquecimiento',
            'Ley_Conc_Final': 'Ley de Conc. Final',
            'Flujo 9 Masa': 'Flujo 9 Masa',
            'Flujo 9 Cut': 'Flujo 9 Cut'
        }
        df_resumen.rename(columns=renombres, inplace=True)
        df_resumen.to_excel(writer, index=False, sheet_name='Top Concentraciones')

guardar_resultados_avanzado(df_filtrado)
# %%
