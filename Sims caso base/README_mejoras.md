# Mejoras en el Código de Simulación

## Resumen de Mejoras

El código ha sido refactorizado para ser más flexible y fácil de mantener, manteniendo la simplicidad original.

## Principales Cambios

### 1. **Configuración Centralizada**
- Se creó la clase `ConfigSimulacion` que centraliza todos los parámetros
- Se creó `ConfigMonteCarlo` para configurar equipos individuales con Monte Carlo
- Todo se configura en un solo lugar al inicio del script

### 2. **Manejo Flexible de Archivos**
- Fácil cambio de archivos Excel de entrada
- Soporte para múltiples hojas
- Configuración de carpeta de resultados

### 3. **Monte Carlo Selectivo**
- Puedes elegir qué equipos usar con Monte Carlo
- Cada equipo puede tener sus propios rangos de variación
- Fácil agregar/quitar equipos del análisis Monte Carlo

### 4. **Código Modular**
- Funciones separadas para carga de datos, simulación, etc.
- Más fácil de entender y mantener
- Reutilizable para diferentes casos

## Ejemplos de Uso

### Ejemplo 1: Configuración Básica (igual al original)

```python
config = ConfigSimulacion(
    archivo_excel='Simulacion_caso_base.xlsx',
    num_simulacion=1,
    usar_monte_carlo=True,
    iteraciones_mc=100000,
)
```

### Ejemplo 2: Cambiar Archivo y Agregar Más Equipos a Monte Carlo

```python
config = ConfigSimulacion(
    archivo_excel='Mi_nuevo_archivo.xlsx',
    hoja_equipos='Equipos',  # Especificar hoja si no es la primera
    num_simulacion=1,
    usar_monte_carlo=True,
    iteraciones_mc=50000,
    equipos_monte_carlo=[
        ConfigMonteCarlo('1ra Cl Ro', (0.01, 0.95), (0.01, 0.95)),
        ConfigMonteCarlo('Scavenger', (0.01, 0.95), (0.01, 0.95)),
        ConfigMonteCarlo('1ra Cl Scv', (0.01, 0.95), (0.01, 0.95)),
        # Agregar nuevo equipo fácilmente:
        ConfigMonteCarlo('2da Cl Ro', (0.05, 0.90), (0.05, 0.90)),
        ConfigMonteCarlo('Rougher', (0.10, 0.85), (0.10, 0.85)),
    ],
)
```

### Ejemplo 3: Simulación SIN Monte Carlo (valores fijos)

```python
config = ConfigSimulacion(
    archivo_excel='Simulacion_caso_base.xlsx',
    num_simulacion=1,
    usar_monte_carlo=False,  # Desactivar Monte Carlo
    max_iteraciones=100,
)
```

### Ejemplo 4: Rangos Personalizados por Equipo

```python
config = ConfigSimulacion(
    archivo_excel='Simulacion_caso_base.xlsx',
    num_simulacion=1,
    usar_monte_carlo=True,
    iteraciones_mc=50000,
    equipos_monte_carlo=[
        # Equipo con rangos más estrechos (más control)
        ConfigMonteCarlo('1ra Cl Ro', (0.20, 0.40), (0.30, 0.50)),
        # Equipo con rangos más amplios
        ConfigMonteCarlo('Scavenger', (0.01, 0.95), (0.01, 0.95)),
        # Equipo con más decimales de precisión
        ConfigMonteCarlo('1ra Cl Scv', (0.01, 0.95), (0.01, 0.95), decimales=3),
    ],
)
```

### Ejemplo 5: Cambiar Parámetros de Alimentación

```python
config = ConfigSimulacion(
    archivo_excel='Simulacion_caso_base.xlsx',
    num_simulacion=1,
    flujo_alimentacion_id=4,  # Cambiar ID del flujo de alimentación
    masa_alimentacion=150.0,   # Cambiar masa de alimentación
    max_iteraciones=200,       # Más iteraciones de convergencia
)
```

## Ventajas de la Nueva Estructura

1. **Fácil de Modificar**: Todo está en la sección de configuración al inicio
2. **Escalable**: Agregar nuevos equipos es solo agregar una línea
3. **Flexible**: Puedes activar/desactivar Monte Carlo por equipo
4. **Mantenible**: Código más organizado y con funciones claras
5. **Reutilizable**: Fácil adaptar para diferentes casos de estudio

## Estructura del Código

```
1. Definiciones de clases (circuito, celda, suma, flujo)
2. Funciones auxiliares:
   - flujos_globales()
   - cargar_equipos_desde_excel()
   - cargar_alimentacion()
   - generar_split_factor_mc()
   - ejecutar_simulacion()
3. Configuración (ConfigSimulacion)
4. Ejecución principal
5. Visualización y guardado de resultados
```

## Notas Importantes

- Los nombres de equipos en `equipos_monte_carlo` deben coincidir exactamente con los nombres en el Excel
- Si un equipo no se encuentra, se mostrará una advertencia pero la simulación continuará
- Los resultados se guardan automáticamente en la carpeta especificada en `carpeta_resultados`
