"""
Aplicación para ajuste de curva de García Zúñiga
R = R_inf * (1 - exp(-k*t^n))
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Configuración de página para pantalla ultra wide
st.set_page_config(
    page_title="Ajuste Curva García Zúñiga",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejor uso del espacio ultra wide
st.markdown("""
    <style>
    .main > div {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stPlotlyChart {
        width: 100%;
    }
    h1 {
        text-align: center;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def garcia_zuniga(t: np.ndarray, r_inf: float, k: float, n: float) -> np.ndarray:
    """
    Modelo de García Zúñiga: R = R_inf * (1 - exp(-k*t^n))
    
    Args:
        t: Array de tiempos
        r_inf: Parámetro R_inf (recuperación asintótica)
        k: Parámetro k (constante de velocidad)
        n: Parámetro n (exponente)
    
    Returns:
        Array de valores de recuperación
    """
    return r_inf * (1 - np.exp(-k * t**n))


def fit_curve(
    tiempos: np.ndarray, 
    recuperaciones: np.ndarray
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Ajusta la curva de García Zúñiga a los datos.
    
    Args:
        tiempos: Array de tiempos
        recuperaciones: Array de recuperaciones
    
    Returns:
        Tupla con (r_inf, k, n, tiempos_fit, recuperaciones_fit)
    """
    # Estimaciones iniciales
    r_inf_guess = np.max(recuperaciones) * 1.1
    k_guess = 1.0 / (np.mean(tiempos) + 1e-6)
    n_guess = 1.0  # Valor inicial para el exponente
    
    # Ajuste de curva
    try:
        popt, _ = curve_fit(
            garcia_zuniga,
            tiempos,
            recuperaciones,
            p0=[r_inf_guess, k_guess, n_guess],
            bounds=([0, 0, 0.1], [np.inf, np.inf, 10]),
            maxfev=5000
        )
        r_inf_fit, k_fit, n_fit = popt
        
        # Generar curva suave para visualización
        t_fit = np.linspace(0, np.max(tiempos) * 1.1, 200)
        r_fit = garcia_zuniga(t_fit, r_inf_fit, k_fit, n_fit)
        
        return r_inf_fit, k_fit, n_fit, t_fit, r_fit
    except Exception as e:
        st.error(f"Error en el ajuste: {str(e)}")
        return None, None, None, None, None


def create_plot(
    tiempos: np.ndarray,
    recuperaciones: np.ndarray,
    t_fit: Optional[np.ndarray],
    r_fit: Optional[np.ndarray],
    r_inf: Optional[float],
    k: Optional[float],
    n: Optional[float],
    nombre_equipo: Optional[str] = None,
    ley_alimentacion: float = 0.0
) -> go.Figure:
    """
    Crea gráfico interactivo con Plotly.
    
    Args:
        tiempos: Array de tiempos
        recuperaciones: Array de recuperaciones
        t_fit: Tiempos para la curva ajustada
        r_fit: Recuperaciones ajustadas
        r_inf: Parámetro R_inf
        k: Parámetro k
        n: Parámetro n (exponente)
        nombre_equipo: Nombre del equipo (ej: Scavenger)
        ley_alimentacion: Ley de alimentación de cobre (%)
    """
    fig = go.Figure()
    
    # Datos experimentales de recuperación
    fig.add_trace(go.Scatter(
        x=tiempos,
        y=recuperaciones,
        mode='markers',
        name='Recuperación (exp.)',
        marker=dict(
            size=8,
            color='#1f77b4',
            line=dict(width=1.5, color='white')
        ),
        hovertemplate='Tiempo: %{x:.2f}<br>Recuperación: %{y:.4f}<extra></extra>'
    ))
    
    # Curva ajustada
    if t_fit is not None and r_fit is not None:
        fig.add_trace(go.Scatter(
            x=t_fit,
            y=r_fit,
            mode='lines',
            name='Recuperación (ajustada)',
            line=dict(width=2.5, color='#ff7f0e'),
            hovertemplate='Tiempo: %{x:.2f}<br>Recuperación: %{y:.4f}<extra></extra>'
        ))
        
        # Línea horizontal para R_inf
        if r_inf is not None:
            fig.add_hline(
                y=r_inf,
                line_dash="dash",
                line_color="gray",
                line_width=1.5,
                annotation_text=f"R_inf = {r_inf:.4f}",
                annotation_position="right"
            )
    
    # Configuración de ejes
    fig.update_xaxes(title_text='Tiempo')
    fig.update_yaxes(title_text='Recuperación (R)')
    
    # Construir título con información del equipo y ley de alimentación
    if nombre_equipo and ley_alimentacion > 0:
        titulo = f'R_inf y Cinética para {nombre_equipo} - Alimentación: {ley_alimentacion:.2f}% Cu'
        if r_inf is not None and k is not None and n is not None:
            subtitulo = f'R = R_inf × (1 - exp(-k×t^n)) | R_inf = {r_inf:.4f}, k = {k:.4f}, n = {n:.4f}'
        else:
            subtitulo = 'R = R_inf × (1 - exp(-k×t^n))'
    else:
        titulo = 'Ajuste Curva García Zúñiga: R = R_inf × (1 - exp(-k×t^n))'
        subtitulo = None
    
    fig.update_layout(
        title={
            'text': titulo + ('<br><sub>' + subtitulo + '</sub>' if subtitulo else ''),
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        hovermode='closest',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=60, r=60, t=60, b=50)
    )
    
    return fig


def main():
    """Función principal de la aplicación."""
    
    st.title("🔬 Ajuste de Curva García Zúñiga")
    st.markdown("---")
    
    # Sidebar para entrada de datos
    with st.sidebar:
        st.header("📊 Entrada de Datos")
        
        input_method = st.radio(
            "Método de entrada:",
            ["Manual", "CSV/Excel", "Pegar desde tabla"]
        )
        
        tiempos = None
        recuperaciones = None
        
        if input_method == "Manual":
            st.subheader("Ingreso Manual")
            num_points = st.number_input(
                "Número de puntos:",
                min_value=2,
                max_value=100,
                value=5,
                step=1
            )
            
            data_points = []
            for i in range(num_points):
                col1, col2 = st.columns(2)
                with col1:
                    t = st.number_input(f"Tiempo {i+1}", value=float(i+1), key=f"t_{i}")
                with col2:
                    r = st.number_input(f"Recuperación {i+1}", value=0.0, key=f"r_{i}")
                data_points.append((t, r))
            
            if data_points:
                tiempos = np.array([p[0] for p in data_points])
                recuperaciones = np.array([p[1] for p in data_points])
        
        elif input_method == "CSV/Excel":
            st.subheader("Cargar Archivo")
            uploaded_file = st.file_uploader(
                "Subir archivo CSV o Excel",
                type=['csv', 'xlsx', 'xls']
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.write("Vista previa:")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        time_col = st.selectbox("Columna de tiempo:", df.columns)
                    with col2:
                        rec_col = st.selectbox("Columna de recuperación:", df.columns)
                    
                    if st.button("Cargar datos"):
                        tiempos = df[time_col].values
                        recuperaciones = df[rec_col].values
                        st.success("Datos cargados correctamente")
                except Exception as e:
                    st.error(f"Error al leer archivo: {str(e)}")
        
        else:  # Pegar desde tabla
            st.subheader("Pegar Datos desde Excel")
            st.markdown("**Copia y pega cada columna desde Excel:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                tiempos_text = st.text_area(
                    "⏱️ Pegar Tiempos (una por línea):",
                    height=150,
                    placeholder="Ejemplo:\n1\n2\n3\n4\n5",
                    key="tiempos_paste"
                )
            
            with col2:
                recuperaciones_text = st.text_area(
                    "📈 Pegar Recuperaciones (una por línea):",
                    height=150,
                    placeholder="Ejemplo:\n0.5\n0.7\n0.85\n0.92\n0.96",
                    key="recuperaciones_paste"
                )
            
            if tiempos_text and recuperaciones_text:
                try:
                    # Parsear tiempos
                    tiempos_lines = [line.strip() for line in tiempos_text.strip().split('\n') if line.strip()]
                    tiempos_list = [float(t.replace(',', '.')) for t in tiempos_lines]
                    
                    # Parsear recuperaciones
                    recuperaciones_lines = [line.strip() for line in recuperaciones_text.strip().split('\n') if line.strip()]
                    recuperaciones_list = [float(r.replace(',', '.')) for r in recuperaciones_lines]
                    
                    # Validar que tengan la misma longitud
                    if len(tiempos_list) != len(recuperaciones_list):
                        st.warning(f"⚠️ Advertencia: Tiempos ({len(tiempos_list)} puntos) y Recuperaciones ({len(recuperaciones_list)} puntos) tienen diferente cantidad. Se usarán los primeros {min(len(tiempos_list), len(recuperaciones_list))} puntos.")
                        min_len = min(len(tiempos_list), len(recuperaciones_list))
                        tiempos_list = tiempos_list[:min_len]
                        recuperaciones_list = recuperaciones_list[:min_len]
                    
                    if tiempos_list and recuperaciones_list:
                        tiempos = np.array(tiempos_list)
                        recuperaciones = np.array(recuperaciones_list)
                        st.success(f"✅ {len(tiempos)} puntos cargados correctamente")
                        
                except ValueError as e:
                    st.error(f"❌ Error al parsear datos: Asegúrate de que todos los valores sean numéricos. Error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")
            elif tiempos_text or recuperaciones_text:
                st.info("💡 Por favor, completa ambos campos (Tiempos y Recuperaciones)")
        
        # Información del equipo y ley de alimentación
        st.markdown("---")
        st.subheader("⚙️ Información del Equipo")
        
        nombre_equipo = st.text_input(
            "🏭 Nombre del Equipo:",
            placeholder="Ej: Scavenger, Rougher, Cleaner, etc.",
            key="nombre_equipo"
        )
        
        ley_alimentacion = st.number_input(
            "📊 Ley de Alimentación Cu (%):",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.2f",
            key="ley_alimentacion"
        )
        
    # Área principal
    if tiempos is not None and recuperaciones is not None:
        # Validación de datos
        if len(tiempos) != len(recuperaciones):
            st.error("Error: Los arrays de tiempo y recuperación deben tener la misma longitud")
        elif len(tiempos) < 2:
            st.error("Error: Se necesitan al menos 2 puntos para el ajuste")
        elif np.any(tiempos < 0):
            st.error("Error: Los tiempos no pueden ser negativos")
        elif np.any(recuperaciones < 0):
            st.warning("Advertencia: Se detectaron valores negativos de recuperación")
        else:
            # Ajuste de curva
            r_inf, k, n, t_fit, r_fit = fit_curve(tiempos, recuperaciones)
            
            # Inicializar variables
            r_squared = None
            tau = None
            
            # Resultados del ajuste - Parte superior compacta
            if r_inf is not None and k is not None and n is not None:
                # Calcular R²
                r_pred = garcia_zuniga(tiempos, r_inf, k, n)
                ss_res = np.sum((recuperaciones - r_pred) ** 2)
                ss_tot = np.sum((recuperaciones - np.mean(recuperaciones)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                tau = 1 / k if k > 0 else np.inf
                
                # Mostrar información del equipo y ley de alimentación
                if nombre_equipo and ley_alimentacion > 0:
                    st.markdown(f"### 📊 Resultados para {nombre_equipo}")
                    st.markdown(f"**R_inf y cinética para una alimentación de {ley_alimentacion:.2f}% Cu en la celda {nombre_equipo}**")
                    st.markdown("---")
                
                # Métricas compactas en una fila
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.metric(
                        label="**R_inf**",
                        value=f"{r_inf:.4f}",
                        help="Recuperación asintótica máxima"
                    )
                
                with col2:
                    st.metric(
                        label="**k**",
                        value=f"{k:.4f}",
                        help="Constante de velocidad"
                    )
                
                with col3:
                    st.metric(
                        label="**n**",
                        value=f"{n:.4f}",
                        help="Exponente"
                    )
                
                with col4:
                    st.metric(
                        label="**R²**",
                        value=f"{r_squared:.4f}",
                        help="Coeficiente de determinación"
                    )
                
                with col5:
                    st.metric(
                        label="**τ**",
                        value=f"{tau:.3f}",
                        help="Tiempo característico (1/k)"
                    )
                
                with col6:
                    if ley_alimentacion > 0:
                        st.metric(
                            label="**Ley Aliment.**",
                            value=f"{ley_alimentacion:.2f}%",
                            help="Ley de alimentación de cobre"
                        )
                    else:
                        st.metric(
                            label="**Puntos**",
                            value=f"{len(tiempos)}",
                            help="Número de datos"
                        )
                
                # Ecuación compacta
                st.markdown(f"**Ecuación:** $R = {r_inf:.4f} \\times (1 - e^{{-{k:.4f} \\times t^{{{n:.4f}}}}})$")
                st.markdown("---")
            
            # Gráfico principal - más grande
            fig = create_plot(tiempos, recuperaciones, t_fit, r_fit, r_inf, k, n, nombre_equipo, ley_alimentacion)
            st.plotly_chart(fig, use_container_width=True)
            
            # Botón para descargar gráfico
            if r_inf is not None and k is not None and n is not None:
                col1, col2 = st.columns(2)
                with col1:
                    # Descargar como HTML
                    html_str = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="📥 Descargar Gráfico (HTML)",
                        data=html_str,
                        file_name=f"grafico_{nombre_equipo.lower().replace(' ', '_') if nombre_equipo else 'gyz'}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                with col2:
                    # Descargar como PNG
                    try:
                        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                        st.download_button(
                            label="📥 Descargar Gráfico (PNG)",
                            data=img_bytes,
                            file_name=f"grafico_{nombre_equipo.lower().replace(' ', '_') if nombre_equipo else 'gyz'}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.info("💡 Para descargar PNG, instala: pip install kaleido")
            
            # Tabla de datos compacta en columnas
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Datos de Recuperación**")
                df_display = pd.DataFrame({
                    'Tiempo': tiempos,
                    'R (exp.)': recuperaciones,
                    'R (ajust.)': garcia_zuniga(tiempos, r_inf, k, n) if r_inf is not None and n is not None else None
                })
                if r_inf is not None:
                    df_display['R (ajust.)'] = df_display['R (ajust.)'].round(4)
                df_display['R (exp.)'] = df_display['R (exp.)'].round(4)
                st.dataframe(df_display, use_container_width=True, hide_index=True, height=300)
            
            with col2:
                if nombre_equipo and ley_alimentacion > 0:
                    st.markdown(f"**📊 Información del Equipo**")
                    info_df = pd.DataFrame({
                        'Equipo': [nombre_equipo],
                        'Ley Aliment. (%)': [f"{ley_alimentacion:.2f}"]
                    })
                    st.dataframe(info_df, use_container_width=True, hide_index=True, height=100)
                    
                    if r_inf is not None and k is not None and n is not None and r_squared is not None:
                        st.markdown(f"**Resultados:**")
                        st.markdown(f"- R_inf = {r_inf:.4f}")
                        st.markdown(f"- k = {k:.4f}")
                        st.markdown(f"- n = {n:.4f}")
                        st.markdown(f"- R² = {r_squared:.4f}")
                else:
                    st.markdown("**💾 Exportar Resultados**")
                    
                    # Crear DataFrame con resultados
                    results_df = pd.DataFrame({
                        'Tiempo': tiempos,
                        'Recuperación_Experimental': recuperaciones,
                        'Recuperación_Ajustada': garcia_zuniga(tiempos, r_inf, k, n) if r_inf is not None and n is not None else None
                    })
                    
                    if r_inf is not None and n is not None:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar CSV",
                            data=csv,
                            file_name="resultados_ajuste_gyz.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Exportar parámetros
                        params_df = pd.DataFrame({
                            'Parámetro': ['R_inf', 'k', 'n', 'R²', 'tau'],
                            'Valor': [r_inf, k, n, r_squared, tau]
                        })
                        csv_params = params_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Parámetros",
                            data=csv_params,
                            file_name="parametros_gyz.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
            
            # Exportar con información del equipo si está disponible
            if nombre_equipo and ley_alimentacion > 0 and r_inf is not None:
                st.markdown("---")
                st.markdown("**💾 Exportar Resultados Completos**")
                results_complete = pd.DataFrame({
                    'Tiempo': tiempos,
                    'Recuperación_Experimental': recuperaciones,
                    'Recuperación_Ajustada': garcia_zuniga(tiempos, r_inf, k, n),
                    'Equipo': nombre_equipo,
                    'Ley_Alimentacion_Cu': ley_alimentacion
                })
                csv_complete = results_complete.to_csv(index=False)
                nombre_archivo = f"resultados_{nombre_equipo.lower().replace(' ', '_')}_ley_{ley_alimentacion:.2f}.csv"
                st.download_button(
                    label="📥 Descargar CSV Completo (con Info Equipo)",
                    data=csv_complete,
                    file_name=nombre_archivo,
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        # Mensaje inicial
        st.info("👈 Por favor, ingresa los datos en el panel lateral para comenzar el ajuste.")
        
        # Mostrar ejemplo
        with st.expander("📖 Ver ejemplo de uso"):
            st.markdown("""
            **Ejemplo de datos:**
            - Tiempo: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            - Recuperación: [0.39, 0.63, 0.78, 0.86, 0.91, 0.94, 0.96, 0.97, 0.98, 0.99]
            
            Estos datos deberían dar aproximadamente:
            - R_inf ≈ 1.0
            - k ≈ 0.5
            - n ≈ 1.0
            """)


if __name__ == "__main__":
    main()
