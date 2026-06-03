"""
dashboard.py — Interfaz web AFMA con Streamlit

Ejecutar:
    cd C:\\Users\\marco\\Desktop\\repo_asesor_git
    streamlit run dashboard.py
"""

import json
import io
from pathlib import Path

import streamlit as st
from PIL import Image

from agente_fundamental import AgenteFundamental
from agente_tecnico import AgenteTecnico
import ticker_resolver as tr

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AFMA — Asesor Financiero",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Estilos personalizados
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Fondo general */
    .stApp {
        background-color: #0d0f14;
        color: #c8d0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2230;
    }

    /* Título principal */
    .afma-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #00d4ff;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .afma-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #4a5568;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0;
    }

    /* Divider */
    .afma-divider {
        border: none;
        border-top: 1px solid #1e2230;
        margin: 1rem 0;
    }

    /* Cards de resultado */
    .result-card {
        background: #111318;
        border: 1px solid #1e2230;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 0.5rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        line-height: 1.7;
        color: #c8d0e0;
        white-space: pre-wrap;
    }

    /* Badge de ticker */
    .ticker-badge {
        display: inline-block;
        background: #0a1628;
        color: #00d4ff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        padding: 2px 10px;
        border-radius: 4px;
        border: 1px solid #00d4ff33;
        margin: 2px;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff22, #0066ff22);
        border: 1px solid #00d4ff55;
        color: #00d4ff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 1px;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4ff33, #0066ff33);
        border-color: #00d4ff;
        color: #ffffff;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        background-color: #111318 !important;
        border: 1px solid #1e2230 !important;
        color: #c8d0e0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        border-radius: 6px !important;
    }

    /* Label de sección */
    .section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }

    /* Veredicto highlight */
    .veredicto-comprar { color: #48bb78; font-weight: 600; }
    .veredicto-vender  { color: #fc8181; font-weight: 600; }
    .veredicto-retener { color: #f6ad55; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Persistencia JSON — activos guardados
# ──────────────────────────────────────────────────────────────────────────────
ACTIVOS_PATH = Path(__file__).parent / "activos_guardados.json"


def cargar_activos() -> list[str]:
    """Lee la lista de tickers guardados. Devuelve lista vacía si no existe."""
    try:
        if ACTIVOS_PATH.exists():
            data = json.loads(ACTIVOS_PATH.read_text(encoding="utf-8"))
            return [t for t in data if isinstance(t, str) and t.strip()]
    except Exception:
        pass
    return []


def guardar_activos(activos: list[str]) -> None:
    """Escribe la lista de tickers al archivo JSON."""
    try:
        ACTIVOS_PATH.write_text(
            json.dumps(sorted(set(activos)), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        st.error(f"Error al guardar activos: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Inicialización del índice de tickers (una sola vez por sesión)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando índice de activos...")
def _inicializar_resolver():
    tr.inicializar()
    return True


@st.cache_resource(show_spinner="Iniciando agentes...")
def _inicializar_agentes():
    return AgenteFundamental(), AgenteTecnico()


# ──────────────────────────────────────────────────────────────────────────────
# Inicializar recursos
# ──────────────────────────────────────────────────────────────────────────────
_inicializar_resolver()
agente_fundamental, agente_tecnico = _inicializar_agentes()

# ──────────────────────────────────────────────────────────────────────────────
# Estado de sesión
# ──────────────────────────────────────────────────────────────────────────────
if "activos" not in st.session_state:
    st.session_state.activos = cargar_activos()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="afma-title">AFMA</p>', unsafe_allow_html=True)
    st.markdown('<p class="afma-subtitle">Asesor Financiero Multi-Agente</p>', unsafe_allow_html=True)
    st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

    # Agregar nuevo ticker
    st.markdown('<p class="section-label">Agregar activo</p>', unsafe_allow_html=True)
    nuevo_ticker = st.text_input(
        label="ticker_input",
        placeholder="Ej: AAPL, GGAL, AL30...",
        label_visibility="collapsed",
        key="nuevo_ticker_input"
    ).strip().upper()

    if st.button("＋ Guardar en lista", key="btn_guardar"):
        if nuevo_ticker:
            ticker_resuelto = tr.resolver_ticker(nuevo_ticker) or nuevo_ticker
            if ticker_resuelto not in st.session_state.activos:
                st.session_state.activos.append(ticker_resuelto)
                guardar_activos(st.session_state.activos)
                st.success(f"{ticker_resuelto} guardado.")
            else:
                st.info(f"{ticker_resuelto} ya está en la lista.")
        else:
            st.warning("Ingresá un ticker válido.")

    st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

    # Lista de activos guardados con botón de eliminar
    st.markdown('<p class="section-label">Activos guardados</p>', unsafe_allow_html=True)

    if not st.session_state.activos:
        st.caption("Ningún activo guardado aún.")
    else:
        for i, ticker in enumerate(sorted(st.session_state.activos)):
            col_t, col_x = st.columns([5, 1])
            with col_t:
                st.markdown(f'<span class="ticker-badge">{ticker}</span>', unsafe_allow_html=True)
            with col_x:
                if st.button("❌", key=f"del_{ticker}_{i}", help=f"Eliminar {ticker}"):
                    st.session_state.activos.remove(ticker)
                    guardar_activos(st.session_state.activos)
                    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# ÁREA CENTRAL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="afma-title" style="font-size:1.5rem">Panel de Análisis</p>', unsafe_allow_html=True)
st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

col_sel, col_manual = st.columns([2, 1])

with col_sel:
    st.markdown('<p class="section-label">Seleccionar de lista guardada</p>', unsafe_allow_html=True)
    opciones = ["— elegir —"] + sorted(st.session_state.activos)
    ticker_seleccionado = st.selectbox(
        label="ticker_select",
        options=opciones,
        label_visibility="collapsed",
        key="ticker_select"
    )

with col_manual:
    st.markdown('<p class="section-label">O escribir ticker manual</p>', unsafe_allow_html=True)
    ticker_manual = st.text_input(
        label="ticker_manual",
        placeholder="Ej: NVDA, MELI...",
        label_visibility="collapsed",
        key="ticker_manual_input"
    ).strip().upper()

# Determinar ticker activo — manual tiene prioridad
ticker_activo = ""
if ticker_manual:
    ticker_activo = tr.resolver_ticker(ticker_manual) or ticker_manual
elif ticker_seleccionado != "— elegir —":
    ticker_activo = ticker_seleccionado

# Mostrar ticker activo seleccionado
if ticker_activo:
    st.markdown(
        f'<p style="font-family: IBM Plex Mono, monospace; color: #00d4ff; font-size: 0.85rem;">'
        f'Activo seleccionado: <strong>{ticker_activo}</strong></p>',
        unsafe_allow_html=True
    )
else:
    st.caption("Seleccioná un activo de la lista o escribí uno manualmente.")

st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Botón principal
# ──────────────────────────────────────────────────────────────────────────────
generar = st.button(
    "⚡ GENERAR REPORTE",
    key="btn_generar",
    disabled=not ticker_activo
)

# ──────────────────────────────────────────────────────────────────────────────
# Ejecución de agentes
# ──────────────────────────────────────────────────────────────────────────────
if generar and ticker_activo:

    # Resolver ticker yfinance
    ticker_yf = tr.obtener_ticker_yfinance(ticker_activo) or ticker_activo
    es_bono   = tr.es_bono(ticker_activo)

    reporte_fundamental = ""
    reporte_tecnico     = ""
    png_bytes           = None

    with st.spinner(f"Analizando {ticker_activo}... esto puede tardar unos segundos."):

        # Agente Fundamental
        try:
            reporte_fundamental = agente_fundamental.analizar_activo(ticker_yf)
        except Exception as e:
            reporte_fundamental = f"⚠️ Error en análisis fundamental: {e}"

        # Agente Técnico
        try:
            if es_bono:
                meta = tr.obtener_metadata_bono(ticker_activo)
                reporte_tecnico = (
                    f"Análisis técnico no disponible para bonos soberanos.\n\n"
                    f"Legislación: {meta['legislacion']}\n"
                    f"Vencimiento: {meta['vencimiento']}\n"
                    f"Cupón: {meta['cupon_anual']}\n\n"
                    f"{meta['descripcion']}"
                )
            else:
                reporte_tecnico = agente_tecnico.analizar_activo(ticker_yf)
                png_bytes       = agente_tecnico.generar_grafico(ticker_yf)
        except Exception as e:
            reporte_tecnico = f"⚠️ Error en análisis técnico: {e}"

    # ── Mostrar resultados en dos columnas ────────────────────────────────
    st.markdown(
        f'<p style="font-family: IBM Plex Mono, monospace; color: #4a5568; '
        f'font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;">'
        f'Resultados — {ticker_activo}</p>',
        unsafe_allow_html=True
    )

    col_fund, col_tec = st.columns(2)

    with col_fund:
        st.markdown('<p class="section-label">Análisis Fundamental</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-card">{reporte_fundamental}</div>', unsafe_allow_html=True)

    with col_tec:
        st.markdown('<p class="section-label">Análisis Técnico</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-card">{reporte_tecnico}</div>', unsafe_allow_html=True)

        # Gráfico PNG si está disponible
        if png_bytes:
            st.markdown('<p class="section-label" style="margin-top:1rem">Gráfico (1 año)</p>', unsafe_allow_html=True)
            try:
                imagen = Image.open(io.BytesIO(png_bytes))
                st.image(imagen, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo renderizar el gráfico: {e}")

    # Botón de descarga del gráfico
    if png_bytes:
        st.download_button(
            label="⬇ Descargar gráfico PNG",
            data=png_bytes,
            file_name=f"{ticker_activo}_grafico.png",
            mime="image/png",
            key="btn_descarga"
        )