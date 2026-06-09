"""
dashboard.py — Interfaz web AFMA con Streamlit

Ejecutar:
    cd C:\\Users\\marco\\Desktop\\repo_asesor_git
    streamlit run dashboard.py
"""

import json
import io
import time
import requests
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
# Estilos
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0d0f14; color: #c8d0e0; }
    [data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #1e2230; }
    .afma-title { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; color: #00d4ff; letter-spacing: -1px; margin-bottom: 0; }
    .afma-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #4a5568; letter-spacing: 3px; text-transform: uppercase; margin-top: 0; }
    .afma-divider { border: none; border-top: 1px solid #1e2230; margin: 1rem 0; }
    .result-card { background: #111318; border: 1px solid #1e2230; border-radius: 8px; padding: 1.25rem; margin-top: 0.5rem; font-family: 'IBM Plex Sans', sans-serif; font-size: 0.9rem; line-height: 1.7; color: #c8d0e0; white-space: pre-wrap; }
    .ticker-badge { display: inline-block; background: #0a1628; color: #00d4ff; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; padding: 2px 10px; border-radius: 4px; border: 1px solid #00d4ff33; margin: 2px; }
    .tendencia-alza { color: #48bb78; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
    .tendencia-baja { color: #fc8181; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
    .tendencia-lateral { color: #f6ad55; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
    .resumen-card { background: #111318; border: 1px solid #1e2230; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; }
    .stButton > button { background: linear-gradient(135deg, #00d4ff22, #0066ff22); border: 1px solid #00d4ff55; color: #00d4ff; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; letter-spacing: 1px; border-radius: 6px; padding: 0.5rem 1.5rem; transition: all 0.2s; width: 100%; }
    .stButton > button:hover { background: linear-gradient(135deg, #00d4ff33, #0066ff33); border-color: #00d4ff; color: #ffffff; }
    .stTextInput > div > div > input, .stSelectbox > div > div { background-color: #111318 !important; border: 1px solid #1e2230 !important; color: #c8d0e0 !important; font-family: 'IBM Plex Mono', monospace !important; border-radius: 6px !important; }
    .section-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; color: #4a5568; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes — activos fijos para el informe
# ──────────────────────────────────────────────────────────────────────────────
BONOS_INFORME         = ["AL30", "AL35", "GD30", "GD35"]
INTERNACIONALES_FIJOS = ["SPY", "QQQ", "NVDA", "AAPL", "MELI"]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ──────────────────────────────────────────────────────────────────────────────
# Persistencia JSON
# ──────────────────────────────────────────────────────────────────────────────
ACTIVOS_PATH = Path(__file__).parent / "activos_guardados.json"

def cargar_activos() -> list[str]:
    try:
        if ACTIVOS_PATH.exists():
            data = json.loads(ACTIVOS_PATH.read_text(encoding="utf-8"))
            return [t for t in data if isinstance(t, str) and t.strip()]
    except Exception:
        pass
    return []

def guardar_activos(activos: list[str]) -> None:
    try:
        ACTIVOS_PATH.write_text(
            json.dumps(sorted(set(activos)), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        st.error(f"Error al guardar activos: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Inicialización (cache — una sola vez por sesión)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando índice de activos...")
def _inicializar_resolver():
    tr.inicializar()
    return True

@st.cache_resource(show_spinner="Iniciando agentes...")
def _inicializar_agentes():
    return AgenteFundamental(), AgenteTecnico()

_inicializar_resolver()
agente_fundamental, agente_tecnico = _inicializar_agentes()

# ──────────────────────────────────────────────────────────────────────────────
# Estado de sesión
# ──────────────────────────────────────────────────────────────────────────────
if "activos" not in st.session_state:
    st.session_state.activos = cargar_activos()

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de análisis rápido
# ──────────────────────────────────────────────────────────────────────────────

def _groq_call(prompt: str, groq_key: str, max_tokens: int = 200) -> str:
    """Llamada directa a Groq para síntesis corta."""
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(30)
            return "⚠️ Cuota Groq temporalmente agotada. Intentá en 30 segundos."
        if resp.status_code != 200:
            return f"⚠️ Error Groq HTTP {resp.status_code}"
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error: {e}"


def _tendencia_from_tecnico(reporte_tecnico: str) -> tuple[str, str]:
    """
    Extrae tendencia (ALCISTA/BAJISTA/LATERAL) y RSI del reporte técnico.
    Devuelve (etiqueta_html, icono).
    """
    r = reporte_tecnico.upper()
    if any(p in r for p in ["ALCIST", "COMPRA", "SUBA", "ALZA", "UPTREND"]):
        return '<span class="tendencia-alza">▲ ALCISTA</span>', "▲"
    if any(p in r for p in ["BAJIST", "VENTA", "BAJA", "DOWNTREND"]):
        return '<span class="tendencia-baja">▼ BAJISTA</span>', "▼"
    return '<span class="tendencia-lateral">→ LATERAL</span>', "→"


def _resumen_activo(ticker: str, ticker_yf: str, es_bono: bool) -> dict:
    """
    Genera el resumen rápido de un activo para el informe.
    Devuelve dict con claves: tendencia_html, icono, veredicto, reporte_tec, reporte_fund.
    """
    reporte_tec  = ""
    reporte_fund = ""

    if es_bono:
        meta = tr.obtener_metadata_bono(ticker)
        reporte_tec  = f"Bono soberano — {meta['legislacion']} — vence {meta['vencimiento']}"
        reporte_fund = agente_fundamental.analizar_activo("ARGT")  # proxy macro
        tendencia_html, icono = '<span class="tendencia-lateral">→ BONO</span>', "→"
    else:
        reporte_tec  = agente_tecnico.analizar_activo(ticker_yf)
        reporte_fund = agente_fundamental.analizar_activo(ticker_yf)
        tendencia_html, icono = _tendencia_from_tecnico(reporte_tec)

    return {
        "tendencia_html": tendencia_html,
        "icono":          icono,
        "reporte_tec":    reporte_tec,
        "reporte_fund":   reporte_fund,
    }


def _render_activo_expandible(ticker: str, resumen: dict, key_prefix: str):
    """
    Renderiza un activo como card expandible con botón de gráfico lazy.
    """
    ticker_yf = tr.obtener_ticker_yfinance(ticker)
    es_bono   = tr.es_bono(ticker)

    with st.expander(
        f"{resumen['icono']}  {ticker}",
        expanded=False
    ):
        st.markdown(
            f'<div class="resumen-card">'
            f'<b>Tendencia:</b> {resumen["tendencia_html"]}<br><br>'
            f'<b>Análisis Técnico:</b><br>{resumen["reporte_tec"]}<br><br>'
            f'<b>Análisis Fundamental:</b><br>{resumen["reporte_fund"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Botón de gráfico lazy — solo plotea cuando se clickea
        if not es_bono and ticker_yf:
            if st.button(f"📈 Ver gráfico de {ticker}", key=f"{key_prefix}_graf_{ticker}"):
                with st.spinner(f"Graficando {ticker}..."):
                    png = agente_tecnico.generar_grafico(ticker_yf)
                if png:
                    st.image(Image.open(io.BytesIO(png)), use_container_width=True)
                    st.download_button(
                        label="⬇ Descargar PNG",
                        data=png,
                        file_name=f"{ticker}_grafico.png",
                        mime="image/png",
                        key=f"{key_prefix}_dl_{ticker}"
                    )
                else:
                    st.warning("No se pudo generar el gráfico.")


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="afma-title">AFMA</p>', unsafe_allow_html=True)
    st.markdown('<p class="afma-subtitle">Asesor Financiero Multi-Agente</p>', unsafe_allow_html=True)
    st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

    st.markdown('<p class="section-label">Agregar activo</p>', unsafe_allow_html=True)
    nuevo_ticker = st.text_input(
        label="ticker_input", placeholder="Ej: AAPL, GGAL, AL30...",
        label_visibility="collapsed", key="nuevo_ticker_input"
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
# ÁREA CENTRAL — Tab 1: Panel de Análisis individual
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="afma-title" style="font-size:1.5rem">AFMA Dashboard</p>', unsafe_allow_html=True)
st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

tab_analisis, tab_informe = st.tabs(["⚡ Análisis Individual", "📋 Informe de Cartera"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Análisis individual (idéntico al dashboard original)
# ──────────────────────────────────────────────────────────────────────────────
with tab_analisis:
    col_sel, col_manual = st.columns([2, 1])

    with col_sel:
        st.markdown('<p class="section-label">Seleccionar de lista guardada</p>', unsafe_allow_html=True)
        opciones = ["— elegir —"] + sorted(st.session_state.activos)
        ticker_seleccionado = st.selectbox(
            label="ticker_select", options=opciones,
            label_visibility="collapsed", key="ticker_select"
        )

    with col_manual:
        st.markdown('<p class="section-label">O escribir ticker manual</p>', unsafe_allow_html=True)
        ticker_manual = st.text_input(
            label="ticker_manual", placeholder="Ej: NVDA, MELI...",
            label_visibility="collapsed", key="ticker_manual_input"
        ).strip().upper()

    ticker_activo = ""
    if ticker_manual:
        ticker_activo = tr.resolver_ticker(ticker_manual) or ticker_manual
    elif ticker_seleccionado != "— elegir —":
        ticker_activo = ticker_seleccionado

    if ticker_activo:
        st.markdown(
            f'<p style="font-family:IBM Plex Mono,monospace;color:#00d4ff;font-size:0.85rem;">'
            f'Activo seleccionado: <strong>{ticker_activo}</strong></p>',
            unsafe_allow_html=True
        )
    else:
        st.caption("Seleccioná un activo de la lista o escribí uno manualmente.")

    st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

    generar = st.button("⚡ GENERAR REPORTE", key="btn_generar", disabled=not ticker_activo)

    if generar and ticker_activo:
        ticker_yf = tr.obtener_ticker_yfinance(ticker_activo) or ticker_activo
        es_bono   = tr.es_bono(ticker_activo)
        reporte_fundamental = reporte_tecnico = ""
        png_bytes = None

        with st.spinner(f"Analizando {ticker_activo}..."):
            try:
                reporte_fundamental = agente_fundamental.analizar_activo(ticker_yf)
            except Exception as e:
                reporte_fundamental = f"⚠️ Error fundamental: {e}"
            try:
                if es_bono:
                    meta = tr.obtener_metadata_bono(ticker_activo)
                    reporte_tecnico = (
                        f"Análisis técnico no disponible para bonos soberanos.\n\n"
                        f"Legislación: {meta['legislacion']}\n"
                        f"Vencimiento: {meta['vencimiento']}\n"
                        f"Cupón: {meta['cupon_anual']}\n\n{meta['descripcion']}"
                    )
                else:
                    reporte_tecnico = agente_tecnico.analizar_activo(ticker_yf)
                    png_bytes       = agente_tecnico.generar_grafico(ticker_yf)
            except Exception as e:
                reporte_tecnico = f"⚠️ Error técnico: {e}"

        st.markdown(
            f'<p style="font-family:IBM Plex Mono,monospace;color:#4a5568;'
            f'font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;">'
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
            if png_bytes:
                st.markdown('<p class="section-label" style="margin-top:1rem">Gráfico (1 año)</p>', unsafe_allow_html=True)
                try:
                    st.image(Image.open(io.BytesIO(png_bytes)), use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo renderizar el gráfico: {e}")
        if png_bytes:
            st.download_button(
                label="⬇ Descargar gráfico PNG", data=png_bytes,
                file_name=f"{ticker_activo}_grafico.png", mime="image/png",
                key="btn_descarga"
            )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Informe de cartera
# ──────────────────────────────────────────────────────────────────────────────
with tab_informe:
    st.markdown('<p class="section-label">Informe de Cartera</p>', unsafe_allow_html=True)
    st.caption("Repaso rápido de tendencia y momentum. Expandí cada activo para ver el detalle y graficá cuando lo necesites.")
    st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)

    subtab_favoritos, subtab_bonos, subtab_internacional = st.tabs([
        "⭐ Mis Favoritos",
        "🇦🇷 Bonos Argentinos",
        "🌐 Radar Internacional"
    ])

    # ── Subtab: Mis Favoritos ─────────────────────────────────────────────
    with subtab_favoritos:
        if not st.session_state.activos:
            st.info("No tenés activos guardados. Agregá algunos desde la barra lateral.")
        else:
            activos_str = ", ".join(sorted(st.session_state.activos))
            st.caption(f"Activos en cartera: {activos_str}")

            if st.button("🔄 Generar repaso de favoritos", key="btn_repaso"):
                resultados_fav = {}
                total = len(st.session_state.activos)
                progress = st.progress(0, text="Iniciando análisis...")

                for idx, ticker in enumerate(sorted(st.session_state.activos)):
                    progress.progress(
                        (idx) / total,
                        text=f"Analizando {ticker} ({idx+1}/{total})..."
                    )
                    ticker_yf = tr.obtener_ticker_yfinance(ticker)
                    es_bono   = tr.es_bono(ticker)
                    try:
                        resultados_fav[ticker] = _resumen_activo(
                            ticker, ticker_yf or ticker, es_bono
                        )
                    except Exception as e:
                        resultados_fav[ticker] = {
                            "tendencia_html": '<span class="tendencia-lateral">? ERROR</span>',
                            "icono": "?",
                            "reporte_tec":  f"Error: {e}",
                            "reporte_fund": "",
                        }
                    time.sleep(0.5)  # pausa entre llamadas a Groq

                progress.progress(1.0, text="Análisis completado.")
                st.session_state["resultados_fav"] = resultados_fav

            # Renderizar resultados guardados en session_state
            if "resultados_fav" in st.session_state:
                for ticker, resumen in st.session_state["resultados_fav"].items():
                    _render_activo_expandible(ticker, resumen, key_prefix="fav")

    # ── Subtab: Bonos Argentinos ──────────────────────────────────────────
    with subtab_bonos:
        st.caption("Bonos soberanos argentinos. El análisis técnico no está disponible para bonos — se usa el ETF ARGT como proxy macro.")

        if st.button("🔄 Analizar bonos", key="btn_bonos"):
            resultados_bonos = {}
            progress_b = st.progress(0, text="Analizando bonos...")

            for idx, ticker in enumerate(BONOS_INFORME):
                progress_b.progress(
                    idx / len(BONOS_INFORME),
                    text=f"Analizando {ticker}..."
                )
                try:
                    resultados_bonos[ticker] = _resumen_activo(ticker, None, es_bono=True)
                except Exception as e:
                    resultados_bonos[ticker] = {
                        "tendencia_html": '<span class="tendencia-lateral">? ERROR</span>',
                        "icono": "?",
                        "reporte_tec":  f"Error: {e}",
                        "reporte_fund": "",
                    }
                time.sleep(0.5)

            progress_b.progress(1.0, text="Análisis completado.")
            st.session_state["resultados_bonos"] = resultados_bonos

        if "resultados_bonos" in st.session_state:
            for ticker, resumen in st.session_state["resultados_bonos"].items():
                meta = tr.obtener_metadata_bono(ticker)
                with st.expander(f"→  {ticker} — {meta['nombre']}", expanded=False):
                    st.markdown(
                        f'<div class="resumen-card">'
                        f'<b>Legislación:</b> {meta["legislacion"]} &nbsp;|&nbsp; '
                        f'<b>Vencimiento:</b> {meta["vencimiento"]} &nbsp;|&nbsp; '
                        f'<b>Cupón:</b> {meta["cupon_anual"]}<br><br>'
                        f'<b>Contexto macro (proxy ARGT):</b><br>{resumen["reporte_fund"]}<br><br>'
                        f'<b>Descripción:</b><br>{meta["descripcion"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── Subtab: Radar Internacional ───────────────────────────────────────
    with subtab_internacional:
        st.caption(
            "Screening de oportunidades internacionales. "
            "Groq analiza el contexto macro actual y selecciona 2 activos con señales de entrada relevantes."
        )

        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)
        groq_key = os.getenv("GROQ_API_KEY", "")

        if st.button("🔍 Buscar oportunidades", key="btn_screening"):
            with st.spinner("Analizando mercado internacional..."):

                # Paso 1: analizar los activos del universo
                datos_universo = {}
                for ticker in INTERNACIONALES_FIJOS:
                    ticker_yf = tr.obtener_ticker_yfinance(ticker) or ticker
                    try:
                        tec  = agente_tecnico.analizar_activo(ticker_yf)
                        fund = agente_fundamental.analizar_activo(ticker_yf)
                        datos_universo[ticker] = f"TÉCNICO: {tec}\nFUNDAMENTAL: {fund}"
                    except Exception as e:
                        datos_universo[ticker] = f"Error: {e}"
                    time.sleep(0.5)

                # Paso 2: Groq selecciona los 2 mejores con fundamento
                contexto_universo = "\n\n".join(
                    f"--- {t} ---\n{v}" for t, v in datos_universo.items()
                )
                prompt_screening = (
                    "Sos un analista financiero senior. Analizás el siguiente universo de activos "
                    "internacionales disponibles como CEDEARs en Argentina:\n\n"
                    f"{contexto_universo}\n\n"
                    "Seleccioná exactamente 2 activos con la mejor combinación de:\n"
                    "1. Tendencia técnica alcista o momentum positivo\n"
                    "2. Fundamento sólido o catalizador de corto/mediano plazo\n"
                    "3. Relación riesgo/retorno favorable para un inversor conservador-moderado\n\n"
                    "Para cada activo seleccionado respondé con este formato exacto:\n"
                    "TICKER: [ticker]\n"
                    "SEÑAL: [ALCISTA/LATERAL/BAJISTA]\n"
                    "CATALIZADOR: [1 línea — por qué ahora]\n"
                    "RIESGO PRINCIPAL: [1 línea — qué podría salir mal]\n\n"
                    "Sé preciso y concreto. Sin preamble ni conclusiones genéricas."
                )

                screening = _groq_call(prompt_screening, groq_key, max_tokens=400)
                st.session_state["screening_result"]  = screening
                st.session_state["screening_universo"] = datos_universo

        if "screening_result" in st.session_state:
            st.markdown('<p class="section-label">Oportunidades seleccionadas</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-card">{st.session_state["screening_result"]}</div>',
                unsafe_allow_html=True
            )

            # Detalle expandible de cada activo del universo analizado
            st.markdown('<hr class="afma-divider">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Detalle del universo analizado</p>', unsafe_allow_html=True)

            for ticker, detalle in st.session_state["screening_universo"].items():
                ticker_yf = tr.obtener_ticker_yfinance(ticker) or ticker
                with st.expander(f"{ticker}", expanded=False):
                    st.markdown(f'<div class="result-card">{detalle}</div>', unsafe_allow_html=True)
                    if st.button(f"📈 Ver gráfico de {ticker}", key=f"scr_graf_{ticker}"):
                        with st.spinner(f"Graficando {ticker}..."):
                            png = agente_tecnico.generar_grafico(ticker_yf)
                        if png:
                            st.image(Image.open(io.BytesIO(png)), use_container_width=True)
                            st.download_button(
                                label="⬇ Descargar PNG", data=png,
                                file_name=f"{ticker}_grafico.png", mime="image/png",
                                key=f"scr_dl_{ticker}"
                            )
                        else:
                            st.warning("No se pudo generar el gráfico.")