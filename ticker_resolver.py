"""
ticker_resolver.py — Módulo de resolución nombre→ticker

Responsabilidades:
  1. Construir y cachear un índice de tickers desde financedatabase
     (exchange BUE + tickers USA conocidos). Se regenera cada 7 días.
  2. Proveer una tabla fija de bonos soberanos argentinos con metadata.
  3. Exponer resolver_ticker(texto) → ticker canónico o None.
  4. Exponer validar_ticker_yfinance(ticker) → bool (tiene datos reales).

Este módulo no toma decisiones de negocio — solo resuelve nombres.
"""

import json
import logging
import re
import os
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CACHE_PATH = BASE_DIR / "ticker_index_cache.json"
CACHE_TTL  = timedelta(days=7)

# ──────────────────────────────────────────────────────────────────────────────
# TABLA FIJA DE BONOS SOBERANOS ARGENTINOS
# Fuente: metadata estática — no depende de yfinance ni financedatabase.
# El AgenteFundamental usa esta tabla como contexto base para el análisis
# macroeconómico. No incluye precio en tiempo real (yfinance no lo soporta).
# ──────────────────────────────────────────────────────────────────────────────
BONOS_ARGENTINOS: dict[str, dict] = {
    "AL30": {
        "nombre":       "Bono Soberano AL30",
        "vencimiento":  "2030-07-09",
        "legislacion":  "Argentina (local)",
        "moneda":       "USD",
        "cupon_anual":  "0.5% hasta 2024, luego step-up",
        "descripcion":  "Bono soberano argentino con vencimiento 2030 bajo ley local. "
                        "Mayor riesgo de restructuración que GD30 pero mayor liquidez local.",
        "riesgo_base":  "medio",
        "yfinance_ticker": None,  # No soportado
    },
    "AL35": {
        "nombre":       "Bono Soberano AL35",
        "vencimiento":  "2035-07-09",
        "legislacion":  "Argentina (local)",
        "moneda":       "USD",
        "cupon_anual":  "3.625%",
        "descripcion":  "Bono soberano argentino con vencimiento 2035 bajo ley local. "
                        "Mayor duration que AL30, más sensible a cambios de tasa.",
        "riesgo_base":  "medio",
        "yfinance_ticker": None,
    },
    "GD30": {
        "nombre":       "Bono Soberano GD30",
        "vencimiento":  "2030-07-09",
        "legislacion":  "Nueva York (extranjera)",
        "moneda":       "USD",
        "cupon_anual":  "0.5% hasta 2024, luego step-up",
        "descripcion":  "Bono soberano argentino con vencimiento 2030 bajo ley de Nueva York. "
                        "Preferido por inversores institucionales por mayor protección legal.",
        "riesgo_base":  "bajo-medio",
        "yfinance_ticker": None,
    },
    "GD35": {
        "nombre":       "Bono Soberano GD35",
        "vencimiento":  "2035-07-09",
        "legislacion":  "Nueva York (extranjera)",
        "moneda":       "USD",
        "cupon_anual":  "3.625%",
        "descripcion":  "Bono soberano argentino con vencimiento 2035 bajo ley de Nueva York. "
                        "Mayor duration, recomendado para perfiles conservadores con horizonte largo.",
        "riesgo_base":  "bajo-medio",
        "yfinance_ticker": None,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# TABLA MANUAL PARA CASOS ESPECIALES
# Cubre nombres coloquiales cortos que financedatabase no resuelve bien
# (ej: "Google" no aparece porque Alphabet está registrada en Luxembourg).
# También normaliza sufijos: GGAL.BA → GGAL para uso interno.
# ──────────────────────────────────────────────────────────────────────────────
ALIAS_MANUALES: dict[str, str] = {
    # Nombres coloquiales → ticker canónico interno
    "google":           "GOOGL",
    "alphabet":         "GOOGL",
    "amazon":           "AMZN",
    "microsoft":        "MSFT",
    "nvidia":           "NVDA",
    "tesla":            "TSLA",
    "apple":            "AAPL",
    "mercadolibre":     "MELI",
    "mercado libre":    "MELI",
    "globant":          "GLOB",
    "jpmorgan":         "JPM",
    "jp morgan":        "JPM",
    "cocacola":         "KO",
    "coca cola":        "KO",
    "coca-cola":        "KO",
    "etf s&p":          "SPY",
    "etf sp500":        "SPY",
    "s&p500":           "SPY",
    "s&p 500":          "SPY",
    "sp500":            "SPY",
    "etf nasdaq":       "QQQ",
    "nasdaq":           "QQQ",
    "qqq":              "QQQ",
    "ypf":              "YPFD",
    "yacimientos":      "YPFD",
    "pampa":            "PAMP",
    "pampa energia":    "PAMP",
    "galicia":          "GGAL",
    "grupo galicia":    "GGAL",
    "banco galicia":    "GGAL",
    "macro":            "BMA",
    "banco macro":      "BMA",
    "supervielle":      "SUPV",
    "loma negra":       "LOMA",
    "ternium":          "TXAR",
    "ternium argentina":"TXAR",
    # Bonos
    "al30":             "AL30",
    "al35":             "AL35",
    "gd30":             "GD30",
    "gd35":             "GD35",
    "bono 2030":        "GD30",  # default al de mejor protección legal
    "bono 2035":        "GD35",
}

# Tickers canónicos internos → ticker de yfinance (con sufijo si hace falta)
TICKER_A_YFINANCE: dict[str, str] = {
    # Acciones locales (necesitan .BA)
    "GGAL": "GGAL.BA",
    "YPFD": "YPFD.BA",
    "PAMP": "PAMP.BA",
    "BMA":  "BMA.BA",
    "TXAR": "TXAR.BA",
    "LOMA": "LOMA.BA",
    "SUPV": "SUPV.BA",
    # CEDEARs / USA (sin sufijo)
    "SPY":   "SPY",
    "QQQ":   "QQQ",
    "AAPL":  "AAPL",
    "MSFT":  "MSFT",
    "GOOGL": "GOOGL",
    "AMZN":  "AMZN",
    "TSLA":  "TSLA",
    "NVDA":  "NVDA",
    "KO":    "KO",
    "JPM":   "JPM",
    "MELI":  "MELI",
    "GLOB":  "GLOB",
    # Bonos: no tienen ticker de yfinance
    "AL30":  None,
    "AL35":  None,
    "GD30":  None,
    "GD35":  None,
}

# Set de tickers canónicos válidos (para validación rápida)
TICKERS_VALIDOS: set[str] = set(TICKER_A_YFINANCE.keys())


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL ÍNDICE DESDE financedatabase
# ──────────────────────────────────────────────────────────────────────────────

def _construir_indice_desde_fd() -> dict[str, str]:
    """
    Descarga todos los activos de BUE desde financedatabase y construye
    un índice {nombre_normalizado: ticker_canonico}.
    Solo se llama cuando el cache está vencido o no existe.
    """
    try:
        import financedatabase as fd
        logger.info("Construyendo índice desde financedatabase (BUE)...")
        equities = fd.Equities()
        argentinos = equities.search(exchange="BUE")

        if argentinos.empty:
            logger.warning("financedatabase no devolvió activos para BUE.")
            return {}

        indice: dict[str, str] = {}
        for ticker_raw, row in argentinos.iterrows():
            # Normalizar ticker: sacar sufijo .BA y pasar a mayúsculas
            ticker_limpio = str(ticker_raw).replace(".BA", "").upper()

            # Solo incluir tickers que estén en nuestro catálogo conocido
            # para evitar incorporar tickers inválidos o duplicados
            if ticker_limpio not in TICKERS_VALIDOS:
                continue

            nombre = str(row.get("longname", "") or "").strip().lower()
            if nombre:
                indice[nombre] = ticker_limpio

        logger.info(f"Índice construido: {len(indice)} entradas desde financedatabase.")
        return indice

    except ImportError:
        logger.warning("financedatabase no instalado. Usando solo alias manuales.")
        return {}
    except Exception as e:
        logger.error(f"Error al construir índice desde financedatabase: {e}")
        return {}


def _cache_valido() -> bool:
    """Devuelve True si el cache existe y tiene menos de 7 días."""
    if not CACHE_PATH.exists():
        return False
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        generado = datetime.fromisoformat(data.get("generado", "2000-01-01"))
        return datetime.now() - generado < CACHE_TTL
    except Exception:
        return False


def _cargar_cache() -> dict[str, str]:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("indice", {})
    except Exception:
        return {}


def _guardar_cache(indice: dict[str, str]) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "generado": datetime.now().isoformat(),
                "indice": indice
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Cache guardado en {CACHE_PATH}")
    except Exception as e:
        logger.warning(f"No se pudo guardar el cache: {e}")


# Índice en memoria — se carga una vez al importar el módulo
_indice_fd: dict[str, str] = {}

def inicializar() -> None:
    """
    Carga el índice desde cache o lo reconstruye desde financedatabase.
    Llamar una vez al arrancar el bot.
    """
    global _indice_fd
    if _cache_valido():
        _indice_fd = _cargar_cache()
        logger.info(f"Índice cargado desde cache ({len(_indice_fd)} entradas).")
    else:
        _indice_fd = _construir_indice_desde_fd()
        if _indice_fd:
            _guardar_cache(_indice_fd)


# ──────────────────────────────────────────────────────────────────────────────
# RESOLUCIÓN nombre → ticker
# ──────────────────────────────────────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    """Normaliza texto para comparación: minúsculas, sin puntuación extra."""
    texto = texto.lower().strip()
    texto = re.sub(r"[áàä]", "a", texto)
    texto = re.sub(r"[éèë]", "e", texto)
    texto = re.sub(r"[íìï]", "i", texto)
    texto = re.sub(r"[óòö]", "o", texto)
    texto = re.sub(r"[úùü]", "u", texto)
    texto = re.sub(r"[^\w\s&]", "", texto)
    return texto


def resolver_ticker(texto_activo: str) -> str | None:
    """
    Resuelve un nombre o código de activo al ticker canónico interno.
    Orden de búsqueda:
      1. Ticker exacto ya canónico (ej: "AAPL", "GGAL")
      2. Alias manuales (cubre nombres coloquiales y bonos)
      3. Índice de financedatabase (nombres largos oficiales)
      4. Búsqueda parcial en alias manuales

    Devuelve el ticker en mayúsculas o None si no se encuentra.
    """
    if not texto_activo or not texto_activo.strip():
        return None

    texto_norm = _normalizar(texto_activo)

    # 1. Ticker exacto canónico
    ticker_upper = texto_activo.strip().upper()
    if ticker_upper in TICKERS_VALIDOS:
        return ticker_upper

    # 2. Alias manuales (búsqueda exacta)
    if texto_norm in ALIAS_MANUALES:
        return ALIAS_MANUALES[texto_norm]

    # 3. Índice financedatabase (búsqueda exacta)
    if texto_norm in _indice_fd:
        return _indice_fd[texto_norm]

    # 4. Búsqueda parcial en alias manuales
    for alias, ticker in ALIAS_MANUALES.items():
        if alias in texto_norm or texto_norm in alias:
            return ticker

    # 5. Búsqueda parcial en índice financedatabase
    for nombre_fd, ticker in _indice_fd.items():
        if texto_norm in nombre_fd or nombre_fd in texto_norm:
            return ticker

    logger.info(f"resolver_ticker: no se encontró ticker para '{texto_activo}'")
    return None


def obtener_ticker_yfinance(ticker_canonico: str) -> str | None:
    """
    Convierte ticker canónico interno al formato que usa yfinance.
    Ej: GGAL → GGAL.BA | AAPL → AAPL | AL30 → None (bonos no soportados)
    """
    return TICKER_A_YFINANCE.get(ticker_canonico.upper())


def obtener_metadata_bono(ticker: str) -> dict | None:
    """
    Devuelve la metadata estática de un bono soberano argentino.
    Usada por el AgenteFundamental como contexto base.
    """
    return BONOS_ARGENTINOS.get(ticker.upper())


def es_bono(ticker: str) -> bool:
    return ticker.upper() in BONOS_ARGENTINOS


def validar_ticker_yfinance(ticker_canonico: str) -> bool:
    """
    Verifica que el ticker tiene datos reales en yfinance.
    Para bonos siempre devuelve True (sabemos que no tienen precio
    pero sí tienen metadata en nuestra tabla).
    """
    if es_bono(ticker_canonico):
        return True

    ticker_yf = obtener_ticker_yfinance(ticker_canonico)
    if not ticker_yf:
        return False

    try:
        hist = yf.Ticker(ticker_yf).history(period="5d")
        return not hist.empty
    except Exception:
        return False