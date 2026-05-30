"""
Agente Técnico — Analista Técnico con Gráfico

Motor: Groq (Llama 3.3 70B) — reemplaza Gemini

Métodos públicos:
  - analizar_activo(simbolo) → str  (veredicto textual, siempre)
  - generar_grafico(simbolo) → bytes | None  (PNG en memoria, None si falla)

El orquestador decide cuándo llamar a cada método.
El agente no sabe nada de Telegram — solo produce texto y bytes.

Indicadores en el gráfico:
  - Precio de cierre (1 año)
  - SMA 10 y SMA 20
  - RSI 14 (panel inferior)
  - Volumen (panel inferior)

Uso directo:
    python agente_tecnico.py
"""

import os
import sys
import io
import json
import time
import logging
import requests
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")   # backend sin pantalla — esencial para Telegram
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

PERIODO_ANALISIS = "90d"   # datos para el análisis técnico (indicadores)
PERIODO_GRAFICO  = "1y"    # datos para el gráfico (1 año)


# ──────────────────────────────────────────────────────────────────────────────
# Motor Groq (igual que en agente_fundamental)
# ──────────────────────────────────────────────────────────────────────────────
def _llamar_groq(system_prompt: str, user_prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens":  400,
    }
    for intento in range(3):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                espera = 60
                logger.warning(f"Groq 429 — esperando {espera}s (intento {intento+1}/3)")
                time.sleep(espera)
                continue
            if resp.status_code != 200:
                logger.error(f"Groq HTTP {resp.status_code}: {resp.text[:200]}")
                return json.dumps({"veredicto": "ESPERAR", "motivo": f"Groq HTTP {resp.status_code}"})
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Groq error intento {intento+1}: {e}")
            if intento < 2:
                time.sleep(5)
    return json.dumps({"veredicto": "ESPERAR", "motivo": "Groq no disponible después de 3 intentos."})


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────────────────────
class AgenteTecnico:
    """
    Analista técnico con motor Groq y generación de gráfico PNG en memoria.
    """

    def __init__(self, api_key: str = None):
        """
        api_key: Groq API key. Si no se pasa, la lee de GROQ_API_KEY en .env.
        Se mantiene compatibilidad con el parámetro api_key del orquestador.
        """
        self.groq_api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY no configurada. "
                "Agregá GROQ_API_KEY=tu_key al .env"
            )

        self.system_prompt = (
            "Eres un analista técnico de inversiones puro. "
            "Evaluás liquidez (volumen), tendencia (medias móviles SMA10/SMA20) "
            "y sobrecompra/sobreventa (RSI 14).\n\n"
            "Devolvé un veredicto técnico MUY BREVE (máximo 3 líneas) indicando "
            "explícitamente si la sugerencia es COMPRA, VENTA o RETENCIÓN, "
            "justificando el riesgo de liquidez y la tendencia."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cálculo de indicadores (igual que la versión original, sin cambios)
    # ──────────────────────────────────────────────────────────────────────────
    def _calcular_indicadores(self, historial: pd.DataFrame) -> pd.DataFrame:
        historial = historial.copy()
        historial["SMA_10"]     = historial["Close"].rolling(window=10).mean()
        historial["SMA_20"]     = historial["Close"].rolling(window=20).mean()

        delta    = historial["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs       = ema_gain / ema_loss
        historial["RSI_14"]     = 100 - (100 / (1 + rs))
        historial["Vol_Avg_10"] = historial["Volume"].rolling(window=10).mean()

        return historial.dropna()

    # ──────────────────────────────────────────────────────────────────────────
    # Análisis textual (método público — interfaz con el orquestador)
    # ──────────────────────────────────────────────────────────────────────────
    def analizar_activo(self, simbolo: str) -> str:
        """
        Descarga 90 días de datos, calcula indicadores y obtiene veredicto de Groq.
        Devuelve string con el veredicto o JSON de error.
        """
        logger.info(f"[AgenteTecnico] Analizando {simbolo}...")
        try:
            ticker   = yf.Ticker(simbolo)
            historial = ticker.history(period=PERIODO_ANALISIS)

            if historial.empty:
                return json.dumps({
                    "veredicto": "ESPERAR",
                    "motivo": f"No se encontraron datos para '{simbolo}'."
                })

            hist_ind = self._calcular_indicadores(historial)

            if len(hist_ind) < 5:
                return json.dumps({
                    "veredicto": "ESPERAR",
                    "motivo": f"Datos insuficientes para '{simbolo}' (mínimo 5 días)."
                })

            ultimos = hist_ind[["Close", "SMA_10", "SMA_20", "RSI_14", "Vol_Avg_10"]].tail(5)
            datos_str = ultimos.to_string(float_format="%.2f")
            prompt    = f"Datos técnicos para {simbolo} (últimos 5 días):\n{datos_str}"

            logger.info(f"[AgenteTecnico] Enviando datos a Groq para {simbolo}...")
            respuesta = _llamar_groq(self.system_prompt, prompt, self.groq_api_key)

            # Log de uso (aproximado — Groq no siempre devuelve token count)
            logger.info(f"[AgenteTecnico] Veredicto recibido para {simbolo}.")
            return respuesta

        except Exception as e:
            logger.error(f"[AgenteTecnico] Error analizando {simbolo}: {e}", exc_info=True)
            return json.dumps({"veredicto": "ESPERAR", "motivo": str(e)})

    # ──────────────────────────────────────────────────────────────────────────
    # Generación de gráfico (método público — solo se llama cuando el usuario
    # lo pide explícitamente)
    # ──────────────────────────────────────────────────────────────────────────
    def generar_grafico(self, simbolo: str) -> bytes | None:
        """
        Genera un gráfico PNG de 1 año con precio, SMA10, SMA20, RSI y volumen.
        Devuelve los bytes del PNG o None si falla.
        El orquestador es responsable de enviarlo por Telegram.
        """
        logger.info(f"[AgenteTecnico] Generando gráfico para {simbolo}...")
        try:
            ticker   = yf.Ticker(simbolo)
            historial = ticker.history(period=PERIODO_GRAFICO)

            if historial.empty:
                logger.warning(f"Sin datos para gráfico de {simbolo}.")
                return None

            hist_ind = self._calcular_indicadores(historial)

            if len(hist_ind) < 20:
                logger.warning(f"Datos insuficientes para gráfico de {simbolo}.")
                return None

            # ── Layout: 3 paneles (precio, RSI, volumen) ──────────────────
            fig = plt.figure(figsize=(12, 8), facecolor="#1a1a2e")
            gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)

            ax_precio  = fig.add_subplot(gs[0])
            ax_rsi     = fig.add_subplot(gs[1], sharex=ax_precio)
            ax_volumen = fig.add_subplot(gs[2], sharex=ax_precio)

            fechas = hist_ind.index
            colores = {
                "fondo":   "#1a1a2e",
                "precio":  "#00d4ff",
                "sma10":   "#ffd700",
                "sma20":   "#ff6b6b",
                "rsi":     "#a8ff78",
                "volumen": "#4a90d9",
                "grid":    "#2a2a4a",
                "texto":   "#e0e0e0",
            }

            # ── Panel 1: Precio + SMA10 + SMA20 ───────────────────────────
            ax_precio.set_facecolor(colores["fondo"])
            ax_precio.plot(fechas, hist_ind["Close"],
                           color=colores["precio"], linewidth=1.5,
                           label="Precio cierre", zorder=3)
            ax_precio.plot(fechas, hist_ind["SMA_10"],
                           color=colores["sma10"], linewidth=1.2,
                           linestyle="--", label="SMA 10", zorder=2)
            ax_precio.plot(fechas, hist_ind["SMA_20"],
                           color=colores["sma20"], linewidth=1.2,
                           linestyle="--", label="SMA 20", zorder=2)
            ax_precio.fill_between(fechas, hist_ind["Close"],
                                   alpha=0.05, color=colores["precio"])
            ax_precio.set_ylabel("Precio", color=colores["texto"], fontsize=10)
            ax_precio.tick_params(colors=colores["texto"], labelbottom=False)
            ax_precio.yaxis.label.set_color(colores["texto"])
            ax_precio.legend(loc="upper left", fontsize=8,
                             facecolor=colores["fondo"],
                             labelcolor=colores["texto"],
                             framealpha=0.7)
            ax_precio.grid(color=colores["grid"], linestyle=":", linewidth=0.5)
            ax_precio.set_title(
                f"{simbolo} — Análisis Técnico (1 año)",
                color=colores["texto"], fontsize=13, fontweight="bold", pad=10
            )
            ax_precio.spines[:].set_color(colores["grid"])

            # ── Panel 2: RSI ───────────────────────────────────────────────
            ax_rsi.set_facecolor(colores["fondo"])
            ax_rsi.plot(fechas, hist_ind["RSI_14"],
                        color=colores["rsi"], linewidth=1.2, label="RSI 14")
            ax_rsi.axhline(70, color="#ff6b6b", linewidth=0.8,
                           linestyle="--", alpha=0.7)
            ax_rsi.axhline(30, color="#a8ff78", linewidth=0.8,
                           linestyle="--", alpha=0.7)
            ax_rsi.fill_between(fechas, hist_ind["RSI_14"], 70,
                                where=(hist_ind["RSI_14"] >= 70),
                                alpha=0.15, color="#ff6b6b",
                                label="Sobrecompra")
            ax_rsi.fill_between(fechas, hist_ind["RSI_14"], 30,
                                where=(hist_ind["RSI_14"] <= 30),
                                alpha=0.15, color="#a8ff78",
                                label="Sobreventa")
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_ylabel("RSI", color=colores["texto"], fontsize=9)
            ax_rsi.tick_params(colors=colores["texto"], labelbottom=False)
            ax_rsi.legend(loc="upper left", fontsize=7,
                          facecolor=colores["fondo"],
                          labelcolor=colores["texto"],
                          framealpha=0.7)
            ax_rsi.grid(color=colores["grid"], linestyle=":", linewidth=0.5)
            ax_rsi.spines[:].set_color(colores["grid"])

            # ── Panel 3: Volumen ───────────────────────────────────────────
            ax_volumen.set_facecolor(colores["fondo"])
            ax_volumen.bar(fechas, hist_ind["Volume"],
                           color=colores["volumen"], alpha=0.6,
                           width=1.5, label="Volumen")
            ax_volumen.plot(fechas, hist_ind["Vol_Avg_10"],
                            color="#ffd700", linewidth=1.0,
                            linestyle="--", label="Vol avg 10d")
            ax_volumen.set_ylabel("Volumen", color=colores["texto"], fontsize=9)
            ax_volumen.tick_params(colors=colores["texto"])
            ax_volumen.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
            )
            ax_volumen.legend(loc="upper left", fontsize=7,
                              facecolor=colores["fondo"],
                              labelcolor=colores["texto"],
                              framealpha=0.7)
            ax_volumen.grid(color=colores["grid"], linestyle=":", linewidth=0.5)
            ax_volumen.spines[:].set_color(colores["grid"])
            ax_volumen.xaxis.set_major_formatter(
                mdates.DateFormatter("%b %Y")
            )
            ax_volumen.xaxis.set_major_locator(
                mdates.MonthLocator(interval=2)
            )
            plt.setp(ax_volumen.xaxis.get_majorticklabels(),
                     rotation=30, ha="right",
                     color=colores["texto"], fontsize=8)

            # ── Exportar a bytes ───────────────────────────────────────────
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=130,
                        bbox_inches="tight", facecolor=colores["fondo"])
            plt.close(fig)
            buf.seek(0)

            logger.info(f"[AgenteTecnico] Gráfico generado para {simbolo} ({len(buf.getvalue())} bytes).")
            return buf.getvalue()

        except Exception as e:
            logger.error(f"[AgenteTecnico] Error generando gráfico para {simbolo}: {e}",
                         exc_info=True)
            plt.close("all")
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Ejecución directa para pruebas
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    agente = AgenteTecnico()

    simbolo = "NVDA"

    print(f"\n{'='*50}")
    print(f" Análisis técnico: {simbolo}")
    print("="*50)
    veredicto = agente.analizar_activo(simbolo)
    print(veredicto)

    print(f"\nGenerando gráfico para {simbolo}...")
    png = agente.generar_grafico(simbolo)
    if png:
        ruta = f"{simbolo}_grafico.png"
        with open(ruta, "wb") as f:
            f.write(png)
        print(f"Gráfico guardado en {ruta}")
    else:
        print("No se pudo generar el gráfico.")