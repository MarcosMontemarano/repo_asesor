"""
Agente Orquestador - Bot de Telegram

Arquitectura:
  - Llama (Ollama): extrae entidades del lenguaje natural
  - Python: toda la lógica de decisión (CHAT vs EXEC)
  - ticker_resolver: resolución nombre→ticker con financedatabase
  - iol_client: precio en tiempo real desde IOL
  - AgenteFundamental: análisis de noticias con Groq + cadena de fuentes
  - AgenteTecnico: análisis técnico con Groq + gráfico PNG opcional

Uso:
    python agente_orquestador.py
"""

import os
import re
import sys
import logging
import json
import time
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

import ticker_resolver as tr
from iol_client import obtener_precio as _obtener_precio_iol
from agente_fundamental import AgenteFundamental
from agente_tecnico import AgenteTecnico

load_dotenv(override=True)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ──────────────────────────────────────────────────────────────────────────────
# Groq — síntesis final del veredicto
# ──────────────────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

def _llamar_groq_sintesis(prompt: str, groq_api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens":  400,
    }
    for intento in range(3):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                logging.warning(f"Groq 429 síntesis — esperando 60s (intento {intento+1}/3)")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                return f"Error en síntesis (HTTP {resp.status_code})"
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logging.error(f"Groq síntesis error intento {intento+1}: {e}")
            if intento < 2:
                time.sleep(5)
    return "No se pudo generar el veredicto final. Intentá de nuevo."


# ──────────────────────────────────────────────────────────────────────────────
# Prompt extractor de Ollama
# ──────────────────────────────────────────────────────────────────────────────
EXTRACTOR_PROMPT = """
Sos un extractor de entidades financieras. Tu único trabajo es leer el
mensaje del usuario y devolver un JSON con exactamente estos campos:

{
  "texto_activo": "nombre o código del activo mencionado (string, vacío si no hay ninguno)",
  "capital":      "solo los dígitos del monto mencionado (string, vacío si no hay monto)",
  "riesgo":       "bajo, medio o alto (inferilo del contexto, vacío si no hay señal)",
  "cantidad_activos": 1,
  "intencion":    "analizar, distribuir, recomendar, consulta, grafico, otro"
}

Reglas estrictas:
- "texto_activo": devolvé el nombre TAL COMO lo escribió el usuario.
- "capital": SOLO dígitos. "100k" → "100000". "$50.000" → "50000". Vacío si no hay monto.
- "riesgo": "quiero algo seguro" → "bajo". "no me importa arriesgar" → "alto". Sin señal → vacío.
- "cantidad_activos": cuántos activos DISTINTOS mencionó. "SPY y AAPL" → 2. Los montos NO son activos.
- "intencion":
    "analizar"    → quiere analizar un activo específico
    "distribuir"  → quiere repartir capital entre varios activos
    "recomendar"  → pide sugerencias sin tener un activo en mente
    "consulta"    → pregunta educativa, de timing, noticias
    "grafico"     → pide ver el gráfico de precios de un activo
    "otro"        → cualquier otra cosa

Solo JSON puro. Sin texto fuera del objeto JSON. Sin explicaciones.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Palabras clave para detección de gráfico y reset
# ──────────────────────────────────────────────────────────────────────────────
PALABRAS_GRAFICO = [
    "gráfico", "grafico", "graficá", "grafica", "chart",
    "mostrar precio", "ver precio", "curva", "evolución", "evolucion"
]

FRASES_RESET = [
    "otro análisis", "nueva inversión", "nuevo análisis",
    "empecemos de nuevo", "empezar de nuevo", "quiero hacer otro",
    "análisis distinto", "inversión distinta", "cambiemos de tema",
    "olvidá lo anterior", "borrá todo", "resetear",
]

VALORES_VACIOS = {"", "0", "vacío", "vacio", "none", "null", "n/a", "no hay"}


def _detecta_reset(texto: str) -> bool:
    t = texto.lower()
    return any(frase in t for frase in FRASES_RESET)


def _detecta_grafico(texto: str) -> bool:
    t = texto.lower()
    return any(p in t for p in PALABRAS_GRAFICO)


def _extraer_numero(valor) -> str:
    if valor is None:
        return ""
    solo = re.sub(r"[^\d]", "", str(valor))
    return "" if solo == "0" else solo


def _limpiar_texto_activo(valor: str) -> str:
    return "" if valor.strip().lower() in VALORES_VACIOS else valor.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Contexto persistente por usuario
# ──────────────────────────────────────────────────────────────────────────────
class ContextoUsuario:
    def __init__(self):
        self.historial:         list[str] = []
        self.capital:           str = ""
        self.riesgo:            str = ""
        self.ticker_pendiente:  str = ""
        self.ticker_en_curso:   str = ""  # ticker válido del turno actual

    def actualizar_desde_entidades(self, entidades: dict) -> None:
        capital_nuevo = _extraer_numero(entidades.get("capital", ""))
        riesgo_nuevo  = str(entidades.get("riesgo", "")).strip().lower()
        if capital_nuevo:
            self.capital = capital_nuevo
        if riesgo_nuevo in ("bajo", "medio", "alto"):
            self.riesgo = riesgo_nuevo

    def resetear_contexto(self) -> None:
        self.capital         = ""
        self.riesgo          = ""
        self.ticker_pendiente = ""
        self.ticker_en_curso  = ""
        self.historial        = []

    def resumen(self) -> str:
        return f"capital={self.capital or '?'}, riesgo={self.riesgo or '?'}"


# ──────────────────────────────────────────────────────────────────────────────
# Orquestador
# ──────────────────────────────────────────────────────────────────────────────
class AgenteOrquestador:

    def __init__(self, telegram_token: str, groq_api_key: str):
        if not telegram_token:
            raise ValueError("TELEGRAM_TOKEN no configurado.")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY no configurado.")

        self.groq_api_key = groq_api_key

        logging.info("Inicializando índice de tickers...")
        tr.inicializar()
        logging.info("Índice listo.")

        self.application = ApplicationBuilder().token(telegram_token).build()
        self._contextos: dict[int, ContextoUsuario] = {}

        self.agente_fundamental = AgenteFundamental()
        self.agente_tecnico     = AgenteTecnico()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Extracción de entidades vía Ollama
    # ──────────────────────────────────────────────────────────────────────────
    def _extraer_entidades(self, chat_id: int, historial: list[str], texto: str) -> dict:
        historial_reciente = historial[-6:]
        prompt = (
            EXTRACTOR_PROMPT
            + "\n\n--- Historial reciente ---\n"
            + "\n".join(historial_reciente)
            + f"\n\nMensaje actual: {texto}"
            + "\n\nDevolvé el JSON ahora:"
        )
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        resp = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
        raw = resp.json()["response"].strip()
        logging.info(f"[chat_id={chat_id}] Entidades: {raw}")
        if not raw or raw == "{}":
            return {}
        datos = json.loads(raw)
        datos["capital"]      = _extraer_numero(datos.get("capital", ""))
        datos["texto_activo"] = _limpiar_texto_activo(str(datos.get("texto_activo", "")))
        return datos

    # ──────────────────────────────────────────────────────────────────────────
    # Lógica de decisión — Python puro
    # ──────────────────────────────────────────────────────────────────────────
    def _decidir(self, entidades: dict, contexto: ContextoUsuario) -> tuple[str, dict]:
        texto_activo = str(entidades.get("texto_activo", "")).strip()
        intencion    = str(entidades.get("intencion", "otro")).lower()
        cantidad     = int(entidades.get("cantidad_activos", 1) or 1)

        contexto.actualizar_desde_entidades(entidades)

        # Gráfico — intención explícita o palabras clave ya detectadas antes
        if intencion == "grafico":
            ticker = tr.resolver_ticker(texto_activo) if texto_activo else contexto.ticker_pendiente
            if ticker:
                return "GRAFICO", {"ticker": ticker}
            return "CHAT", {"motivo": "falta_ticker_grafico"}

        # Intenciones que nunca son EXEC
        if intencion in ("distribuir", "recomendar", "consulta"):
            return "CHAT", {"motivo": intencion}

        # Múltiples activos
        if cantidad > 1:
            return "CHAT", {"motivo": "multiples_activos"}

        # Resolver ticker
        ticker = tr.resolver_ticker(texto_activo) if texto_activo else None

        # Ticker no encontrado en catálogo
        if texto_activo and not ticker:
            return "CHAT", {"motivo": "ticker_no_encontrado", "texto": texto_activo}

        # Guardar ticker en curso aunque no ejecutemos aún
        if ticker:
            contexto.ticker_en_curso = ticker

        # Ticker nuevo con contexto existente → pedir confirmación
        if (
            ticker
            and contexto.ticker_pendiente
            and ticker != contexto.ticker_pendiente
            and contexto.capital
            and contexto.riesgo
        ):
            return "CHAT", {
                "motivo":  "confirmar_contexto",
                "ticker":  ticker,
                "capital": contexto.capital,
                "riesgo":  contexto.riesgo,
            }

        # Recuperar ticker del contexto si el mensaje no trae uno
        if not ticker:
            ticker = contexto.ticker_en_curso or None

        if not ticker:
            return "CHAT", {"motivo": "falta_ticker"}
        if not contexto.capital:
            return "CHAT", {"motivo": "falta_capital", "ticker": ticker}
        if not contexto.riesgo:
            return "CHAT", {"motivo": "falta_riesgo", "ticker": ticker}

        contexto.ticker_pendiente = ticker
        return "EXEC", {
            "ticker":  ticker,
            "capital": contexto.capital,
            "riesgo":  contexto.riesgo,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Mensajes CHAT con texto fijo o Groq
    # ──────────────────────────────────────────────────────────────────────────
    def _generar_mensaje_chat(self, motivo: str, datos: dict, contexto: ContextoUsuario) -> str:
        if motivo == "falta_capital":
            ticker = datos.get("ticker", "")
            meta   = tr.obtener_metadata_bono(ticker)
            nombre = meta["nombre"] if meta else ticker
            return f"Perfecto, vamos a analizar {nombre}. ¿Con cuánto capital contás?"

        if motivo == "falta_riesgo":
            ticker = datos.get("ticker", "")
            return (
                f"Ya tengo el activo y el capital. "
                f"¿Cuál es tu tolerancia al riesgo para {ticker}? "
                f"Podés elegir: bajo, medio o alto."
            )

        if motivo == "falta_ticker":
            return "¿Qué activo específico querés que analice primero?"

        if motivo == "falta_ticker_grafico":
            return "¿De qué activo querés ver el gráfico?"

        if motivo == "ticker_no_encontrado":
            texto = datos.get("texto", "")
            return (
                f"No encontré '{texto}' en el catálogo de IOL. "
                f"¿Podés confirmar el nombre o ticker exacto?"
            )

        if motivo == "confirmar_contexto":
            ticker  = datos.get("ticker", "")
            capital = datos.get("capital", "")
            riesgo  = datos.get("riesgo", "")
            meta    = tr.obtener_metadata_bono(ticker)
            nombre  = meta["nombre"] if meta else ticker
            return (
                f"Antes de analizar {nombre}, confirmame: "
                f"¿seguimos con ${capital} y riesgo {riesgo}, "
                f"o querés cambiar algún parámetro?"
            )

        # Casos creativos → Groq
        prompt = (
            "Sos AFMA, asesor financiero de IOL Argentina. "
            "Respondé de forma natural, cálida y sin jerga técnica.\n\n"
            f"Historial reciente: {contexto.historial[-4:]}\n"
            f"Contexto actual: {contexto.resumen()}\n"
            f"Motivo: {motivo}\n"
            f"Datos: {datos}\n\n"
            "Respondé en máximo 4 líneas."
        )
        try:
            return _llamar_groq_sintesis(prompt, self.groq_api_key)
        except Exception as e:
            logging.error(f"Error generando CHAT con Groq: {e}")
            return "¿Podés contarme más sobre lo que buscás?"

    # ──────────────────────────────────────────────────────────────────────────
    # Handlers de Telegram
    # ──────────────────────────────────────────────────────────────────────────
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self._contextos[chat_id] = ContextoUsuario()
        logging.info(f"[chat_id={chat_id}] /start")
        await update.message.reply_text(
            "Hola! Soy AFMA, tu asesor financiero para IOL (InvertirOnline).\n\n"
            "Podés hablarme de forma natural. Por ejemplo:\n"
            "  \"Tengo $100.000 y quiero invertir en CEDEARs\"\n"
            "  \"Qué bonos en dólares recomendás para riesgo bajo?\"\n"
            "  \"Quiero analizar YPF con riesgo alto\"\n"
            "  \"Mostrá el gráfico de NVDA\"\n\n"
            "En qué te ayudo hoy?"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id       = update.effective_chat.id
        texto_usuario = update.message.text

        if chat_id not in self._contextos:
            self._contextos[chat_id] = ContextoUsuario()
        contexto = self._contextos[chat_id]

        # Reset explícito
        if _detecta_reset(texto_usuario):
            contexto.resetear_contexto()
            await update.message.reply_text(
                "Perfecto, empezamos de nuevo. "
                "Contame qué activo te interesa y con cuánto capital."
            )
            return

        # Detección de gráfico antes de llamar a Ollama (más rápido y robusto)
        if _detecta_grafico(texto_usuario):
            entidades = {"intencion": "grafico", "texto_activo": "", "capital": "", "riesgo": "", "cantidad_activos": 1}
            # Intentar extraer ticker del texto igualmente
            try:
                entidades_ollama = self._extraer_entidades(chat_id, contexto.historial, texto_usuario)
                if entidades_ollama.get("texto_activo"):
                    entidades["texto_activo"] = entidades_ollama["texto_activo"]
            except Exception:
                pass
        else:
            contexto.historial.append(f"Usuario: {texto_usuario}")
            try:
                entidades = self._extraer_entidades(chat_id, contexto.historial, texto_usuario)
                if not entidades:
                    await update.message.reply_text("No pude procesar tu mensaje. ¿Podés repetirlo?")
                    return
            except requests.exceptions.ConnectionError:
                logging.error(f"[chat_id={chat_id}] Ollama no disponible.")
                await update.message.reply_text(
                    "El motor de IA local no está disponible. "
                    "Por favor contactá al administrador."
                )
                return
            except Exception as e:
                logging.error(f"[chat_id={chat_id}] Error extrayendo entidades: {e}", exc_info=True)
                await update.message.reply_text("Ocurrió un error. Intentá de nuevo.")
                return

        try:
            accion, datos = self._decidir(entidades, contexto)
            logging.info(f"[chat_id={chat_id}] accion={accion} datos={datos} ctx={contexto.resumen()}")

            # ── GRÁFICO ───────────────────────────────────────────────────
            if accion == "GRAFICO":
                ticker    = datos["ticker"]
                ticker_yf = tr.obtener_ticker_yfinance(ticker)
                simbolo   = ticker_yf or ticker

                await update.message.reply_text(f"Generando gráfico de {ticker}... un momento 📊")

                png = self.agente_tecnico.generar_grafico(simbolo)
                if png:
                    await update.message.reply_photo(
                        photo=png,
                        caption=f"📈 {ticker} — Análisis Técnico (1 año)\nSMA10 | SMA20 | RSI | Volumen"
                    )
                else:
                    await update.message.reply_text(
                        f"No pude generar el gráfico para {ticker}. "
                        f"Es posible que no haya datos disponibles."
                    )
                return

            # ── CHAT ──────────────────────────────────────────────────────
            if accion == "CHAT":
                motivo  = datos.get("motivo", "otro")
                mensaje = self._generar_mensaje_chat(motivo, datos, contexto)
                contexto.historial.append(f"Asesor: {mensaje}")
                await update.message.reply_text(mensaje)

            # ── EXEC ──────────────────────────────────────────────────────
            elif accion == "EXEC":
                ticker  = datos["ticker"]
                capital = datos["capital"]
                riesgo  = datos["riesgo"]

                await update.message.reply_text(
                    f"Analizando {ticker} — capital ${capital}, riesgo {riesgo}... un momento 🔍"
                )

                # Precio IOL y nominales
                precio_actual    = _obtener_precio_iol(ticker)
                nominales_reales = 0
                if precio_actual and precio_actual > 0:
                    nominales_reales = int(float(capital) // precio_actual)

                # Ticker para yfinance
                ticker_yf = tr.obtener_ticker_yfinance(ticker)
                simbolo   = ticker_yf or ticker

                # Agente Fundamental
                reporte_fundamental = "Análisis fundamental no disponible."
                try:
                    logging.info(f"[EXEC] AgenteFundamental → {simbolo}")
                    reporte_fundamental = self.agente_fundamental.analizar_activo(simbolo)
                except Exception as e:
                    logging.error(f"AgenteFundamental error: {e}", exc_info=True)

                time.sleep(1)  # pausa entre llamadas a Groq

                # Agente Técnico
                reporte_tecnico = "Análisis técnico no disponible."
                try:
                    logging.info(f"[EXEC] AgenteTecnico → {simbolo}")
                    if tr.es_bono(ticker):
                        meta_bono = tr.obtener_metadata_bono(ticker)
                        reporte_tecnico = (
                            f"Análisis técnico no disponible para bonos soberanos. "
                            f"Contexto: {meta_bono['descripcion']}"
                        )
                    else:
                        reporte_tecnico = self.agente_tecnico.analizar_activo(simbolo)
                except Exception as e:
                    logging.error(f"AgenteTecnico error: {e}", exc_info=True)

                # Síntesis final con Groq
                precio_str = f"${precio_actual:.2f} ARS" if precio_actual else "no disponible desde IOL"
                prompt_final = (
                    f"Sos el Asesor Financiero final de IOL. Cliente amateur.\n"
                    f"Activo: {ticker} | Capital: ${capital} | Riesgo: {riesgo}\n"
                    f"Precio actual en IOL: {precio_str}\n"
                    f"Nominales que puede comprar: {nominales_reales}\n\n"
                    f"Análisis Fundamental:\n{reporte_fundamental}\n\n"
                    f"Análisis Técnico:\n{reporte_tecnico}\n\n"
                    "Sintetizá la decisión SIN jerga técnica. "
                    "Usá EXACTAMENTE este formato:\n\n"
                    f"Veredicto: [COMPRAR / VENDER / RETENER] — {nominales_reales} nominales\n"
                    "Motivo: [máximo 2 líneas simples]\n\n"
                    "Al final preguntá si quiere profundizar o analizar otro activo."
                )

                mensaje_final = _llamar_groq_sintesis(prompt_final, self.groq_api_key)
                contexto.historial.append(f"Asesor: {mensaje_final}")
                await update.message.reply_text(mensaje_final)

        except json.JSONDecodeError as e:
            logging.error(f"[chat_id={chat_id}] JSONDecodeError: {e}", exc_info=True)
            await update.message.reply_text(
                "Tuve un problema procesando tu mensaje. ¿Podemos intentarlo de nuevo?"
            )
        except Exception as e:
            logging.error(f"[chat_id={chat_id}] Error inesperado: {e}", exc_info=True)
            await update.message.reply_text(
                "Ocurrió un error inesperado. Intentá de nuevo en unos segundos."
            )

    def run(self):
        logging.info("Iniciando bot AFMA...")
        self.application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN no configurado.")
        sys.exit(1)
    if not GROQ_API_KEY:
        logging.error("GROQ_API_KEY no configurado.")
        sys.exit(1)

    try:
        bot = AgenteOrquestador(
            telegram_token=TELEGRAM_TOKEN,
            groq_api_key=GROQ_API_KEY,
        )
        bot.run()
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error al inicializar: {e}")
        sys.exit(1)