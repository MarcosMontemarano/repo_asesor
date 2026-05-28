"""
Agente Orquestador - Bot de Telegram

Arquitectura:
  - Llama (Ollama): extrae entidades del lenguaje natural (texto_activo,
    capital, riesgo, cantidad_activos, intencion). Sin reglas de negocio.
  - Python: toda la lógica de decisión (CHAT vs EXEC), resolución de
    tickers, persistencia de contexto entre análisis.
  - ticker_resolver: módulo independiente para resolución nombre→ticker.

Uso:
    python agente_orquestador.py
"""

import os
import re
import sys
import logging
import json
import requests
import google.genai as genai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

import ticker_resolver as tr

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ──────────────────────────────────────────────────────────────────────────────
# PROMPT DE OLLAMA — Solo extracción de entidades, sin reglas de negocio.
# El prompt es intencionalmente corto para que Llama no se desvíe.
# ──────────────────────────────────────────────────────────────────────────────
EXTRACTOR_PROMPT = """
Sos un extractor de entidades financieras. Tu único trabajo es leer el
mensaje del usuario y devolver un JSON con exactamente estos campos:

{
  "texto_activo": "nombre o código del activo mencionado (string, vacío si no hay ninguno)",
  "capital":      "solo los dígitos del monto mencionado (string, vacío si no hay monto)",
  "riesgo":       "bajo, medio o alto (inferilo del contexto, vacío si no hay señal)",
  "cantidad_activos": 1,
  "intencion":    "analizar, distribuir, recomendar, consulta, otro"
}

Reglas estrictas:
- "texto_activo": devolvé el nombre TAL COMO lo escribió el usuario. No lo traduzcas ni lo conviertas.
- "capital": SOLO dígitos. Nunca nombres de empresas. "100k" → "100000". "$50.000" → "50000". Vacío si no hay monto.
- "riesgo": inferilo del contexto. "quiero algo seguro" → "bajo". "no me importa arriesgar" → "alto". Sin señal → vacío.
- "cantidad_activos": cuántos activos DISTINTOS mencionó. "SPY y AAPL" → 2.
- "intencion":
    "analizar"    → quiere analizar un activo específico
    "distribuir"  → quiere repartir capital entre varios activos
    "recomendar"  → pide sugerencias sin tener un activo en mente
    "consulta"    → pregunta educativa, de timing, noticias, gráficos
    "otro"        → cualquier otra cosa

Solo JSON puro. Sin texto fuera del objeto JSON. Sin explicaciones.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Frases que indican que el usuario quiere iniciar un análisis nuevo
# (resetean capital y riesgo del contexto)
# ──────────────────────────────────────────────────────────────────────────────
FRASES_RESET = [
    "otro análisis", "nueva inversión", "nuevo análisis",
    "empecemos de nuevo", "empezar de nuevo", "quiero hacer otro",
    "análisis distinto", "inversión distinta", "cambiemos de tema",
    "olvidá lo anterior", "borrá todo", "resetear",
]


def _detecta_reset(texto: str) -> bool:
    texto_norm = texto.lower()
    return any(frase in texto_norm for frase in FRASES_RESET)


def _extraer_numero(valor) -> str:
    if valor is None:
        return ""
    return re.sub(r"[^\d]", "", str(valor))


# ──────────────────────────────────────────────────────────────────────────────
# Contexto persistente por usuario
# Almacena capital y riesgo entre análisis mientras no se resetee.
# ──────────────────────────────────────────────────────────────────────────────
class ContextoUsuario:
    """
    Mantiene el estado de una conversación:
      - historial: lista de turnos para que Llama tenga contexto
      - capital / riesgo: persisten entre análisis
      - ticker_pendiente: ticker del último análisis completado
    """

    def __init__(self):
        self.historial:        list[str] = []
        self.capital:          str = ""
        self.riesgo:           str = ""
        self.ticker_pendiente: str = ""

    def actualizar_desde_entidades(self, entidades: dict) -> None:
        """Actualiza capital y riesgo solo si el mensaje trae valores nuevos."""
        capital_nuevo = _extraer_numero(entidades.get("capital", ""))
        riesgo_nuevo  = str(entidades.get("riesgo", "")).strip().lower()

        if capital_nuevo:
            self.capital = capital_nuevo
        if riesgo_nuevo in ("bajo", "medio", "alto"):
            self.riesgo = riesgo_nuevo

    def resetear_contexto(self) -> None:
        """Resetea capital, riesgo e historial para un análisis nuevo."""
        self.capital          = ""
        self.riesgo           = ""
        self.ticker_pendiente = ""
        self.historial        = []

    def datos_completos(self, ticker: str) -> bool:
        return bool(ticker and self.capital and self.riesgo)

    def resumen(self) -> str:
        return f"capital={self.capital or '?'}, riesgo={self.riesgo or '?'}"


class AgenteOrquestador:
    """
    Gestiona el bot de Telegram y orquesta los agentes de análisis.
    """

    def __init__(self, telegram_token: str, gemini_api_key: str):
        if not telegram_token:
            raise ValueError("El token de Telegram no fue proporcionado.")
        if not gemini_api_key:
            raise ValueError("La API Key de Gemini no fue proporcionada.")

        # Inicializar índice de tickers (carga cache o regenera)
        logging.info("Inicializando índice de tickers...")
        tr.inicializar()
        logging.info("Índice de tickers listo.")

        self.application = ApplicationBuilder().token(telegram_token).build()
        self._contextos: dict[int, ContextoUsuario] = {}

        try:
            self.genai_client = genai.Client(api_key=gemini_api_key)
            self.genai_model  = "gemini-flash-latest"
        except Exception as e:
            raise RuntimeError(f"Error al configurar Gemini: {e}") from e

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    # ──────────────────────────────────────────────────────────────────────────
    # Extracción de entidades vía Ollama
    # ──────────────────────────────────────────────────────────────────────────
    def _extraer_entidades(self, chat_id: int, historial: list[str], texto_usuario: str) -> dict:
        """
        Llama a Ollama con el historial reciente + mensaje actual.
        Devuelve el dict de entidades normalizado.
        """
        historial_reciente = historial[-6:]  # últimos 6 turnos como contexto

        prompt_completo = (
            EXTRACTOR_PROMPT
            + "\n\n--- Historial reciente ---\n"
            + "\n".join(historial_reciente)
            + f"\n\nMensaje actual del usuario: {texto_usuario}"
            + "\n\nDevolvé el JSON ahora:"
        )

        payload = {
            "model": "llama3.2",
            "prompt": prompt_completo,
            "stream": False,
            "format": "json"
        }

        resp = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=60)
        resp.raise_for_status()

        raw = resp.json()["response"].strip()
        logging.info(f"[chat_id={chat_id}] Entidades extraídas: {raw}")

        if not raw or raw == "{}":
            return {}

        datos = json.loads(raw)
        # Normalizar capital a dígitos puros
        datos["capital"] = _extraer_numero(datos.get("capital", ""))
        return datos

    # ──────────────────────────────────────────────────────────────────────────
    # Lógica de decisión — 100% en Python
    # ──────────────────────────────────────────────────────────────────────────
    def _decidir(
        self,
        entidades: dict,
        contexto: ContextoUsuario,
        texto_usuario: str
    ) -> tuple[str, dict]:
        """
        Decide la acción a tomar basándose en las entidades extraídas y el
        contexto persistente del usuario.

        Devuelve (accion, datos) donde accion es "CHAT" o "EXEC" y datos
        contiene los parámetros relevantes para cada caso.
        """
        texto_activo    = str(entidades.get("texto_activo", "")).strip()
        intencion       = str(entidades.get("intencion", "otro")).lower()
        cantidad        = int(entidades.get("cantidad_activos", 1) or 1)

        # Actualizar contexto con los datos nuevos del mensaje
        contexto.actualizar_desde_entidades(entidades)

        # 1. Intenciones que nunca son EXEC
        if intencion in ("distribuir", "recomendar", "consulta"):
            return "CHAT", {"motivo": intencion}

        # 2. Múltiples activos → siempre CHAT
        if cantidad > 1:
            return "CHAT", {"motivo": "multiples_activos"}

        # 3. Resolver ticker
        ticker = tr.resolver_ticker(texto_activo) if texto_activo else None

        # 4. Ticker fuera del catálogo
        if texto_activo and not ticker:
            return "CHAT", {"motivo": "ticker_no_encontrado", "texto": texto_activo}

        # 5. Si hay ticker nuevo y el contexto ya tenía datos, pedir confirmación
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

        # 6. Sin ticker en el mensaje
        if not ticker:
            return "CHAT", {"motivo": "falta_ticker"}

        # 7. Faltan datos — preguntar específicamente qué falta
        if not contexto.capital:
            return "CHAT", {"motivo": "falta_capital", "ticker": ticker}
        if not contexto.riesgo:
            return "CHAT", {"motivo": "falta_riesgo", "ticker": ticker}

        # 8. Todo completo → EXEC
        contexto.ticker_pendiente = ticker
        return "EXEC", {
            "ticker":  ticker,
            "capital": contexto.capital,
            "riesgo":  contexto.riesgo,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Generación de mensajes CHAT con Gemini
    # ──────────────────────────────────────────────────────────────────────────
    def _generar_mensaje_chat(self, motivo: str, datos: dict, contexto: ContextoUsuario) -> str:
        """
        Genera el mensaje de respuesta para cada caso de CHAT.
        Casos simples usan texto fijo. Casos que requieren creatividad usan Gemini.
        """
        if motivo == "falta_capital":
            ticker = datos.get("ticker", "")
            meta   = tr.obtener_metadata_bono(ticker)
            nombre = meta["nombre"] if meta else ticker
            return (
                f"Perfecto, vamos a analizar {nombre}. "
                f"¿Con cuánto capital contás para esta inversión?"
            )

        if motivo == "falta_riesgo":
            ticker = datos.get("ticker", "")
            return (
                f"Ya tengo el activo y el capital. "
                f"¿Cuál es tu tolerancia al riesgo para {ticker}? "
                f"Podés elegir: bajo, medio o alto."
            )

        if motivo == "falta_ticker":
            return (
                "Entiendo lo que buscás. "
                "¿Qué activo específico querés que analice primero?"
            )

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

        # Para distribuir, recomendar, consulta y otros → Gemini genera la respuesta
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
            resp = self.genai_client.models.generate_content(
                model=self.genai_model,
                contents=[prompt]
            )
            return resp.text.strip()
        except Exception as e:
            logging.error(f"Error al generar mensaje CHAT con Gemini: {e}")
            return "¿Podés contarme más sobre lo que buscás?"

    # ──────────────────────────────────────────────────────────────────────────
    # Handlers de Telegram
    # ──────────────────────────────────────────────────────────────────────────
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self._contextos[chat_id] = ContextoUsuario()
        logging.info(f"[chat_id={chat_id}] /start. Contexto reseteado.")

        await update.message.reply_text(
            "Hola! Soy AFMA, tu asesor financiero para IOL (InvertirOnline).\n\n"
            "Podés hablarme de forma natural. Por ejemplo:\n"
            "  \"Tengo $100.000 y quiero invertir en CEDEARs\"\n"
            "  \"Qué bonos en dólares recomendás para riesgo bajo?\"\n"
            "  \"Quiero analizar YPF con riesgo alto\"\n\n"
            "En qué te ayudo hoy?"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id      = update.effective_chat.id
        texto_usuario = update.message.text

        # Obtener o crear contexto del usuario
        if chat_id not in self._contextos:
            self._contextos[chat_id] = ContextoUsuario()
        contexto = self._contextos[chat_id]

        # Detectar si el usuario quiere empezar un análisis nuevo
        if _detecta_reset(texto_usuario):
            contexto.resetear_contexto()
            await update.message.reply_text(
                "Perfecto, empezamos de nuevo. "
                "Contame qué activo te interesa y con cuánto capital."
            )
            return

        contexto.historial.append(f"Usuario: {texto_usuario}")

        try:
            # ── PASO 1: Llama extrae entidades ────────────────────────────
            entidades = self._extraer_entidades(chat_id, contexto.historial, texto_usuario)

            if not entidades:
                await update.message.reply_text(
                    "No pude procesar tu mensaje. "
                    "Podés repetirlo de otra forma?"
                )
                return

            # ── PASO 2: Python decide la acción ───────────────────────────
            accion, datos = self._decidir(entidades, contexto, texto_usuario)

            logging.info(
                f"[chat_id={chat_id}] accion={accion} datos={datos} "
                f"contexto={contexto.resumen()}"
            )

            # ── CASO CHAT ─────────────────────────────────────────────────
            if accion == "CHAT":
                motivo  = datos.get("motivo", "otro")
                mensaje = self._generar_mensaje_chat(motivo, datos, contexto)
                contexto.historial.append(f"Asesor: {mensaje}")
                await update.message.reply_text(mensaje)

            # ── CASO EXEC ─────────────────────────────────────────────────
            elif accion == "EXEC":
                ticker  = datos["ticker"]
                capital = datos["capital"]
                riesgo  = datos["riesgo"]

                await update.message.reply_text(
                    f"Analizando {ticker} — capital ${capital}, riesgo {riesgo}... un momento"
                )

                # Metadata de bono si aplica
                meta_bono = tr.obtener_metadata_bono(ticker)

                # ── Agentes especializados ─────────────────────────────────
                # TODO: reemplazar por instancias reales de AgenteTecnico y AgenteFundamental
                ticker_yf   = tr.obtener_ticker_yfinance(ticker)
                payload_str = f"TICKER:{ticker}|YF:{ticker_yf}|CAPITAL:{capital}|RIESGO:{riesgo}"
                logging.info(f"[MOCK] AgenteTecnico  → {payload_str}")
                logging.info(f"[MOCK] AgenteFundamental → {payload_str}")

                if meta_bono:
                    reporte_tecnico = (
                        f"Análisis técnico no disponible para bonos soberanos. "
                        f"Contexto: {meta_bono['descripcion']}"
                    )
                    reporte_fundamental = (
                        f"Sentimiento NEUTRAL para {ticker}: "
                        f"bono {meta_bono['legislacion']}, "
                        f"vencimiento {meta_bono['vencimiento']}, "
                        f"cupón {meta_bono['cupon_anual']}."
                    )
                else:
                    reporte_tecnico = (
                        f"Señal COMPRA para {ticker}: RSI 38 (sobreventa), "
                        "cruce alcista SMA10/SMA20, volumen sobre promedio 10d."
                    )
                    reporte_fundamental = (
                        f"Sentimiento POSITIVO para {ticker}: noticias de expansión "
                        "y resultados trimestrales sobre estimaciones."
                    )
                # ── FIN MOCK ───────────────────────────────────────────────

                prompt_final = (
                    f"Sos el Asesor Financiero final de IOL. Cliente amateur.\n"
                    f"Activo: {ticker} | Capital: ${capital} | Riesgo: {riesgo}\n\n"
                    f"Análisis Técnico: {reporte_tecnico}\n"
                    f"Análisis Fundamental: {reporte_fundamental}\n\n"
                    "Sintetizá la decisión SIN jerga técnica. "
                    "Devolvé EXACTAMENTE este formato:\n\n"
                    "Veredicto: [COMPRAR / VENDER / RETENER] — [cantidad aprox. de nominales]\n"
                    "Motivo: [máximo 2 líneas simples]\n\n"
                    "Al final preguntá si quiere profundizar o analizar otro activo."
                )

                response = self.genai_client.models.generate_content(
                    model=self.genai_model,
                    contents=[prompt_final]
                )
                mensaje_final = response.text.strip()
                contexto.historial.append(f"Asesor: {mensaje_final}")
                await update.message.reply_text(mensaje_final)

        except requests.exceptions.ConnectionError:
            logging.error(f"[chat_id={chat_id}] Ollama no disponible.")
            await update.message.reply_text(
                "El motor de IA local no esta disponible. "
                "Por favor contacta al administrador."
            )

        except json.JSONDecodeError as e:
            logging.error(f"[chat_id={chat_id}] JSONDecodeError: {e}", exc_info=True)
            await update.message.reply_text(
                "Tuve un problema procesando tu mensaje. "
                "Podemos intentarlo de nuevo?"
            )

        except Exception as e:
            logging.error(f"[chat_id={chat_id}] Error inesperado: {e}", exc_info=True)
            await update.message.reply_text(
                "Ocurrio un error inesperado. Intentá de nuevo en unos segundos."
            )

    def run(self):
        logging.info("Iniciando bot AFMA...")
        self.application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN no configurado.")
        sys.exit(1)
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY no configurado.")
        sys.exit(1)

    try:
        bot = AgenteOrquestador(telegram_token=TELEGRAM_TOKEN, gemini_api_key=GEMINI_API_KEY)
        bot.run()
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error al inicializar: {e}")
        sys.exit(1)