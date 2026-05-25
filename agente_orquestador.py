"""
Agente Orquestador - Bot de Telegram

Función:
    Interactuar con el usuario a través de Telegram, orquestar los análisis
    de los agentes especializados (técnico y fundamental) y consolidar una
    recomendación final de inversión usando un modelo de IA.

Skills:
    - Interfaz de usuario por chat (python-telegram-bot)
    - Extracción de entidades (tickers) de texto
    - Orquestación de múltiples agentes de IA
    - Generación de recomendaciones financieras consolidadas

Uso:
    python agente_orquestador.py
"""

import os
import sys
import logging
import json
import requests
import google.genai as genai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ──────────────────────────────────────────────────────────────────────────────
# PROMPT DEL ROUTER — Reglas de diseño:
#   • Solo 2 valores válidos para "accion": CHAT o EXEC. Sin excepciones.
#   • La regla "BUSCAR" / "RECOMENDAR" / cualquier otro verbo está explícitamente
#     prohibida — Llama 3.2 tiende a inventar acciones intermedias.
#   • Si el usuario NO tiene ticker exacto → accion=CHAT, el modelo escribe
#     él mismo las recomendaciones dentro del campo "mensaje".
#   • Se ejemplifican los dos únicos casos con JSON literal para que el modelo
#     no necesite "razonar" qué schema usar.
# ──────────────────────────────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = """\
Eres el enrutador de un asesor financiero de IOL (InvertirOnline) Argentina.
Tu único trabajo es extraer tres datos: CAPITAL, RIESGO y un TICKER EXACTO.
Luego devuelves UN SOLO objeto JSON. Nada más.

═══════════════════════════════════════════
 DEFINICIÓN ESTRICTA DE TICKER
═══════════════════════════════════════════
Un ticker es un símbolo de mercado concreto: SPY, AAPL, KO, GGAL, YPFD, AL30, GD30.
Las siguientes palabras NO son tickers; son clases de activos:
  CEDEAR, Bono, Caución, FCI, Acción, ETF, Obligación Negociable.
Si el usuario menciona una de esas palabras sin dar un símbolo exacto,
NO tienes ticker y DEBES usar accion=CHAT.

═══════════════════════════════════════════
 LAS ÚNICAS DOS ACCIONES PERMITIDAS
═══════════════════════════════════════════
ACCION 1 — "CHAT"  (usar cuando falta el ticker exacto)
  Redacta tú mismo, con lenguaje natural y persuasivo, un mensaje donde:
  - Recomiendes 3 tickers reales de IOL adecuados al capital y riesgo del usuario.
  - Expliques brevemente por qué cada uno es una buena opción.
  - Preguntes cuál quiere analizar.
  Formato JSON obligatorio:
  {"accion": "CHAT", "mensaje": "<tu mensaje aquí>"}

ACCION 2 — "EXEC"  (usar solo cuando tienes los 3 datos confirmados)
  Formato JSON obligatorio:
  {"accion": "EXEC", "ticker": "<TICKER>", "capital": "<capital>", "riesgo": "<riesgo>"}

PROHIBIDO: inventar cualquier otra acción como BUSCAR, RECOMENDAR, ANALIZAR, etc.
PROHIBIDO: devolver texto fuera del objeto JSON.
PROHIBIDO: agregar markdown, comentarios o explicaciones fuera del JSON.

═══════════════════════════════════════════
 EJEMPLOS
═══════════════════════════════════════════
Historial: "Usuario: quiero invertir 100000 pesos en CEDEARs con riesgo medio"
Respuesta correcta:
{"accion": "CHAT", "mensaje": "¡Excelente elección! Con $100.000 y riesgo medio, te recomiendo explorar estos CEDEARs: \\n• SPY (ETF del S&P 500): exposición diversificada al mercado americano, ideal para riesgo moderado.\\n• KO (Coca-Cola): empresa defensiva con dividendos estables, baja volatilidad.\\n• AAPL (Apple): tecnología consolidada con crecimiento constante.\\n¿Cuál de estos querés que analice en detalle?"}

Historial: "Usuario: quiero invertir 100000 pesos en CEDEARs con riesgo medio" / "Asesor: [recomendó SPY, KO, AAPL]" / "Usuario: analizá SPY"
Respuesta correcta:
{"accion": "EXEC", "ticker": "SPY", "capital": "100000", "riesgo": "medio"}
"""


class AgenteOrquestador:
    """
    Gestiona el bot de Telegram, recibe solicitudes de los usuarios,
    y orquesta las llamadas a los agentes de análisis para dar una respuesta final.
    """

    def __init__(self, telegram_token: str, gemini_api_key: str):
        if not telegram_token:
            raise ValueError("El token de Telegram no fue proporcionado.")
        if not gemini_api_key:
            raise ValueError("La API Key de Gemini no fue proporcionada.")

        self.application = ApplicationBuilder().token(telegram_token).build()

        try:
            self.genai_client = genai.Client(api_key=gemini_api_key)
            self.genai_model = "gemini-flash-latest"
        except Exception as e:
            raise RuntimeError(f"Error al configurar el cliente de Gemini: {e}") from e

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _llamar_ollama(self, historial: list[str]) -> dict:
        """
        Llama al modelo local Ollama con el historial de conversación.
        Retorna el dict JSON parseado o lanza una excepción.
        """
        full_prompt = ROUTER_SYSTEM_PROMPT + "\n\n--- Historial de conversación ---\n" + "\n".join(historial)

        payload = {
            "model": "llama3.2",
            "prompt": full_prompt,
            "stream": False,
            "format": "json"
        }

        resp = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=60)
        resp.raise_for_status()

        raw = resp.json()["response"].strip()
        logging.info(f"RAW Ollama JSON: {raw}")

        datos = json.loads(raw)  # Puede lanzar JSONDecodeError

        accion = datos.get("accion", "").upper()
        if accion not in ("CHAT", "EXEC"):
            # El modelo devolvió una acción inventada → la convertimos a CHAT
            # recuperando el campo "mensaje" si existe, o generando uno genérico.
            logging.warning(
                f"Acción desconocida '{accion}' recibida del router. "
                f"Degradando a CHAT. Payload completo: {datos}"
            )
            mensaje_fallback = datos.get("mensaje") or (
                "Entiendo que te interesan esos activos. ¿Podrías decirme el ticker exacto "
                "que querés analizar? Por ejemplo: SPY, AAPL, AL30 o GGAL."
            )
            return {"accion": "CHAT", "mensaje": mensaje_fallback}

        # Normalizar la acción a mayúsculas para el caller
        datos["accion"] = accion
        return datos

    # ──────────────────────────────────────────────────────────────────────────
    # Handlers de Telegram
    # ──────────────────────────────────────────────────────────────────────────

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        mensaje = (
            "¡Hola! Soy tu asesor financiero experto en IOL (InvertirOnline).\n\n"
            "Puedo ayudarte a analizar activos. Contame:\n"
            "  • ¿Cuánto capital querés invertir?\n"
            "  • ¿Cuál es tu tolerancia al riesgo? (bajo / medio / alto)\n"
            "  • ¿Qué activo o clase de activo te interesa?\n\n"
            "¿Buscás ideas? Podríamos explorar:\n"
            "✅ *CEDEARs*: Para invertir en empresas de USA en pesos.\n"
            "✅ *Bonos*: Opciones de renta fija en dólares o pesos.\n"
            "✅ *Acciones locales*: GGAL, YPFD, y más."
        )
        await update.message.reply_text(mensaje, parse_mode="Markdown")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        texto_usuario = update.message.text
        historial: list[str] = context.user_data.get("historial", [])
        historial.append(f"Usuario: {texto_usuario}")

        try:
            datos_ia = self._llamar_ollama(historial)
            accion = datos_ia["accion"]  # Siempre "CHAT" o "EXEC" tras _llamar_ollama

            # ── CASO 1: El router necesita más información ──────────────────
            if accion == "CHAT":
                mensaje = datos_ia.get("mensaje", "¿Podés contarme más sobre qué activo te interesa?")
                historial.append(f"Asesor: {mensaje}")
                context.user_data["historial"] = historial
                await update.message.reply_text(mensaje)

            # ── CASO 2: Tenemos todo — ejecutar el análisis completo ────────
            elif accion == "EXEC":
                ticker = datos_ia.get("ticker", "").upper()
                capital = datos_ia.get("capital", "")
                riesgo = datos_ia.get("riesgo", "")

                if not all([ticker, capital, riesgo]):
                    logging.error(f"EXEC recibido con campos vacíos: {datos_ia}")
                    await update.message.reply_text(
                        "Algo falló al procesar tus datos. ¿Podés repetir el ticker, "
                        "capital y nivel de riesgo?"
                    )
                    return

                # Limpiar el historial para la próxima consulta
                context.user_data.pop("historial", None)

                await update.message.reply_text(
                    f"¡Perfecto! Tengo todo lo que necesito. Analizando *{ticker}*... un momento 🔍",
                    parse_mode="Markdown"
                )

                # ── MOCK de los agentes especializados ─────────────────────
                # TODO: reemplazar por llamadas reales a AgenteTecnico y AgenteFundamental
                payload_str = f"TICKER:{ticker}|CAPITAL:{capital}|RIESGO:{riesgo}"
                logging.info(f"[MOCK] AgenteTecnico → {payload_str}")
                reporte_tecnico = (
                    f"Análisis Técnico (mock): Señal de COMPRA para {ticker} "
                    "basada en RSI bajo (38) y cruce alcista de medias móviles SMA10/SMA20."
                )

                logging.info(f"[MOCK] AgenteFundamental → {payload_str}")
                reporte_fundamental = (
                    f"Análisis Fundamental (mock): Sentimiento POSITIVO para {ticker} "
                    "por noticias de expansión de mercado y resultados trimestrales sólidos."
                )
                # ── FIN MOCK ────────────────────────────────────────────────

                prompt_final = (
                    f"Eres el Asesor Financiero final de IOL. Tu cliente es amateur. "
                    f"Activo analizado: {ticker} | Capital: {capital} | Riesgo: {riesgo}\n\n"
                    f"Análisis Técnico: {reporte_tecnico}\n"
                    f"Análisis Fundamental: {reporte_fundamental}\n\n"
                    "Sintetiza la decisión SIN usar jerga compleja. "
                    "Devuelve EXACTAMENTE este formato y nada más:\n\n"
                    "Veredicto: [COMPRAR / VENDER / RETENER] — [cantidad aproximada de nominales "
                    "a operar basándote en el capital].\n"
                    "Motivo: [máximo 2 líneas muy simples y directas]\n\n"
                    "¿Querés que profundice en algún detalle o analizamos otra opción?"
                )

                response = self.genai_client.models.generate_content(
                    model=self.genai_model,
                    contents=[prompt_final]
                )
                await update.message.reply_text(response.text.strip())

        except requests.exceptions.ConnectionError:
            logging.error("No se pudo conectar con Ollama en http://127.0.0.1:11434")
            context.user_data.pop("historial", None)
            await update.message.reply_text(
                "⚠️ El motor de IA local no está disponible en este momento. "
                "Por favor, contactá al administrador."
            )

        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError en respuesta de Ollama: {e}")
            context.user_data.pop("historial", None)
            await update.message.reply_text(
                "Tuve un problema interno al procesar tu mensaje. "
                "¿Podemos empezar de nuevo? Contame qué activo te interesa."
            )

        except Exception as e:
            logging.error(f"Error inesperado en handle_message: {e}", exc_info=True)
            context.user_data.pop("historial", None)
            await update.message.reply_text(
                "Ocurrió un error inesperado. Por favor, intentá de nuevo en unos segundos."
            )

    def run(self):
        """Inicia el bot en modo polling."""
        logging.info("Iniciando bot AFMA...")
        self.application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN no configurado. Revisá tu .env")
        sys.exit(1)
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY no configurado. Revisá tu .env")
        sys.exit(1)

    try:
        bot = AgenteOrquestador(telegram_token=TELEGRAM_TOKEN, gemini_api_key=GEMINI_API_KEY)
        bot.run()
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error al inicializar el bot: {e}")
        sys.exit(1)