"""
Agente Orquestador - Bot de Telegram

Función:
    Interactuar con el usuario a través de Telegram, orquestar los análisis
    de los agentes especializados (técnico y fundamental) y consolidar una
    recomendación final de inversión usando un modelo de IA.

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
# CATÁLOGO DE ACTIVOS IOL
# Principio: mínimo suficiente. No es una base de datos, es el vocabulario
# del dominio. Le da al router la capacidad de mapear nombres naturales
# ("YPF", "el petróleo argentino", "Coca Cola") a tickers exactos.
# ──────────────────────────────────────────────────────────────────────────────
IOL_CATALOG = """
ACCIONES LOCALES (Bolsa Argentina):
  GGAL  = Grupo Financiero Galicia (banco)
  YPFD  = YPF S.A. (petróleo y gas, empresa nacional)
  BMA   = Banco Macro
  PAMP  = Pampa Energía (energía eléctrica)
  TXAR  = Ternium Argentina (acero)
  LOMA  = Loma Negra (construcción/cemento)
  SUPV  = Supervielle (banco)

CEDEARs (acciones extranjeras que cotizan en pesos en Argentina):
  SPY   = ETF S&P 500 (diversificación total del mercado USA)
  QQQ   = ETF Nasdaq 100 (tecnología USA)
  AAPL  = Apple Inc.
  MSFT  = Microsoft Corp.
  GOOGL = Alphabet (Google)
  AMZN  = Amazon
  TSLA  = Tesla
  NVDA  = NVIDIA (semiconductores/IA)
  KO    = Coca-Cola (consumo defensivo, dividendos)
  JPM   = JPMorgan Chase (banco USA)
  MELI  = MercadoLibre (tecnología latinoamericana)
  GLOB  = Globant (tecnología argentina cotizando en NYSE)

BONOS SOBERANOS ARGENTINOS (renta fija en USD):
  AL30  = Bono vence 2030, legislación local
  AL35  = Bono vence 2035, legislación local
  GD30  = Bono vence 2030, legislación Nueva York (más seguro para inversores)
  GD35  = Bono vence 2035, legislación Nueva York
  ADVERTENCIA: AL30 y GD30 son BONOS SOBERANOS, NO acciones ni CEDEARs.

PERFIL DE RIESGO POR CLASE DE ACTIVO:
  Muy bajo  → Cauciones / FCI money market
  Bajo      → Bonos (AL30, GD30)
  Medio     → CEDEARs defensivos (KO, SPY)
  Medio-Alto→ Acciones locales (GGAL, YPFD)
  Alto      → CEDEARs growth (TSLA, NVDA)
"""

# ──────────────────────────────────────────────────────────────────────────────
# PROMPT DEL ROUTER
# ──────────────────────────────────────────────────────────────────────────────
ROUTER_SYSTEM_PROMPT = f"""
Eres el enrutador conversacional de AFMA, asesor financiero de IOL Argentina.
Tu trabajo: mantener una conversación natural para obtener CAPITAL, RIESGO y
un TICKER EXACTO, y devolver un JSON de acción.

{IOL_CATALOG}

CÓMO EXTRAER LOS DATOS:
- CAPITAL: cualquier monto mencionado por el usuario.
- RIESGO: bajo/medio/alto. Inferilo del contexto si no lo dice explícitamente.
  "quiero algo seguro" = bajo. "no me importa arriesgar" = alto. Duda = medio.
- TICKER: usá el catálogo para convertir nombres naturales a tickers exactos.
  "YPF" = YPFD | "Apple" = AAPL | "bonos" genérico = preguntá cuál.
  Si el usuario menciona varios activos, tomá el primero e informale que vas de a uno.

LAS ÚNICAS DOS ACCIONES VÁLIDAS:

ACCION "CHAT" — cuando falta algún dato O el usuario pide recomendaciones/distribución.
  - Respondé con lenguaje natural, cálido, directo. Nunca repetir recomendaciones ya dadas.
  - Si pide distribución de capital entre varios activos, proponé el reparto y preguntá por cuál empezar.
  Formato obligatorio: {{"accion": "CHAT", "mensaje": "<tu respuesta>"}}

ACCION "EXEC" — solo cuando tenés CAPITAL + RIESGO + TICKER exacto confirmados.
  Formato obligatorio: {{"accion": "EXEC", "ticker": "<TICKER>", "capital": "<capital>", "riesgo": "<riesgo>"}}

REGLAS CRÍTICAS:
1. Solo JSON puro. Sin texto fuera del objeto JSON.
2. Solo "CHAT" o "EXEC". NUNCA: BUSCAR, RECOMENDAR, ANALIZAR, DISTRIBUIR, etc.
3. Nunca inventar tickers fuera del catálogo.
4. Nunca confundir AL30/GD30 (bonos) con acciones o CEDEARs.
5. Leer el historial completo antes de responder.

EJEMPLOS:

Caso 1 — usuario da clase de activo sin ticker:
  Historial: "Usuario: quiero invertir 100000 en CEDEARs con riesgo medio"
  Respuesta: {{"accion": "CHAT", "mensaje": "Buena eleccion. Con $100.000 y riesgo medio te recomiendo:\\n- SPY: ETF del S&P500, maxima diversificacion en USA.\\n- KO (Coca-Cola): defensiva, dividendos estables, baja volatilidad.\\n- AAPL (Apple): tecnologia solida con crecimiento constante.\\n,Con cual empezamos?"}}

Caso 2 — usuario pide distribucion entre varios activos:
  Historial: "Usuario: quiero invertir en SPY, AAPL y YPF, como repartis 100000?"
  Respuesta: {{"accion": "CHAT", "mensaje": "Con $100.000 y riesgo medio una distribucion equilibrada seria:\\n- SPY: 50% ($50.000) ancla del portafolio.\\n- AAPL: 30% ($30.000) tecnologia consolidada.\\n- YPFD: 20% ($20.000) apuesta al sector energetico argentino.\\nLos analisis son de a uno. Arrancamos por SPY?"}}

Caso 3 — usuario confirma ticker:
  Historial: "[Asesor propuso SPY, KO, AAPL]\\nUsuario: analizá SPY"
  Respuesta: {{"accion": "EXEC", "ticker": "SPY", "capital": "100000", "riesgo": "medio"}}

Caso 4 — usuario da todo junto con nombre de empresa:
  Historial: "Usuario: quiero analizar YPF con 50000 pesos riesgo alto"
  Respuesta: {{"accion": "EXEC", "ticker": "YPFD", "capital": "50000", "riesgo": "alto"}}
"""


class AgenteOrquestador:
    """
    Gestiona el bot de Telegram y orquesta los agentes de análisis.
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
    # Helper: llamada a Ollama con degradación graceful
    # ──────────────────────────────────────────────────────────────────────────
    def _llamar_ollama(self, historial: list) -> dict:
        """
        Llama al modelo local Ollama con el historial completo.
        Garantiza que la respuesta sea siempre {"accion": "CHAT"|"EXEC", ...}
        aunque el modelo devuelva una acción inventada.
        """
        full_prompt = (
            ROUTER_SYSTEM_PROMPT
            + "\n\nHISTORIAL DE CONVERSACION ACTUAL:\n"
            + "\n".join(historial)
            + "\n\nDevuelve tu respuesta JSON ahora:"
        )

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

        datos = json.loads(raw)  # JSONDecodeError se propaga al caller

        accion = str(datos.get("accion", "")).upper()

        if accion not in ("CHAT", "EXEC"):
            logging.warning(
                f"Accion invalida '{accion}' del router. Degradando a CHAT. Payload: {datos}"
            )
            mensaje_rescatado = (
                datos.get("mensaje")
                or datos.get("message")
                or datos.get("respuesta")
                or (
                    "Entiendo lo que busca. Para analizar el activo, "
                    "confirme el ticker exacto que quiere analizar primero."
                )
            )
            return {"accion": "CHAT", "mensaje": mensaje_rescatado}

        datos["accion"] = accion
        return datos

    # ──────────────────────────────────────────────────────────────────────────
    # Handlers de Telegram
    # ──────────────────────────────────────────────────────────────────────────
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        mensaje = (
            "Hola! Soy AFMA, tu asesor financiero para IOL (InvertirOnline).\n\n"
            "Podés hablarme de forma natural. Por ejemplo:\n"
            "  \"Tengo $100.000 y quiero invertir en CEDEARs\"\n"
            "  \"Qué bonos en dólares recomendás para riesgo bajo?\"\n"
            "  \"Quiero analizar YPF con riesgo alto\"\n\n"
            "En qué te ayudo hoy?"
        )
        await update.message.reply_text(mensaje)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        texto_usuario = update.message.text
        historial = context.user_data.get("historial", [])
        historial.append(f"Usuario: {texto_usuario}")

        try:
            datos_ia = self._llamar_ollama(historial)
            accion = datos_ia["accion"]

            # ── CASO CHAT: continuar conversación ─────────────────────────
            if accion == "CHAT":
                mensaje = datos_ia.get("mensaje", "Podés contarme más? Qué activo te interesa?")
                historial.append(f"Asesor: {mensaje}")
                context.user_data["historial"] = historial
                await update.message.reply_text(mensaje)

            # ── CASO EXEC: lanzar análisis completo ───────────────────────
            elif accion == "EXEC":
                ticker  = datos_ia.get("ticker", "").upper()
                capital = datos_ia.get("capital", "")
                riesgo  = datos_ia.get("riesgo", "")

                if not all([ticker, capital, riesgo]):
                    logging.error(f"EXEC con campos vacios: {datos_ia}")
                    historial.append("Asesor: [error interno — datos incompletos]")
                    context.user_data["historial"] = historial
                    await update.message.reply_text(
                        "Casi llegamos, pero me faltó un dato. "
                        "Confirmame el ticker, capital y nivel de riesgo."
                    )
                    return

                context.user_data.pop("historial", None)

                await update.message.reply_text(
                    f"Analizando {ticker} — capital ${capital}, riesgo {riesgo}... un momento"
                )

                # ── MOCK agentes especializados ────────────────────────────
                # TODO: reemplazar por instancias reales de AgenteTecnico y AgenteFundamental
                payload_str = f"TICKER:{ticker}|CAPITAL:{capital}|RIESGO:{riesgo}"
                logging.info(f"[MOCK] AgenteTecnico  -> {payload_str}")
                reporte_tecnico = (
                    f"Señal COMPRA para {ticker}: RSI 38 (sobreventa), "
                    "cruce alcista SMA10/SMA20, volumen sobre promedio 10d."
                )
                logging.info(f"[MOCK] AgenteFundamental -> {payload_str}")
                reporte_fundamental = (
                    f"Sentimiento POSITIVO para {ticker}: noticias de expansión "
                    "y resultados trimestrales sobre estimaciones."
                )
                # ── FIN MOCK ───────────────────────────────────────────────

                prompt_final = (
                    f"Eres el Asesor Financiero final de IOL. Cliente amateur. "
                    f"Activo: {ticker} | Capital: ${capital} | Riesgo: {riesgo}\n\n"
                    f"Análisis Técnico: {reporte_tecnico}\n"
                    f"Análisis Fundamental: {reporte_fundamental}\n\n"
                    "Sintetizá la decisión SIN jerga técnica. "
                    "Devolvé EXACTAMENTE este formato:\n\n"
                    "Veredicto: [COMPRAR / VENDER / RETENER] — [cantidad aprox. de nominales]\n"
                    "Motivo: [máximo 2 líneas simples]\n\n"
                    "¿Querés profundizar en algún detalle o analizamos otro activo?"
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
                "El motor de IA local no esta disponible ahora mismo. "
                "Por favor contacta al administrador."
            )

        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError del router Ollama: {e}", exc_info=True)
            context.user_data.pop("historial", None)
            await update.message.reply_text(
                "Tuve un problema procesando tu mensaje. "
                "Podemos empezar de nuevo? Contame que activo te interesa."
            )

        except Exception as e:
            logging.error(f"Error inesperado: {e}", exc_info=True)
            context.user_data.pop("historial", None)
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
        logging.error("TELEGRAM_TOKEN no configurado. Revisa tu .env")
        sys.exit(1)
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY no configurado. Revisa tu .env")
        sys.exit(1)

    try:
        bot = AgenteOrquestador(telegram_token=TELEGRAM_TOKEN, gemini_api_key=GEMINI_API_KEY)
        bot.run()
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error al inicializar el bot: {e}")
        sys.exit(1)