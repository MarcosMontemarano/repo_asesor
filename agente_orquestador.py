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
  GD30  = Bono vence 2030, legislación Nueva York (más seguro)
  GD35  = Bono vence 2035, legislación Nueva York
  ADVERTENCIA: AL30 y GD30 son BONOS SOBERANOS, NO acciones ni CEDEARs.

PERFIL DE RIESGO POR CLASE DE ACTIVO:
  Muy bajo   → Cauciones / FCI money market
  Bajo       → Bonos (AL30, GD30)
  Medio      → CEDEARs defensivos (KO, SPY)
  Medio-Alto → Acciones locales (GGAL, YPFD, PAMP)
  Alto       → CEDEARs growth (TSLA, NVDA)
"""

# ──────────────────────────────────────────────────────────────────────────────
# PROMPT DEL ROUTER
#
# Decisiones de diseño:
#   1. El HISTORIAL va PRIMERO. Llama 3.2 le da más peso al texto inicial.
#   2. Ejemplos con PLACEHOLDERS, no texto literal copiable.
#   3. Regla explícita: el ticker a ejecutar es SIEMPRE el último mencionado
#      por el usuario, no el más frecuente en el historial.
#   4. Solo "accion" sin tilde como clave JSON. Se refuerza con mayúsculas
#      y ejemplo literal de la clave exacta.
# ──────────────────────────────────────────────────────────────────────────────
ROUTER_INSTRUCTIONS = f"""
{IOL_CATALOG}

═══════════════════════════════════════════
 TU ROL
═══════════════════════════════════════════
Eres el enrutador de AFMA, asesor financiero de IOL Argentina.
Leé el historial de arriba y extraé: CAPITAL, RIESGO y TICKER.
Devolvé UN SOLO objeto JSON. Sin texto fuera del JSON.
La clave de acción se escribe EXACTAMENTE así: "accion" (sin tilde, sin acento).

═══════════════════════════════════════════
 QUÉ ES UN TICKER
═══════════════════════════════════════════
Un ticker es el código corto con el que se identifica una empresa o activo
en la bolsa. Por ejemplo: PAMP es el ticker de Pampa Energía, AAPL es el
ticker de Apple, AL30 es el ticker de un bono soberano argentino.
Cuando el usuario menciona una empresa por nombre, usá el catálogo para
encontrar su ticker exacto. Cuando uses un ticker en tu mensaje, siempre
aclará el nombre de la empresa entre paréntesis.
Ejemplo correcto: "PAMP (Pampa Energía)" o "AAPL (Apple)".

═══════════════════════════════════════════
 EXTRACCIÓN DE DATOS
═══════════════════════════════════════════
CAPITAL : cualquier monto que mencione el usuario.
RIESGO  : bajo / medio / alto.
          Si no lo dice, inferilo del contexto.
          "quiero algo seguro" = bajo.
          "no me importa arriesgar" = alto.
          Sin señal clara = medio.
TICKER  : usá el catálogo para mapear nombres a tickers exactos.
          "YPF" → YPFD | "Pampa" → PAMP | "Apple" → AAPL
          "bonos" sin especificar → preguntá cuál.

REGLA CRÍTICA DE TICKER:
  El ticker a usar en EXEC es SIEMPRE el último activo mencionado
  explícitamente por el usuario en el historial, no el más frecuente
  ni el primero que aparece. Si el usuario dice "analizá PAMP" después
  de haber hablado de YPF, el ticker es PAMP.

  Si el usuario menciona varios activos a la vez, tomá el primero de esa
  lista e informale que los analizás de a uno.

═══════════════════════════════════════════
 ACCIONES VÁLIDAS — SOLO ESTAS DOS
═══════════════════════════════════════════
"CHAT" : cuando falta algún dato, el usuario pide recomendaciones,
         o pide distribución de capital entre varios activos.
         → Respondé basándote en el historial real, no en los ejemplos.
         → Si pide distribución, proponé el reparto y preguntá por cuál empezar.
         → NUNCA repitas información que ya está en el historial.
         → NUNCA preguntes datos que el usuario ya dio en el historial.
         → Al mencionar un ticker, siempre aclarás el nombre entre paréntesis.
         Formato: {{"accion": "CHAT", "mensaje": "[respuesta personalizada]"}}

"EXEC" : solo cuando tenés CAPITAL + RIESGO + TICKER confirmados.
         Formato: {{"accion": "EXEC", "ticker": "[TICKER]", "capital": "[monto]", "riesgo": "[nivel]"}}

PROHIBIDO inventar acciones como BUSCAR, RECOMENDAR, ANALIZAR, etc.
PROHIBIDO usar "acción" con tilde. La clave es siempre "accion".
PROHIBIDO inventar tickers que no estén en el catálogo.
PROHIBIDO preguntar por datos que el usuario ya proporcionó.

═══════════════════════════════════════════
 ESTRUCTURA DE EJEMPLOS
═══════════════════════════════════════════
Caso A — falta el ticker, el usuario pide ideas:
  → {{"accion": "CHAT", "mensaje": "[Recomendás 3 tickers del catálogo con nombre entre paréntesis, adecuados al capital y riesgo mencionados en el historial. Preguntás cuál analizar primero.]"}}

Caso B — el usuario pide distribución entre varios activos:
  → {{"accion": "CHAT", "mensaje": "[Proponés un reparto porcentual concreto para los activos que mencionó, con nombre entre paréntesis. Informás que los análisis son de a uno. Preguntás por cuál empezar.]"}}

Caso C — el usuario confirma un ticker o nombra una empresa directamente:
  → {{"accion": "EXEC", "ticker": "[ticker exacto del catálogo, el último mencionado por el usuario]", "capital": "[capital del historial]", "riesgo": "[riesgo del historial o inferido]"}}

Caso D — el usuario da todo junto en un mensaje:
  → {{"accion": "EXEC", "ticker": "[ticker mapeado desde el catálogo]", "capital": "[monto mencionado]", "riesgo": "[nivel mencionado o inferido]"}}
"""


def _normalizar_clave_accion(datos: dict) -> dict:
    """
    Llama 3.2 a veces devuelve 'acción' con tilde en lugar de 'accion'.
    Esta función busca la clave correcta con o sin tilde y la normaliza,
    garantizando que el resto del código siempre lea 'accion'.
    """
    if "accion" not in datos:
        for variante in ("acción", "Accion", "Acción", "ACTION", "action"):
            if variante in datos:
                datos["accion"] = datos.pop(variante)
                logging.warning(f"Clave '{variante}' normalizada a 'accion'.")
                break
    return datos


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

        # Historial respaldado en la instancia, keyed por chat_id.
        # Fuente de verdad independiente de context.user_data.
        self._historiales: dict[int, list[str]] = {}

        try:
            self.genai_client = genai.Client(api_key=gemini_api_key)
            self.genai_model = "gemini-flash-latest"
        except Exception as e:
            raise RuntimeError(f"Error al configurar el cliente de Gemini: {e}") from e

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: llamada a Ollama
    # ──────────────────────────────────────────────────────────────────────────
    def _llamar_ollama(self, chat_id: int, historial: list[str]) -> dict:
        """
        Construye el prompt con el historial PRIMERO y las instrucciones DESPUÉS.
        Trunca a los últimos 10 turnos para no exceder el contexto de Llama.
        """
        historial_reciente = historial[-10:]

        full_prompt = (
            "═══════════════════════════════════════════\n"
            " HISTORIAL DE CONVERSACIÓN ACTUAL\n"
            "═══════════════════════════════════════════\n"
            + "\n".join(historial_reciente)
            + "\n\n"
            + "═══════════════════════════════════════════\n"
            " INSTRUCCIONES DEL SISTEMA\n"
            "═══════════════════════════════════════════\n"
            + ROUTER_INSTRUCTIONS
            + "\n\nAnalizá el historial completo y devolvé tu respuesta JSON ahora:"
        )

        logging.info(
            f"[chat_id={chat_id}] Llamando Ollama. "
            f"Historial ({len(historial_reciente)} turnos): {historial_reciente}"
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
        logging.info(f"[chat_id={chat_id}] RAW Ollama JSON: {raw}")

        datos = json.loads(raw)

        # Fix: normalizar clave "acción" con tilde → "accion"
        datos = _normalizar_clave_accion(datos)

        accion = str(datos.get("accion", "")).upper()

        if accion not in ("CHAT", "EXEC"):
            logging.warning(
                f"[chat_id={chat_id}] Acción inválida '{accion}'. "
                f"Degradando a CHAT. Payload completo: {datos}"
            )
            mensaje_rescatado = (
                datos.get("mensaje")
                or datos.get("message")
                or datos.get("respuesta")
                or "Entiendo lo que buscás. ¿Me confirmás qué activo querés analizar primero?"
            )
            return {"accion": "CHAT", "mensaje": mensaje_rescatado}

        datos["accion"] = accion
        return datos

    # ──────────────────────────────────────────────────────────────────────────
    # Handlers de Telegram
    # ──────────────────────────────────────────────────────────────────────────
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        self._historiales[chat_id] = []
        logging.info(f"[chat_id={chat_id}] Sesión iniciada con /start. Historial reseteado.")

        mensaje = (
            "¡Hola! Soy AFMA, tu asesor financiero para IOL (InvertirOnline).\n\n"
            "Podés hablarme de forma natural. Por ejemplo:\n"
            "  \"Tengo $100.000 y quiero invertir en CEDEARs\"\n"
            "  \"¿Qué bonos en dólares recomendás para riesgo bajo?\"\n"
            "  \"Quiero analizar YPF con riesgo alto\"\n\n"
            "¿En qué te ayudo hoy?"
        )
        await update.message.reply_text(mensaje)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        texto_usuario = update.message.text

        historial = self._historiales.get(chat_id, [])
        historial.append(f"Usuario: {texto_usuario}")

        try:
            datos_ia = self._llamar_ollama(chat_id, historial)
            accion = datos_ia["accion"]

            # ── CASO CHAT ─────────────────────────────────────────────────
            if accion == "CHAT":
                mensaje = datos_ia.get("mensaje", "¿Podés contarme más? ¿Qué activo te interesa?")
                historial.append(f"Asesor: {mensaje}")
                self._historiales[chat_id] = historial
                logging.info(
                    f"[chat_id={chat_id}] CHAT respondido. "
                    f"Historial ahora tiene {len(historial)} entradas."
                )
                await update.message.reply_text(mensaje)

            # ── CASO EXEC ─────────────────────────────────────────────────
            elif accion == "EXEC":
                ticker  = datos_ia.get("ticker", "").upper()
                capital = str(datos_ia.get("capital", ""))
                riesgo  = datos_ia.get("riesgo", "")

                if not all([ticker, capital, riesgo]):
                    logging.error(f"[chat_id={chat_id}] EXEC con campos vacíos: {datos_ia}")
                    historial.append("Asesor: [error interno — datos incompletos]")
                    self._historiales[chat_id] = historial
                    await update.message.reply_text(
                        "Casi llegamos, pero me faltó algún dato. "
                        "¿Me confirmás el ticker, capital y nivel de riesgo?"
                    )
                    return

                self._historiales[chat_id] = []
                logging.info(
                    f"[chat_id={chat_id}] EXEC disparado: {ticker} | {capital} | {riesgo}. "
                    "Historial reseteado."
                )

                await update.message.reply_text(
                    f"Analizando {ticker} — capital ${capital}, riesgo {riesgo}... un momento 🔍"
                )

                # ── MOCK agentes especializados ────────────────────────────
                # TODO: reemplazar por instancias reales de AgenteTecnico y AgenteFundamental
                payload_str = f"TICKER:{ticker}|CAPITAL:{capital}|RIESGO:{riesgo}"
                logging.info(f"[MOCK] AgenteTecnico  → {payload_str}")
                reporte_tecnico = (
                    f"Señal COMPRA para {ticker}: RSI 38 (sobreventa), "
                    "cruce alcista SMA10/SMA20, volumen sobre promedio 10d."
                )
                logging.info(f"[MOCK] AgenteFundamental → {payload_str}")
                reporte_fundamental = (
                    f"Sentimiento POSITIVO para {ticker}: noticias de expansión "
                    "y resultados trimestrales sobre estimaciones."
                )
                # ── FIN MOCK ───────────────────────────────────────────────

                prompt_final = (
                    f"Eres el Asesor Financiero final de IOL. Cliente amateur.\n"
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
            logging.error(f"[chat_id={chat_id}] No se pudo conectar con Ollama.")
            self._historiales.pop(chat_id, None)
            await update.message.reply_text(
                "⚠️ El motor de IA local no está disponible ahora mismo. "
                "Por favor, contactá al administrador."
            )

        except json.JSONDecodeError as e:
            logging.error(f"[chat_id={chat_id}] JSONDecodeError de Ollama: {e}", exc_info=True)
            self._historiales.pop(chat_id, None)
            await update.message.reply_text(
                "Tuve un problema procesando tu mensaje. "
                "¿Podemos empezar de nuevo? Contame qué activo te interesa."
            )

        except Exception as e:
            logging.error(f"[chat_id={chat_id}] Error inesperado: {e}", exc_info=True)
            self._historiales.pop(chat_id, None)
            await update.message.reply_text(
                "Ocurrió un error inesperado. Intentá de nuevo en unos segundos."
            )

    def run(self):
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