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
import re
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
# Cambios respecto a la versión anterior:
#
#   Patrón 1 — Llama inventa RECOMENDAR/ANALIZAR cuando tiene "suficiente info":
#     Solución: regla positiva explícita — si el usuario menciona MÁS DE UN
#     activo, la acción es SIEMPRE CHAT sin excepción. EXEC solo para uno.
#
#   Patrón 2 — Nombre largo de empresa (Pampa Energía, ETF del Nasdaq) no
#     dispara EXEC aunque el modelo conozca el ticker:
#     Solución: regla explícita — si en tu respuesta ibas a escribir un ticker
#     concreto del catálogo y tenés capital y riesgo, eso es EXEC, no CHAT.
#     + Ejemplos de mapeo nombre→ticker para los casos que fallaron.
#
#   Patrón 3 — JSON vacío {}:
#     Solución: en Python (_normalizar_claves y chequeo de dict vacío).
#     En el prompt: se refuerza que la respuesta SIEMPRE debe tener "accion".
# ──────────────────────────────────────────────────────────────────────────────
ROUTER_INSTRUCTIONS = f"""
{IOL_CATALOG}

═══════════════════════════════════════════
 TU ROL
═══════════════════════════════════════════
Eres el enrutador de AFMA, asesor financiero de IOL Argentina.
Leé el mensaje del usuario y extraé: CAPITAL, RIESGO y TICKER.
Devolvé UN SOLO objeto JSON. Sin texto fuera del JSON.
La respuesta SIEMPRE debe contener la clave "accion". Nunca devuelvas {{}}.
La clave de acción se escribe EXACTAMENTE así: "accion" (sin tilde).
La clave del ticker se escribe EXACTAMENTE así: "ticker" (sin e al final).

═══════════════════════════════════════════
 QUÉ ES UN TICKER
═══════════════════════════════════════════
Un ticker es el código corto de un activo en la bolsa.
Cuando uses un ticker en tu mensaje, aclarás el nombre entre paréntesis.
Ejemplos de mapeo nombre → ticker:
  "YPF" o "YPF S.A."        → YPFD
  "Pampa" o "Pampa Energía" → PAMP
  "Google" o "Alphabet"     → GOOGL
  "MercadoLibre"            → MELI
  "ETF del Nasdaq"          → QQQ
  "ETF del S&P" o "S&P500"  → SPY
  "Galicia"                 → GGAL

═══════════════════════════════════════════
 EXTRACCIÓN DE DATOS
═══════════════════════════════════════════
CAPITAL : un número. Solo dígitos. NUNCA nombres de empresas en capital.
          Si no hay monto, capital va vacío: "".
RIESGO  : bajo / medio / alto. Sin señal clara = medio.
TICKER  : usá el catálogo y la tabla de mapeo de arriba.
          Si menciona varios activos → CHAT (ver regla crítica abajo).

═══════════════════════════════════════════
 REGLAS CRÍTICAS — LEER ANTES DE RESPONDER
═══════════════════════════════════════════
REGLA 1 — UN SOLO ACTIVO PARA EXEC:
  Si el usuario menciona MÁS DE UN activo en el mismo mensaje,
  la acción es SIEMPRE CHAT. Nunca EXEC con múltiples tickers.
  Ejemplo: "SPY y AAPL con 100000" → CHAT, no EXEC.

REGLA 2 — NOMBRE DE EMPRESA CON CAPITAL Y RIESGO ES EXEC:
  Si el usuario nombra UNA empresa o producto financiero (aunque use el
  nombre largo), y tenés capital y riesgo → la acción es EXEC con el
  ticker mapeado del catálogo.
  Ejemplo: "analizá Pampa Energía con 60000 riesgo alto" → EXEC ticker=PAMP
  Ejemplo: "el ETF del Nasdaq con 30000 riesgo medio"   → EXEC ticker=QQQ
  Ejemplo: "analizá Google con 120000 riesgo alto"       → EXEC ticker=GOOGL

REGLA 3 — JSON NUNCA VACÍO:
  La respuesta siempre tiene "accion". Si no sabés qué hacer, usá CHAT
  con un mensaje pidiendo más información.

═══════════════════════════════════════════
 ACCIONES VÁLIDAS — SOLO ESTAS DOS
═══════════════════════════════════════════
"CHAT" : cuando falta CAPITAL, RIESGO o TICKER, o el usuario menciona
         múltiples activos, pide distribución, planes, gráficos, noticias,
         preguntas de seguimiento o cualquier cosa fuera del análisis puntual.
         Formato: {{"accion": "CHAT", "mensaje": "[respuesta breve]"}}

"EXEC" : solo cuando tenés UN ticker + CAPITAL (número) + RIESGO confirmados.
         Formato: {{"accion": "EXEC", "ticker": "[TICKER]", "capital": "[número]", "riesgo": "[nivel]"}}

PROHIBIDO: RECOMENDAR, ANALIZAR, BUSCAR, DISTRIBUIR, o cualquier otra acción.
PROHIBIDO: "acción" con tilde. Siempre "accion".
PROHIBIDO: "ticket". Siempre "ticker".
PROHIBIDO: nombres de empresas en el campo capital.
PROHIBIDO: EXEC con múltiples tickers.
PROHIBIDO: devolver JSON vacío {{}}.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de normalización y validación
# ──────────────────────────────────────────────────────────────────────────────

def _normalizar_claves(datos: dict) -> dict:
    """Corrige typos conocidos en claves del JSON de Ollama."""
    if "accion" not in datos:
        for variante in ("acción", "Accion", "Acción", "ACTION", "action"):
            if variante in datos:
                datos["accion"] = datos.pop(variante)
                logging.warning(f"Clave '{variante}' normalizada a 'accion'.")
                break
    if "ticker" not in datos:
        for variante in ("ticket", "Ticker", "TICKER", "tiker"):
            if variante in datos:
                datos["ticker"] = datos.pop(variante)
                logging.warning(f"Clave '{variante}' normalizada a 'ticker'.")
                break
    return datos


def _extraer_numero(valor) -> str:
    """Extrae solo dígitos del campo capital. Devuelve '' si no hay número."""
    if valor is None:
        return ""
    solo_digitos = re.sub(r"[^\d]", "", str(valor))
    return solo_digitos


def _validar_exec(datos: dict) -> tuple[bool, str]:
    """Valida que EXEC tenga ticker, capital numérico y riesgo."""
    ticker  = datos.get("ticker", "").strip().upper()
    capital = _extraer_numero(datos.get("capital", ""))
    riesgo  = str(datos.get("riesgo", "")).strip()

    if not ticker:
        return False, "falta_ticker"
    if not capital:
        return False, "falta_capital"
    if not riesgo:
        return False, "falta_riesgo"
    return True, ""


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
        Historial primero, instrucciones después.
        Truncado a 10 turnos para no exceder el contexto de Llama.
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

        # Patrón 3: JSON vacío — degradar a CHAT antes de cualquier otra lógica
        if not raw or raw == "{}":
            logging.warning(f"[chat_id={chat_id}] JSON vacío recibido. Degradando a CHAT.")
            return {
                "accion": "CHAT",
                "mensaje": "Entiendo lo que buscás. ¿Me confirmás qué activo querés analizar primero?"
            }

        datos = json.loads(raw)

        # Patrón 3: dict vacío después de parsear
        if not datos:
            logging.warning(f"[chat_id={chat_id}] Dict vacío tras parseo. Degradando a CHAT.")
            return {
                "accion": "CHAT",
                "mensaje": "Entiendo lo que buscás. ¿Me confirmás qué activo querés analizar primero?"
            }

        datos = _normalizar_claves(datos)
        accion = str(datos.get("accion", "")).upper()

        if accion not in ("CHAT", "EXEC"):
            logging.warning(
                f"[chat_id={chat_id}] Acción inválida '{accion}'. "
                f"Degradando a CHAT. Payload: {datos}"
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
        logging.info(f"[chat_id={chat_id}] /start. Historial reseteado.")

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
                    f"Historial: {len(historial)} entradas."
                )
                await update.message.reply_text(mensaje)

            # ── CASO EXEC ─────────────────────────────────────────────────
            elif accion == "EXEC":
                datos_ia["capital"] = _extraer_numero(datos_ia.get("capital", ""))
                datos_ia["ticker"]  = str(datos_ia.get("ticker", "")).strip().upper()

                es_valido, motivo = _validar_exec(datos_ia)

                if not es_valido:
                    mensajes_error = {
                        "falta_capital": (
                            "Ya tengo el activo y el riesgo. "
                            "¿Con cuánto capital contás para esta inversión?"
                        ),
                        "falta_ticker": "¿Qué activo específico querés analizar?",
                        "falta_riesgo": (
                            "¿Cuál es tu tolerancia al riesgo? "
                            "Podés elegir: bajo, medio o alto."
                        ),
                    }
                    msg_error = mensajes_error.get(
                        motivo,
                        "Me faltó un dato. ¿Podés repetir el ticker, capital y riesgo?"
                    )
                    logging.error(
                        f"[chat_id={chat_id}] EXEC inválido — motivo: {motivo}. "
                        f"Payload: {datos_ia}"
                    )
                    historial.append(f"Asesor: {msg_error}")
                    self._historiales[chat_id] = historial
                    await update.message.reply_text(msg_error)
                    return

                ticker  = datos_ia["ticker"]
                capital = datos_ia["capital"]
                riesgo  = datos_ia["riesgo"]

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