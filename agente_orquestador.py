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

ROUTER_INSTRUCTIONS = f"""
{IOL_CATALOG}

═══════════════════════════════════════════
 TU ROL
═══════════════════════════════════════════
Eres el enrutador de AFMA, asesor financiero de IOL Argentina.
Leé el historial de arriba y extraé: CAPITAL, RIESGO y TICKER.
Devolvé UN SOLO objeto JSON. Sin texto fuera del JSON.
La clave de acción se escribe EXACTAMENTE así: "accion" (sin tilde).
La clave del ticker se escribe EXACTAMENTE así: "ticker" (sin e al final).

═══════════════════════════════════════════
 QUÉ ES UN TICKER
═══════════════════════════════════════════
Un ticker es el código corto con el que se identifica una empresa o activo
en la bolsa. Por ejemplo: PAMP es el ticker de Pampa Energía, AAPL es el
ticker de Apple, AL30 es el ticker de un bono soberano argentino.
Cuando el usuario menciona una empresa por nombre, usá el catálogo para
encontrar su ticker exacto. Cuando uses un ticker en tu mensaje al usuario,
siempre aclarás el nombre de la empresa entre paréntesis.
Ejemplo correcto: "PAMP (Pampa Energía)" o "AAPL (Apple)".

═══════════════════════════════════════════
 EXTRACCIÓN DE DATOS
═══════════════════════════════════════════
CAPITAL : un número. Solo dígitos, sin letras ni nombres de empresas.
          Si el usuario NO mencionó un monto, el campo capital va vacío: "".
          NUNCA pongas el nombre de un activo en el campo capital.
RIESGO  : bajo / medio / alto.
          Si no lo dice, inferilo del contexto.
          Sin señal clara = medio.
TICKER  : usá el catálogo para mapear nombres a tickers exactos.
          "YPF" → YPFD | "Pampa" → PAMP | "Apple" → AAPL
          "bonos" sin especificar → preguntá cuál en un mensaje CHAT.

REGLA CRÍTICA DE TICKER:
  El ticker a usar en EXEC es SIEMPRE el último activo mencionado
  explícitamente por el usuario, no el más frecuente ni el primero.

═══════════════════════════════════════════
 ACCIONES VÁLIDAS — SOLO ESTAS DOS
═══════════════════════════════════════════
"CHAT" : cuando falta CAPITAL, RIESGO o TICKER, o el usuario pide
         recomendaciones o distribución de capital.
         → Respondé basándote en el historial real.
         → Si falta el capital, pedilo específicamente.
         → Si falta el ticker, recomendá opciones y preguntá cuál.
         → NUNCA repitas información ya dada en el historial.
         → NUNCA preguntes datos que el usuario ya dio.
         → Al mencionar tickers, siempre aclarás el nombre entre paréntesis.
         Formato: {{"accion": "CHAT", "mensaje": "[respuesta personalizada]"}}

"EXEC" : solo cuando tenés CAPITAL (número) + RIESGO + TICKER confirmados.
         Formato: {{"accion": "EXEC", "ticker": "[TICKER]", "capital": "[número]", "riesgo": "[nivel]"}}

PROHIBIDO: acciones como BUSCAR, RECOMENDAR, ANALIZAR, etc.
PROHIBIDO: usar "acción" con tilde. La clave es siempre "accion".
PROHIBIDO: usar "ticket". La clave es siempre "ticker".
PROHIBIDO: poner nombres de empresas o tickers en el campo "capital".
PROHIBIDO: inventar tickers fuera del catálogo.
PROHIBIDO: preguntar datos que el usuario ya dio en el historial.

═══════════════════════════════════════════
 ESTRUCTURA DE EJEMPLOS
═══════════════════════════════════════════
Caso A — falta el ticker:
  → {{"accion": "CHAT", "mensaje": "[Recomendás 3 tickers con nombre entre paréntesis, adecuados al capital y riesgo del historial. Preguntás cuál analizar primero.]"}}

Caso B — falta el capital:
  → {{"accion": "CHAT", "mensaje": "[Le decís que tenés el ticker y el riesgo, solo falta saber con cuánto capital cuenta.]"}}

Caso C — falta el riesgo:
  → {{"accion": "CHAT", "mensaje": "[Le preguntás si prefiere riesgo bajo, medio o alto.]"}}

Caso D — el usuario pide distribución entre varios activos:
  → {{"accion": "CHAT", "mensaje": "[Proponés un reparto porcentual concreto con nombres entre paréntesis. Informás que los análisis son de a uno. Preguntás por cuál empezar.]"}}

Caso E — el usuario confirma un ticker o da todo junto:
  → {{"accion": "EXEC", "ticker": "[ticker exacto del catálogo]", "capital": "[número]", "riesgo": "[nivel]"}}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de normalización del JSON de Ollama
# Llama 3.2 produce typos previsibles. Los corregimos en Python, no en el prompt,
# porque el prompt ya es suficientemente largo y los modelos pequeños
# no siempre respetan restricciones tipográficas con consistencia.
# ──────────────────────────────────────────────────────────────────────────────

def _normalizar_claves(datos: dict) -> dict:
    """
    Corrige typos conocidos en las claves del JSON de Ollama:
      - "acción" / "Accion" / "ACTION" → "accion"
      - "ticket" / "Ticker" / "TICKER" → "ticker"
    """
    # Normalizar clave de acción
    if "accion" not in datos:
        for variante in ("acción", "Accion", "Acción", "ACTION", "action"):
            if variante in datos:
                datos["accion"] = datos.pop(variante)
                logging.warning(f"Clave '{variante}' normalizada a 'accion'.")
                break

    # Normalizar clave de ticker
    if "ticker" not in datos:
        for variante in ("ticket", "Ticker", "TICKER", "tiker"):
            if variante in datos:
                datos["ticker"] = datos.pop(variante)
                logging.warning(f"Clave '{variante}' normalizada a 'ticker'.")
                break

    return datos


def _extraer_numero(valor) -> str:
    """
    Recibe el valor del campo 'capital' y devuelve solo dígitos.
    Si el valor es numérico, lo convierte a string limpio.
    Si es un string con letras (ej: "YPF", "100k"), extrae solo los dígitos.
    Si no hay dígitos, devuelve cadena vacía.
    """
    if valor is None:
        return ""
    texto = str(valor)
    solo_digitos = re.sub(r"[^\d]", "", texto)
    return solo_digitos


def _validar_exec(datos: dict) -> tuple[bool, str]:
    """
    Valida que un dict EXEC tenga ticker, capital numérico y riesgo.
    Devuelve (es_valido, mensaje_de_error).
    """
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
                # Normalizar capital a número puro antes de validar
                datos_ia["capital"] = _extraer_numero(datos_ia.get("capital", ""))
                datos_ia["ticker"]  = str(datos_ia.get("ticker", "")).strip().upper()

                es_valido, motivo = _validar_exec(datos_ia)

                if not es_valido:
                    # Mensaje específico según qué dato falta
                    mensajes_error = {
                        "falta_capital": (
                            "Ya tengo el activo y el riesgo. "
                            "¿Con cuánto capital contás para esta inversión?"
                        ),
                        "falta_ticker": (
                            "¿Qué activo específico querés analizar?"
                        ),
                        "falta_riesgo": (
                            "¿Cuál es tu tolerancia al riesgo? "
                            "Podés elegir: bajo, medio o alto."
                        ),
                    }
                    msg_error = mensajes_error.get(motivo, "Me faltó un dato. ¿Podés repetir el ticker, capital y riesgo?")
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