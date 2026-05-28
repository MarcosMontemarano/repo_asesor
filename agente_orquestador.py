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
import time
from iolConn import Iol
import google.genai as genai
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

import ticker_resolver as tr
from agente_fundamental import AgenteFundamental
from agente_tecnico import AgenteTecnico

load_dotenv(override=True)

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
- "capital": SOLO dígitos. Elimina siempre puntos, comas y símbolos de moneda. "100k" → "100000". "$50.000" → "50000". "60.000" → "60000". Vacío si no hay monto.
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


def obtener_precio_iol(ticker: str) -> float:
    """
    Obtiene el precio de un activo desde la API de IOL en tiempo real.
    Utiliza iolConn y maneja credenciales desde el entorno.
    Devuelve 0.0 si no se puede obtener el precio.
    """
    try:
        iol_user = os.getenv("IOL_USER")
        iol_password = os.getenv("IOL_PASSWORD")

        if not iol_user or not iol_password:
            logging.warning("Credenciales de IOL (IOL_USER, IOL_PASSWORD) no configuradas en .env. No se puede obtener precio real.")
            return 0.0

        # Nota: instanciar y loguear en cada llamada no es óptimo.
        # En una app de producción, se podría crear un cliente singleton.
        iol = Iol(username=iol_user, password=iol_password)
        iol.login()

        # El ticker canónico (ej: GGAL, AL30) es el que usa IOL para el mercado local.
        cotizacion = iol.get_instrumento_cotizacion(
            simbolo=ticker,
            mercado="bCBA"  # Mercado de Buenos Aires
        )

        if cotizacion and cotizacion.get('ultimoPrecio'):
            precio = float(cotizacion['ultimoPrecio'])
            logging.info(f"Precio de {ticker} obtenido de IOL: {precio}")
            return precio
        else:
            logging.warning(f"No se pudo obtener 'ultimoPrecio' para {ticker} desde IOL. Respuesta: {cotizacion}")
            return 0.0

    except Exception as e:
        logging.warning(f"No se pudo obtener precio de {ticker} desde IOL: {e}")
        return 0.0  # Devolver 0 para que el flujo principal no se rompa.


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
            self.genai_model = "gemini-flash-latest"
            # Instanciar agentes especializados
            self.agente_fundamental = AgenteFundamental(api_key=gemini_api_key)
            self.agente_tecnico = AgenteTecnico(api_key=gemini_api_key)
        except Exception as e:
            raise RuntimeError(f"Error al configurar Gemini o los agentes: {e}") from e

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

                # --- Dimensionador de Posición ---
                precio_actual = obtener_precio_iol(ticker)
                nominales_reales = 0
                if precio_actual > 0:
                    nominales_reales = int(float(capital) // precio_actual)

                # --- Obtener ticker para yfinance ---
                ticker_yf = tr.obtener_ticker_yfinance(ticker)

                # --- Agentes especializados ---
                reporte_fundamental = "Análisis fundamental no disponible."
                reporte_tecnico = "Análisis técnico no disponible."

                # 1. Agente Fundamental
                if ticker_yf:
                    try:
                        logging.info(f"[EXEC] Iniciando análisis fundamental para {ticker_yf}")
                        reporte_fundamental = self.agente_fundamental.analizar_activo(ticker_yf)
                    except Exception as e:
                        logging.error(f"Error en AgenteFundamental: {e}", exc_info=True)
                        reporte_fundamental = f"Error al generar análisis fundamental: {e}"
                else:
                    meta_bono = tr.obtener_metadata_bono(ticker)
                    if meta_bono:
                        reporte_fundamental = (
                            f"Sentimiento NEUTRAL para {ticker}: bono {meta_bono['legislacion']}, "
                            f"vencimiento {meta_bono['vencimiento']}, cupón {meta_bono['cupon_anual']}."
                        )

                # Pausa para no saturar la API de Gemini
                time.sleep(2)

                # 2. Agente Técnico
                rsi_prueba = 35
                media_movil_prueba = 14000

                if ticker_yf:
                    try:
                        logging.info(f"[EXEC] Iniciando análisis técnico para {ticker_yf}")
                        # Se ejecuta el llamado al agente como fue diseñado. Los placeholders
                        # se definen para cumplir el request, pero el agente es autocontenido.
                        reporte_tecnico = self.agente_tecnico.analizar_activo(ticker_yf)
                    except Exception as e:
                        logging.error(f"Error en AgenteTecnico: {e}", exc_info=True)
                        reporte_tecnico = f"Error al generar análisis técnico: {e}"
                else:
                    reporte_tecnico = "Análisis técnico no disponible para bonos."

                # 3. Combinar resultados y generar veredicto final con Gemini
                prompt_final = (
                    f"Sos el Asesor Financiero final de IOL. Cliente amateur.\n"
                    f"Activo: {ticker} | Capital: ${capital} | Riesgo: {riesgo}\n\n"
                    f"Análisis Fundamental: {reporte_fundamental}\n\n"
                    f"Análisis Técnico: {reporte_tecnico}\n"
                    f"REGLA MATEMÁTICA: El precio actual de {ticker} en IOL es ${precio_actual:.2f} ARS. "
                    f"Con el capital de ${capital} ARS, el usuario puede comprar EXACTAMENTE {nominales_reales} nominales. "
                    "TIENES TOTALMENTE PROHIBIDO inventar precios o calcular cantidades. Usa exclusivamente estos números duros en tu veredicto.\n\n"
                    "Sintetizá la decisión SIN jerga técnica. "
                    "Devolvé EXACTAMENTE este formato:\n\n"
                    f"Veredicto: [COMPRAR / VENDER / RETENER] — [{nominales_reales} nominales]\n"
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