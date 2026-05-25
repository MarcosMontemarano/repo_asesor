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
import re
import json
import requests
import google.genai as genai
from dotenv import load_dotenv
from google.genai import types
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# Cargar variables de entorno desde el archivo .env
# Esto permite leer TELEGRAM_TOKEN y GEMINI_API_KEY de forma segura.
load_dotenv()

# Habilitar logging para depuración
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class AgenteOrquestador:
    """
    Clase que gestiona el bot de Telegram, recibe las solicitudes de los usuarios,
    y orquesta las llamadas a los agentes de análisis para dar una respuesta final.
    """

    def __init__(self, telegram_token: str, gemini_api_key: str):
        """
        Inicializa el bot, los clientes de API y los manejadores de mensajes.
        """
        if not telegram_token:
            raise ValueError("El token de Telegram no fue proporcionado.")
        if not gemini_api_key:
            raise ValueError("La API Key de Gemini no fue proporcionada.")

        # Configuración de Telegram
        self.application = ApplicationBuilder().token(telegram_token).build()

        # Configuración de Gemini AI
        try:
            self.genai_client = genai.Client(api_key=gemini_api_key)
            self.genai_model = "gemini-flash-latest"
            self.system_prompt = (
                'Eres un asesor financiero experto en IOL Argentina. El usuario busca '
                'respuestas directas de COMPRA/VENTA/RETENCIÓN con el porcentaje de '
                'capital a usar. Propón siempre una estrategia clara.'
            )
            # System prompt para el router conversacional
            self.router_system_prompt = '''Eres el enrutador de un sistema financiero conectado a InvertirOnline (IOL). NO memorices bases de datos ni inventes a qué se dedican las empresas. Tu único trabajo es clasificar la intención del usuario, extraer datos y devolver un JSON estricto. REGLAS INQUEBRANTABLES:

    Si el usuario menciona un ticker exacto y real del mercado (ej. SPY, AL30, GGAL), devuelve: {"accion": "EXEC", "ticker": "[ticker]", "capital": "[capital]", "riesgo": "[riesgo]"}.
    Si el usuario pide invertir en una clase de activo (ej. bonos, cedears, acciones, fci) pero NO da un ticker exacto, debes delegar la búsqueda a la base de datos externa. Devuelve INMEDIATAMENTE: {"accion": "BUSCAR", "categoria": "[clase de activo mencionada]", "capital": "[capital]", "riesgo": "[riesgo]"}.
    Si falta el capital o el riesgo, devuelve: {"accion": "CHAT", "mensaje": "[tu pregunta breve y natural para obtener los datos faltantes]"}.'''
            # La instanciación del modelo con memoria (chat) se hará por usuario
            # en handle_message para evitar el uso de la clase obsoleta GenerativeModel.
        except Exception as e:
            raise RuntimeError(f"Error al configurar el cliente de IA de Gemini: {e}") from e

        # Registrar los manejadores de comandos y mensajes
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Responde al comando /start con un saludo proactivo y sugerencias.
        """
        mensaje = (
            "¡Hola! Soy tu asesor financiero experto en IOL (InvertirOnline).\n\n"
            "Puedo ayudarte a analizar activos. Simplemente envíame el ticker de un "
            "activo que te interese (ej. 'GGAL', 'AAPL', 'AL30').\n\n"
            "¿Buscas ideas? Podríamos explorar:\n"
            "✅ **CEDEARs**: Para invertir en empresas de USA en pesos.\n"
            "✅ **Bonos**: Opciones de renta fija en dólares o pesos.\n"
            "✅ **Cauciones**: Para inversiones a muy corto plazo."
        )
        await update.message.reply_text(mensaje)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Maneja los mensajes de texto, mantiene una conversación para obtener los datos
        necesarios y luego orquesta el análisis.
        """
        chat_id = update.effective_chat.id
        texto_usuario = update.message.text
        historial = context.user_data.get('historial', [])
        historial.append(f"Usuario: {texto_usuario}")

        try:
            # --- LLM Router Híbrido: Usando Ollama local ---
            full_prompt = f"{self.router_system_prompt}\n\n--- Historial ---\n" + "\n".join(historial)

            payload = {
                "model": "llama3.2",
                "prompt": full_prompt,
                "stream": False,
                "format": "json"
            }

            ollama_response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
            ollama_response.raise_for_status()

            json_response = ollama_response.json()
            respuesta_ia_str = json_response['response'].strip()
            logging.info(f"RAW IA JSON string: {respuesta_ia_str}")

            try:
                datos_ia = json.loads(respuesta_ia_str)
                accion = datos_ia.get("accion")

                if accion == "CHAT":
                    mensaje = datos_ia.get("mensaje", "No sé qué decir. ¿Podemos intentarlo de nuevo?")
                    historial.append(f"Asesor: {mensaje}")
                    context.user_data['historial'] = historial
                    await update.message.reply_text(mensaje)

                elif accion == "EXEC":
                    ticker = datos_ia.get("ticker")
                    capital = datos_ia.get("capital")
                    riesgo = datos_ia.get("riesgo")

                    if not all([ticker, capital, riesgo]):
                        raise ValueError(f"Faltan datos en la respuesta EXEC del JSON: {datos_ia}")
                    
                    context.user_data.pop('historial', None) # Limpiar historial para la próxima consulta

                    await update.message.reply_text(f"¡Excelente! He reunido toda la información. Analizando {ticker}... un momento por favor.")

                    # --- MOCK: Llamadas a los otros agentes (que siguen usando Gemini) ---
                    payload_str = f"TICKER:{ticker}|CAPITAL:{capital}|RIESGO:{riesgo}"
                    logging.info(f"[MOCK] Llamando a AgenteTecnico con el payload: {payload_str}")
                    reporte_tecnico = f"Análisis Técnico (mock): Señal de COMPRA para {ticker} basada en RSI bajo y cruce de medias móviles."

                    logging.info(f"[MOCK] Llamando a AgenteFundamental con el payload: {payload_str}")
                    reporte_fundamental = f"Análisis Fundamental (mock): Sentimiento POSITIVO para {ticker} por noticias de expansión de mercado."
                    # --- FIN DEL MOCK ---

                    prompt_final = (
                        f"Eres el Asesor Financiero final de IOL. Tu cliente es amateur. Aquí tienes los análisis del activo {ticker}: "
                        f"Técnico: {reporte_tecnico} Fundamental: {reporte_fundamental} Capital: {capital} | Riesgo: {riesgo}\n"
                        "Tu regla inquebrantable: Sintetiza la decisión sin usar jerga compleja. Debes cruzar los datos y devolver EXACTAMENTE este formato y nada más:\n"
                        "Veredicto: [COMPRAR / VENDER / RETENER] [Cantidad aproximada de nominales a operar basándote en el capital]. "
                        "Explicaciones breves: [Máximo 2 líneas resumiendo el motivo de la decisión de forma muy simple y directa].\n"
                        "¿Querés que profundice en algún detalle técnico o fundamental, o procedemos a ver otra opción?"
                    )

                    response = self.genai_client.models.generate_content(model=self.genai_model, contents=[prompt_final])
                    await update.message.reply_text(response.text.strip())
                else:
                    # Esto maneja casos donde 'accion' está ausente o tiene un valor desconocido
                    raise ValueError(f"Acción desconocida o faltante en la respuesta JSON: {accion}")

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.error(f"Error al procesar la respuesta JSON del router: {e}. Respuesta recibida: '{respuesta_ia_str}'")
                context.user_data.pop('historial', None) # Limpiar sesión
                await update.message.reply_text("No estoy seguro de cómo proceder. ¿Podemos empezar de nuevo? Por favor, dime qué activo te interesa.")

        except requests.exceptions.ConnectionError:
            logging.error("No se pudo conectar con el servidor de Ollama en http://127.0.0.1:11434")
            await update.message.reply_text("⚠️ Aviso del sistema: El motor de IA local no está disponible. Por favor, contacta al administrador.")
            context.user_data.pop('historial', None)
        except Exception as e:
            logging.error(f"Error al procesar el mensaje con el router local: {e}")
            await update.message.reply_text("Lo siento, ocurrió un error inesperado al procesar tu solicitud.")
            context.user_data.pop('historial', None)

    def run(self):
        """Inicia el bot y lo mantiene corriendo en modo polling."""
        logging.info("Iniciando bot...")
        self.application.run_polling(drop_pending_updates=True)


if __name__ == '__main__':
    # Cargar tokens desde variables de entorno para seguridad
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not TELEGRAM_TOKEN:
        logging.error("La variable de entorno TELEGRAM_TOKEN no está configurada. Revisa tu archivo .env.")
        sys.exit(1)
    if not GEMINI_API_KEY:
        logging.error("La variable de entorno GEMINI_API_KEY no está configurada. Revisa tu archivo .env.")
        sys.exit(1)

    try:
        bot = AgenteOrquestador(telegram_token=TELEGRAM_TOKEN, gemini_api_key=GEMINI_API_KEY)
        bot.run()
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error al inicializar el bot: {e}")
        sys.exit(1)
