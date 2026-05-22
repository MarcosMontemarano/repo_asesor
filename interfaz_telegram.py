"""
Bot de Telegram - Interfaz de Asesor Financiero
Utiliza python-telegram-bot v20.x
"""

import os
import sys
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# Habilitar logging para depuración
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- CONFIGURACIÓN DEL TOKEN ---
# Es mejor práctica cargar los tokens desde variables de entorno para no exponerlos.
TOKEN = os.getenv("TELEGRAM_TOKEN", "8716747750:AAFuxj4Q1KZPuhycbc4D0HjfZBGHvb4bXY8")
if not TOKEN or TOKEN == "TU_TOKEN_AQUI":
    logging.error("No se encontró un token válido. Abortando inicio del bot.")
    sys.exit(1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Responde al comando /start."""
    mensaje = '¡Hola! Soy tu asesor financiero multi-agente. ¿En qué te ayudo hoy?'
    await update.message.reply_text(mensaje)

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Hace eco del mensaje del usuario."""
    # update.message.text contiene el texto enviado por el usuario
    await update.message.reply_text(update.message.text)

def main() -> None:
    """Inicia el bot."""
    # Construir la aplicación con el token
    application = ApplicationBuilder().token(TOKEN).build()

    # Manejadores de comandos y mensajes
    application.add_handler(CommandHandler("start", start))
    
    # Manejador para hacer eco de los mensajes de texto (ignorando comandos)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Ejecutar el bot de forma continua (polling) hasta que el usuario presione Ctrl+C
    application.run_polling()

if __name__ == '__main__':
    main()