# agente_orquestador.py

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from iol_api import IOL_API
from gmail_api import GmailAPI
from agente_investigador import investigar_activo
from agente_analista import analizar_sentimiento_noticias
from logger_config import get_logger

# Cargar variables de entorno
load_dotenv()

# Configurar logger
logger = get_logger(__name__)

# Configuración de la API de IOL
IOL_USERNAME = os.getenv("IOL_USERNAME")
IOL_PASSWORD = os.getenv("IOL_PASSWORD")

# Configuración de la API de Gmail
GMAIL_SENDER = os.getenv("GMAIL_SENDER")
GMAIL_RECIPIENT = os.getenv("GMAIL_RECIPIENT")

# --- Inicialización de APIs ---
try:
    logger.info("Inicializando APIs...")
    iol_api = IOL_API(IOL_USERNAME, IOL_PASSWORD)
    gmail_api = GmailAPI()
    logger.info("APIs inicializadas correctamente.")
except Exception as e:
    logger.error(f"Error al inicializar las APIs: {e}")
    exit()

# --- Funciones Auxiliares ---

def _obtener_precio_iol(ticker):
    """
    Obtiene el precio actual de un ticker desde la API de IOL.
    Maneja la autenticación y posibles errores.
    """
    try:
        if not iol_api.is_token_valid():
            logger.info("Token de IOL expirado o no válido. Refrescando...")
            iol_api.refresh_token()
            logger.info("Token de IOL refrescado correctamente.")

        logger.info(f"Obteniendo cotización para {ticker}...")
        cotizacion = iol_api.get_cotizacion(ticker)
        if cotizacion:
            precio = cotizacion.get('ultimoPrecio')
            logger.info(f"Precio obtenido para {ticker}: {precio}")
            return precio
        else:
            logger.warning(f"No se pudo obtener la cotización para {ticker}.")
            return None
    except Exception as e:
        logger.error(f"Error al obtener precio de IOL para {ticker}: {e}")
        return None


# --- Lógica Principal (EXEC) ---

def ejecutar_estrategia(ticker, umbral_compra, umbral_venta):
    """
    Ejecuta la estrategia de trading para un ticker específico.
    """
    logger.info(f"--- Iniciando estrategia para {ticker} ---")

    # 1. Obtener precio actual
    precio_actual = _obtener_precio_iol(ticker)
    if precio_actual is None:
        logger.error(f"No se pudo obtener el precio para {ticker}. Abortando estrategia.")
        return

    # 2. Investigar activo (noticias)
    noticias = investigar_activo(ticker)
    if not noticias:
        logger.warning(f"No se encontraron noticias para {ticker}. No se puede realizar análisis de sentimiento.")
        # Podríamos decidir continuar sin análisis de sentimiento o abortar.
        # Por ahora, continuamos pero el sentimiento será neutro.
        sentimiento_general = "NEUTRO"
        confianza = 0.5
    else:
        # 3. Analizar sentimiento de las noticias
        sentimiento_general, confianza = analizar_sentimiento_noticias(noticias)

    # 4. Tomar decisión
    decision = "MANTENER"
    if sentimiento_general == "POSITIVO" and confianza > 0.7:
        # Simulación de lógica de compra basada en un precio objetivo ficticio
        # En un caso real, aquí iría una lógica más compleja (ej. análisis técnico)
        precio_objetivo_compra = precio_actual * (1 - umbral_compra)
        logger.info(f"Sentimiento POSITIVO. Precio actual: {precio_actual}, Precio objetivo compra: {precio_objetivo_compra}")
        if precio_actual <= precio_objetivo_compra:
            decision = "COMPRAR"
    elif sentimiento_general == "NEGATIVO" and confianza > 0.7:
        # Simulación de lógica de venta
        precio_objetivo_venta = precio_actual * (1 + umbral_venta)
        logger.info(f"Sentimiento NEGATIVO. Precio actual: {precio_actual}, Precio objetivo venta: {precio_objetivo_venta}")
        if precio_actual >= precio_objetivo_venta:
            decision = "VENDER"

    logger.info(f"Decisión para {ticker}: {decision} (Sentimiento: {sentimiento_general}, Confianza: {confianza:.2f})")

    # 5. Ejecutar operación y notificar (si es necesario)
    if decision in ["COMPRAR", "VENDER"]:
        asunto = f"Alerta de Trading: {decision} {ticker}"
        cuerpo = f"""
        Hola,

        Se ha generado una recomendación de trading para el activo {ticker}.

        - Decisión: {decision}
        - Precio Actual: ${precio_actual}
        - Sentimiento de Mercado: {sentimiento_general} (Confianza: {confianza:.2f})
        - Fecha y Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Por favor, revise su plataforma de trading para confirmar la operación.

        Saludos,
        Su Asistente de Trading Automatizado
        """
        try:
            logger.info(f"Enviando notificación por correo para {decision} {ticker}...")
            gmail_api.send_email(GMAIL_SENDER, GMAIL_RECIPIENT, asunto, cuerpo)
            logger.info("Notificación enviada correctamente.")
        except Exception as e:
            logger.error(f"Error al enviar la notificación por correo: {e}")

    logger.info(f"--- Estrategia para {ticker} finalizada ---")


if __name__ == "__main__":
    # --- Configuración de la Estrategia ---
    TICKERS_A_SEGUIR = ["GGAL", "PAMP", "YPFD"] # Ejemplo de tickers
    UMBRAL_COMPRA = 0.02  # 2% por debajo del precio actual para considerar compra
    UMBRAL_VENTA = 0.03   # 3% por encima del precio actual para considerar venta
    INTERVALO_EJECUCION_MINUTOS = 15

    logger.info("Iniciando el Agente Orquestador de Trading.")
    logger.info(f"Tickers a monitorear: {TICKERS_A_SEGUIR}")
    logger.info(f"Intervalo de ejecución: {INTERVALO_EJECUCION_MINUTOS} minutos")

    while True:
        try:
            for ticker in TICKERS_A_SEGUIR:
                ejecutar_estrategia(ticker, UMBRAL_COMPRA, UMBRAL_VENTA)
                time.sleep(5) # Pequeña pausa entre tickers para no saturar la API

            logger.info(f"Ciclo completado. Esperando {INTERVALO_EJECUCION_MINUTOS} minutos para la próxima ejecución.")
            time.sleep(INTERVALO_EJECUCION_MINUTOS * 60)

        except KeyboardInterrupt:
            logger.info("Proceso interrumpido por el usuario. Finalizando...")
            break
        except Exception as e:
            logger.critical(f"Ocurrió un error inesperado en el bucle principal: {e}")
            logger.info("Esperando 5 minutos antes de reintentar...")
            time.sleep(300) # Espera 5 minutos antes de reintentar en caso de error grave
