"""
Agente 2: Analista Técnico

Función:
    Realizar análisis técnico básico sobre un activo financiero utilizando
    medias móviles y un modelo de IA para generar un veredicto.

Skills:
    - Descarga de datos de mercado (yfinance)
    - Cálculo de indicadores técnicos (pandas)
    - Interacción con APIs de IA generativa (google-generativeai)

Uso:
    python agente_tecnico.py
"""

import os
import sys
import pandas as pd
import yfinance as yf
import google.generativeai as genai

# --- CONFIGURACIÓN DE LA API KEY DE GEMINI ---
# Pega aquí tu API Key de Google AI Studio.
# Es una mejor práctica guardarla como una variable de entorno.
GEMINI_API_KEY = "AIzaSyB4f8X76fnKLUb3oxVtYDGknmbi2tBufHY"

class AgenteTecnico:
    """
    Una clase para realizar análisis técnico de activos financieros.
    """

    def __init__(self, api_key: str):
        """
        Inicializa el agente, configurando el acceso a la API de IA.

        Args:
            api_key (str): La clave de API para el servicio de IA generativa.
        """
        if not api_key or api_key == "PEGA_AQUI_TU_GEMINI_API_KEY":
            raise ValueError("La API Key de Gemini no ha sido configurada. "
                             "Obtén una en Google AI Studio y pégala en el script.")
        
        try:
            genai.configure(api_key=api_key)
            # Usamos gemini-1.5-flash, que es más estable que la versión 'latest'.
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction="Eres un analista técnico de inversiones puro. "
                                   "Analiza estos precios y medias móviles y dame un "
                                   "veredicto técnico muy breve (máximo 3 líneas) "
                                   "indicando si la tendencia sugiere compra, venta o retención."
            )
        except Exception as e:
            print(f"Error al configurar el modelo de IA: {e}", file=sys.stderr)
            self.model = None

    def analizar_activo(self, simbolo: str) -> str:
        """
        Descarga datos, calcula SMAs y obtiene un veredicto de la IA.

        Args:
            simbolo (str): El símbolo del activo a analizar (ej. 'AAPL').

        Returns:
            str: El veredicto del analista de IA o un mensaje de error.
        """
        if not self.model:
            return "Error: El modelo de IA no está disponible."

        try:
            # 1. Descargar datos de yfinance
            print(f"Descargando historial para {simbolo}...")
            historial = yf.Ticker(simbolo).history(period="30d")

            if historial.empty:
                return f"Error: No se encontraron datos para el símbolo '{simbolo}'."

            # 2. Calcular Medias Móviles Simples (SMA)
            historial['SMA_10'] = historial['Close'].rolling(window=10).mean()
            historial['SMA_20'] = historial['Close'].rolling(window=20).mean()

            # Extraer los últimos valores disponibles
            ultimo_cierre = historial['Close'].iloc[-1]
            ultima_sma10 = historial['SMA_10'].iloc[-1]
            ultima_sma20 = historial['SMA_20'].iloc[-1]

            # 3. Preparar el prompt para la IA
            prompt_data = (
                f"Análisis técnico para {simbolo}:\n"
                f"- Precio de Cierre Reciente: {ultimo_cierre:.2f} USD\n"
                f"- Media Móvil de 10 días (SMA 10): {ultima_sma10:.2f} USD\n"
                f"- Media Móvil de 20 días (SMA 20): {ultima_sma20:.2f} USD\n\n"
                "Basado en la relación entre el precio y estas medias móviles, ¿cuál es tu veredicto?"
            )
            
            print("Enviando datos a la IA para análisis...")
            response = self.model.generate_content(prompt_data)
            
            return response.text.strip()

        except Exception as e:
            error_msg = f"Ocurrió un error durante el análisis de {simbolo}: {e}"
            print(error_msg, file=sys.stderr)
            return error_msg

if __name__ == "__main__":
    # Para ejecutar este script, necesitas instalar las librerías:
    # pip install pandas yfinance google-generativeai
    try:
        agente = AgenteTecnico(api_key=GEMINI_API_KEY)
        simbolo_ejemplo = "AAPL"
        veredicto = agente.analizar_activo(simbolo_ejemplo)
        
        print("\n--- Veredicto del Analista Técnico ---")
        print(f"Activo: {simbolo_ejemplo}")
        print(veredicto)
        print("------------------------------------")

    except ValueError as ve:
        print(f"Error de configuración: {ve}", file=sys.stderr)
        sys.exit(1)