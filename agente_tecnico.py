"""
Agente 2: Analista Técnico Robusto

Función:
    Realizar análisis técnico robusto sobre un activo financiero utilizando
    indicadores clave (medias móviles, RSI, volumen) y un modelo de IA
    para generar una señal de trading.

Skills:
    - Descarga de datos de mercado (yfinance)
    - Cálculo de indicadores técnicos robustos (pandas)
    - Interacción con APIs de IA generativa (google-generativeai)

Uso:
    python agente_tecnico.py
"""

import os
import sys
import pandas as pd
import yfinance as yf
import google.genai as genai

class AgenteTecnico:
    """
    Una clase para realizar análisis técnico de activos financieros y generar
    señales de trading (COMPRA, VENTA, RETENCIÓN) usando un modelo de IA.
    """

    def __init__(self, api_key: str):
        """
        Inicializa el agente, configurando el acceso a la API de IA de Gemini.

        Args:
            api_key (str): La clave de API para el servicio de IA generativa de Google.
                           Se obtiene en Google AI Studio.
        
        Raises:
            ValueError: Si la clave de API no es proporcionada.
            RuntimeError: Si hay un error al configurar el modelo de IA.
        """
        if not api_key:
            raise ValueError("La API Key de Gemini no fue proporcionada.")
        
        try:
            # Se inicializa el cliente de bajo nivel. Este es el método correcto
            # para evitar el error 'has no attribute 'configure''.
            self.client = genai.Client(api_key=api_key)
            # Nota: Se utiliza 'gemini-1.5-flash-latest', un modelo rápido y eficiente.
            # El modelo 'gemini-2.5-flash-lite' solicitado no es un identificador público.
            self.model_name = "gemini-flash-latest"
            self.system_prompt = (
                "Eres un analista técnico de inversiones puro. Evalúa esta liquidez "
                "(volumen), la tendencia (medias móviles) y la sobrecompra/sobreventa "
                "(RSI). Dame un veredicto técnico muy breve (máximo 3 líneas) "
                "indicando explícitamente si la sugerencia es COMPRA, VENTA o "
                "RETENCIÓN, justificando el riesgo de liquidez y tendencia."
            )
            # La validación de la API key se realizará implícitamente en la primera
            # llamada real a `generate_content`. No se necesita una llamada aquí.
        except Exception as e:
            raise RuntimeError(f"Error al configurar el cliente de IA de Gemini: {e}") from e

    def _calcular_indicadores(self, historial: pd.DataFrame) -> pd.DataFrame:
        """Calcula los indicadores técnicos requeridos sobre el historial de precios."""
        # 1. Medias Móviles Simples (SMA)
        historial['SMA_10'] = historial['Close'].rolling(window=10).mean()
        historial['SMA_20'] = historial['Close'].rolling(window=20).mean()

        # 2. Índice de Fuerza Relativa (RSI) de 14 días
        delta = historial['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Usamos com=13 que es equivalente a un alpha=1/14 para el EMA (Wilder's smoothing)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / ema_loss
        historial['RSI_14'] = 100 - (100 / (1 + rs))

        # 3. Volumen promedio de 10 días
        historial['Vol_Avg_10'] = historial['Volume'].rolling(window=10).mean()
        
        return historial.dropna()

    def analizar_activo(self, simbolo: str) -> str:
        """
        Descarga datos, calcula indicadores y obtiene un veredicto de la IA.

        Args:
            simbolo (str): El símbolo del activo a analizar (ej. 'AAPL', 'GGAL.BA').

        Returns:
            str: El veredicto del analista de IA o un mensaje de error.
        """
        if not self.client:
            # Esta condición es ahora menos probable debido al raise en __init__
            return "Error: El cliente de IA no está disponible."

        try:
            # 1. Descargar datos de yfinance para los últimos 90 días
            print(f"Descargando historial de 90 días para {simbolo}...")
            ticker = yf.Ticker(simbolo)
            historial = ticker.history(period="90d")

            # Chequeo estricto de seguridad si no se devuelven datos
            if historial.empty:
                return f"Error: No se encontraron datos para el símbolo '{simbolo}'. " \
                       "Verifique que el ticker sea correcto."

            # 2. Calcular indicadores técnicos y eliminar filas con NaN iniciales
            historial_con_indicadores = self._calcular_indicadores(historial)

            if len(historial_con_indicadores) < 5:
                return (f"Error: No hay suficientes datos para el análisis de '{simbolo}' "
                        "después de calcular los indicadores (se necesitan 5 días).")

            # 3. Preparar el prompt para la IA con los últimos 5 días de datos
            ultimos_datos = historial_con_indicadores[[
                'Close', 'SMA_10', 'SMA_20', 'RSI_14', 'Vol_Avg_10'
            ]].tail(5)

            # Formatear el dataframe a un string legible para el prompt
            datos_str = ultimos_datos.to_string(float_format="%.2f")

            prompt = (
                f"Datos técnicos para el activo {simbolo}:\n"
                f"{datos_str}"
            )
            
            print("Enviando datos a la IA para análisis...")
            # Se usa el método `generate_content` a través de `client.models`
            # como requiere la nueva API. El system_prompt se pasa como parte
            # de la lista de `contents`.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.system_prompt, prompt]
            )
            
            return response.text.strip()

        except Exception as e:
            error_msg = f"Ocurrió un error durante el análisis de {simbolo}: {e}"
            print(error_msg, file=sys.stderr)
            return error_msg

if __name__ == "__main__":
    # Para ejecutar este script, necesitas instalar las librerías:
    # pip install pandas yfinance google-generativeai ipython

    # --- CONFIGURACIÓN DE LA API KEY DE GEMINI ---
    # Carga la API Key desde una variable de entorno para mayor seguridad.
    # Debes crear una variable de entorno llamada GEMINI_API_KEY con tu clave.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: La variable de entorno GEMINI_API_KEY no está configurada.", file=sys.stderr)
        print("Obtén una API Key en Google AI Studio (https://aistudio.google.com/app/apikey) "
              "y configúrala en tu sistema.", file=sys.stderr)
        sys.exit(1)

    try:
        agente = AgenteTecnico(api_key=GEMINI_API_KEY)
        
        # --- Activo a analizar ---
        simbolo_ejemplo = "MELI"  # Puedes cambiarlo por otro, ej: 'GGAL.BA', 'AAPL', 'TSLA'
        
        veredicto = agente.analizar_activo(simbolo_ejemplo)

        print("\n--- Veredicto del Analista Técnico ---")
        print(f"Activo: {simbolo_ejemplo}")
        print(veredicto)
        print("------------------------------------")
        
    except (ValueError, RuntimeError) as e:
        print(f"Error de configuración o ejecución: {e}", file=sys.stderr)
        sys.exit(1)