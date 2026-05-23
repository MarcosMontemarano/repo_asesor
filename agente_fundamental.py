"""
Agente 3: Analista Fundamental de Noticias

Función:
    Extraer las noticias más recientes de un activo financiero y utilizar un
    modelo de IA para analizar el sentimiento del mercado basado en los titulares.
    Implementa un fallback a un ETF proxy (ARGT) para activos sin noticias.

Skills:
    - Extracción de noticias de mercado (yfinance)
    - Interacción con APIs de IA generativa (google-generativeai)
    - Procesamiento de texto para prompts de IA

Uso:
    python agente_fundamental.py
"""

import os
import sys
import time
import yfinance as yf
import google.genai as genai

class AgenteFundamental:
    """
    Una clase para realizar análisis fundamental basado en noticias de activos
    financieros y generar una clasificación de sentimiento (POSITIVO, NEGATIVO, NEUTRAL)
    usando un modelo de IA.
    """

    def __init__(self, api_key: str):
        """
        Inicializa el agente, configurando el acceso a la API de IA de Gemini.

        Args:
            api_key (str): La clave de API para el servicio de IA generativa de Google.

        Raises:
            ValueError: Si la clave de API no es proporcionada.
            RuntimeError: Si hay un error al configurar el modelo de IA.
        """
        if not api_key:
            raise ValueError("La API Key de Gemini no fue proporcionada.")

        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-flash-latest"
            # Prompt para análisis fundamental de un activo específico
            self.system_prompt_template = (
                "Eres un analista fundamental financiero experto. Lee los siguientes "
                "titulares y resúmenes.\n\n"
                "ATENCIÓN FILTRO DE RUIDO: El activo a analizar es estrictamente {simbolo}. "
                "Si el titular o el resumen NO mencionan explícitamente a {simbolo} o al "
                "nombre de su empresa matriz, DEBES clasificar la fuente obligatoriamente "
                "como IRRELEVANTE. Está totalmente PROHIBIDO basar tu CONCLUSIÓN final en "
                "fuentes irrelevantes.\n\n"
                "Tu respuesta debe tener este formato exacto:\n"
                "    Enumera cada noticia de forma muy breve e indica si su índole es "
                "POSITIVA, NEGATIVA, NEUTRAL o IRRELEVANTE. (Ej: - Fuente 1: [Breve título/resumen] "
                "-> Índole: Positiva).\n"
                "    Al final, deja un renglón en blanco y escribe la palabra CONCLUSIÓN: "
                "seguida de tu veredicto final (POSITIVO, NEGATIVO o NEUTRAL) y "
                "justifica en máximo 2 líneas por qué."
            )
            # Prompt para análisis macroeconómico usando el proxy ARGT
            self.proxy_system_prompt = (
                "Eres un analista macroeconómico experto. El usuario quiere invertir en "
                "renta fija local/bonos, por lo que te proveo noticias recientes del ETF "
                "de Argentina (ARGT) como proxy del contexto país. Basado en estas noticias, "
                "define si el sentimiento macroeconómico es POSITIVO, NEGATIVO o NEUTRAL. "
                "Enumera 3 fuentes brevemente y termina con un renglón en blanco seguido "
                "de la palabra CONCLUSIÓN: tu veredicto y 2 líneas de justificación."
            )
        except Exception as e:
            raise RuntimeError(f"Error al configurar el cliente de IA de Gemini: {e}") from e

    def _generar_contenido_con_reintento(self, **kwargs) -> genai.types.GenerateContentResponse:
        """
        Envuelve la llamada a la API de Gemini con una lógica de reintento para
        manejar errores de límite de cuota (429).
        """
        for i in range(2):  # 0: primer intento, 1: reintento
            try:
                response = self.client.models.generate_content(**kwargs)
                return response
            except Exception as e:
                # Si es un error de cuota y es el primer intento
                if '429' in str(e) and i < 1:
                    print("Límite de cuota de API alcanzado. Esperando 60 segundos antes de reintentar...")
                    time.sleep(60)
                    continue  # Pasa a la siguiente iteración para reintentar
                else:
                    # Si es otro error o si el reintento también falla, se lanza la excepción
                    raise e
        # Este punto no debería ser alcanzado si la lógica es correcta, pero por si acaso.
        raise RuntimeError(
            "La llamada a la API falló después de un reintento por límite de cuota."
        )

    def analizar_activo(self, simbolo: str) -> str:
        """
        Descarga noticias, las formatea y obtiene un veredicto de la IA.
        Si no encuentra noticias, usa el ETF 'ARGT' como proxy macroeconómico.

        Args:
            simbolo (str): El símbolo del activo a analizar (ej. 'AAPL', 'MELI').

        Returns:
            str: La clasificación de sentimiento de la IA o un mensaje de error.
        """
        try:
            # 1. Intenta extraer noticias del símbolo original
            print(f"Buscando noticias recientes para {simbolo} en yfinance...")
            ticker = yf.Ticker(simbolo)
            noticias = ticker.news
            prompt_a_usar = self.system_prompt_template.format(simbolo=simbolo)

            # 2. Si no hay noticias, usa el fallback al proxy 'ARGT'
            if not noticias:
                print(f"No se encontraron noticias para '{simbolo}'. Usando 'ARGT' como proxy macroeconómico...")
                ticker_proxy = yf.Ticker('ARGT')
                noticias = ticker_proxy.news
                prompt_a_usar = self.proxy_system_prompt

            # Si ni el original ni el proxy tienen noticias, devuelve un mensaje.
            if not noticias:
                return "NEUTRAL - No se encontraron noticias para el activo ni para el proxy 'ARGT'."

            # 3. Procesamiento normal de las noticias encontradas (originales o del proxy)
            print("Noticias encontradas. Procesando...")
            ultimas_noticias = noticias[:5]
            titulares_formateados = []
            for noticia in ultimas_noticias:
                # Verificar si el item es un diccionario
                if not isinstance(noticia, dict):
                    continue

                # Acceder de forma segura al contenido (anidado o directo)
                contenido = noticia.get('content', noticia)
                if not isinstance(contenido, dict):
                    continue

                # Extraer título y resumen
                titulo = contenido.get('title', '')
                resumen = contenido.get('summary', '')

                # Validar: ignorar si están vacíos o si el título parece un UUID (sin espacios)
                if not titulo or not resumen or ' ' not in titulo:
                    continue

                # Concatenar título y resumen para formar el texto de la noticia
                texto_noticia = f"{titulo}. {resumen}"
                titulares_formateados.append(f"- {texto_noticia}")

            prompt_noticias = "\n".join(titulares_formateados)

            if not prompt_noticias.strip():
                return "NEUTRAL - No se pudo extraer contenido de las noticias para analizar."

            # 4. Enviar a la IA para análisis con el prompt correspondiente
            print("Enviando datos a la IA para análisis de sentimiento...")
            response = self._generar_contenido_con_reintento(
                model=self.model_name,
                contents=[prompt_a_usar, prompt_noticias]
            )

            return response.text.strip()

        except Exception as e:
            error_msg = f"Ocurrió un error durante el análisis de {simbolo}: {e}"
            print(error_msg, file=sys.stderr)
            return f"ERROR - {error_msg}"


if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Error: La variable de entorno GEMINI_API_KEY no está configurada.", file=sys.stderr)
        sys.exit(1)

    agente = AgenteFundamental(api_key=GEMINI_API_KEY)
    # Ejemplo con un bono que probablemente no tenga noticias en yfinance para probar el fallback
    simbolo_ejemplo = "NVDA"
    veredicto = agente.analizar_activo(simbolo_ejemplo)
    print("\n--- Veredicto del Analista Fundamental ---")
    print(f"Activo: {simbolo_ejemplo}")
    print(veredicto)
    print("------------------------------------------")

    # Ejemplo con una acción que sí tiene noticias para probar la ruta principal
    simbolo_ejemplo_2 = "GLOB"
    veredicto_2 = agente.analizar_activo(simbolo_ejemplo_2)
    print("\n--- Veredicto del Analista Fundamental ---")
    print(f"Activo: {simbolo_ejemplo_2}")
    print(veredicto_2)
    print("------------------------------------------")
