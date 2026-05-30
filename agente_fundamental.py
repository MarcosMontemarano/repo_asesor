"""
Agente Fundamental — Analista de Noticias y Sentimiento de Mercado

Motor: Groq (Llama 3.3 70B) — reemplaza Gemini

Cadena de fuentes (Score de Cobertura Informacional — SCI):
  Nivel 1 (paralelo): yfinance + Finnhub
  Nivel 2 (si SCI < 3): NewsAPI + Alpha Vantage
  Nivel 3 (si SCI < 3): Reddit → Twitter/X

SCI = (noticias_relevantes × 1.5) + (noticias_recientes_48h × 1.0) + (fuentes_distintas × 0.5)
Umbral mínimo para considerar información suficiente: SCI >= 3

Uso:
    python agente_fundamental.py
"""

import os
import sys
import time
import logging
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import yfinance as yf

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
SCI_UMBRAL          = 3.0
PESO_RELEVANCIA     = 1.5
PESO_RECENCIA       = 1.0
PESO_DIVERSIDAD     = 0.5
VENTANA_RECENCIA_H  = 48   # horas para considerar una noticia "reciente"
MAX_NOTICIAS_NIVEL  = 5    # máximo de noticias a procesar por fuente

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"


# ──────────────────────────────────────────────────────────────────────────────
# Estructura de noticia normalizada
# Todas las fuentes producen este formato para que el SCI sea uniforme.
# ──────────────────────────────────────────────────────────────────────────────
def _noticia(titulo: str, resumen: str, publisher: str, timestamp_unix: float | None) -> dict:
    return {
        "titulo":    titulo.strip(),
        "resumen":   resumen.strip(),
        "publisher": publisher.strip().lower(),
        "ts":        timestamp_unix,   # None si no disponible
    }


# ──────────────────────────────────────────────────────────────────────────────
# Score de Cobertura Informacional (SCI)
# Fundamento: EMH forma semi-fuerte — solo cuenta información reciente,
# relevante y proveniente de fuentes diversas.
# ──────────────────────────────────────────────────────────────────────────────
def _calcular_sci(noticias: list[dict], simbolo: str) -> tuple[float, list[dict]]:
    """
    Calcula el SCI de una lista de noticias normalizadas.
    Devuelve (sci_score, noticias_relevantes).
    Una noticia es relevante si menciona el símbolo o empresa en título o resumen.
    """
    ahora = datetime.now(timezone.utc)
    ventana = timedelta(hours=VENTANA_RECENCIA_H)
    simbolo_lower = simbolo.lower()

    relevantes   = []
    recientes    = 0
    publishers   = set()

    for n in noticias:
        texto = (n["titulo"] + " " + n["resumen"]).lower()
        es_relevante = simbolo_lower in texto

        if es_relevante:
            relevantes.append(n)
            publishers.add(n["publisher"])

            if n["ts"]:
                try:
                    dt = datetime.fromtimestamp(n["ts"], tz=timezone.utc)
                    if ahora - dt <= ventana:
                        recientes += 1
                except Exception:
                    pass

    sci = (
        len(relevantes)  * PESO_RELEVANCIA +
        recientes        * PESO_RECENCIA   +
        len(publishers)  * PESO_DIVERSIDAD
    )

    logger.info(
        f"SCI para {simbolo}: {sci:.2f} "
        f"(relevantes={len(relevantes)}, recientes={recientes}, "
        f"publishers={len(publishers)})"
    )
    return sci, relevantes


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 1: yfinance
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_yfinance(simbolo: str) -> list[dict]:
    try:
        ticker  = yf.Ticker(simbolo)
        noticias_raw = ticker.news or []
        resultado = []
        for n in noticias_raw[:MAX_NOTICIAS_NIVEL]:
            if not isinstance(n, dict):
                continue
            contenido = n.get("content", n)
            if not isinstance(contenido, dict):
                continue
            titulo  = contenido.get("title", "")
            resumen = contenido.get("summary", "")
            if not titulo or " " not in titulo:
                continue
            publisher = contenido.get("provider", {})
            if isinstance(publisher, dict):
                publisher = publisher.get("displayName", "yfinance")
            ts = None
            pub_date = contenido.get("pubDate") or contenido.get("publishedAt")
            if pub_date:
                try:
                    ts = datetime.fromisoformat(
                        pub_date.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass
            resultado.append(_noticia(titulo, resumen, publisher or "yfinance", ts))
        logger.info(f"yfinance: {len(resultado)} noticias para {simbolo}")
        return resultado
    except Exception as e:
        logger.warning(f"yfinance error para {simbolo}: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 2: Finnhub
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_finnhub(simbolo: str) -> list[dict]:
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        logger.warning("FINNHUB_API_KEY no configurada. Saltando Finnhub.")
        return []
    try:
        hoy   = datetime.now().strftime("%Y-%m-%d")
        desde = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        url   = "https://finnhub.io/api/v1/company-news"
        resp  = requests.get(
            url,
            params={"symbol": simbolo, "from": desde, "to": hoy, "token": api_key},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Finnhub HTTP {resp.status_code} para {simbolo}")
            return []
        items = resp.json()[:MAX_NOTICIAS_NIVEL]
        resultado = [
            _noticia(
                i.get("headline", ""),
                i.get("summary", ""),
                i.get("source", "finnhub"),
                i.get("datetime"),
            )
            for i in items
            if i.get("headline")
        ]
        logger.info(f"Finnhub: {len(resultado)} noticias para {simbolo}")
        return resultado
    except Exception as e:
        logger.warning(f"Finnhub error para {simbolo}: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 3: NewsAPI
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_newsapi(simbolo: str) -> list[dict]:
    api_key = os.getenv("NEWSAPI_KEY", "")
    if not api_key:
        logger.warning("NEWSAPI_KEY no configurada. Saltando NewsAPI.")
        return []
    try:
        url  = "https://newsapi.org/v2/everything"
        resp = requests.get(
            url,
            params={
                "q":        simbolo,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": MAX_NOTICIAS_NIVEL,
                "apiKey":   api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"NewsAPI HTTP {resp.status_code}")
            return []
        articles = resp.json().get("articles", [])
        resultado = []
        for a in articles:
            titulo  = a.get("title", "")
            resumen = a.get("description", "") or a.get("content", "")
            pub     = a.get("source", {}).get("name", "newsapi")
            ts      = None
            published = a.get("publishedAt")
            if published:
                try:
                    ts = datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass
            if titulo and " " in titulo:
                resultado.append(_noticia(titulo, resumen, pub, ts))
        logger.info(f"NewsAPI: {len(resultado)} noticias para {simbolo}")
        return resultado
    except Exception as e:
        logger.warning(f"NewsAPI error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 4: Alpha Vantage (News Sentiment)
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_alphavantage(simbolo: str) -> list[dict]:
    api_key = os.getenv("ALPHAVANTAGE_KEY", "")
    if not api_key:
        logger.warning("ALPHAVANTAGE_KEY no configurada. Saltando Alpha Vantage.")
        return []
    try:
        url  = "https://www.alphavantage.co/query"
        resp = requests.get(
            url,
            params={
                "function": "NEWS_SENTIMENT",
                "tickers":  simbolo,
                "limit":    MAX_NOTICIAS_NIVEL,
                "apikey":   api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Alpha Vantage HTTP {resp.status_code}")
            return []
        feed = resp.json().get("feed", [])
        resultado = []
        for a in feed:
            titulo  = a.get("title", "")
            resumen = a.get("summary", "")
            pub     = a.get("source", "alphavantage")
            ts      = None
            time_str = a.get("time_published", "")
            if time_str:
                try:
                    # Formato: 20240115T143000
                    ts = datetime.strptime(
                        time_str, "%Y%m%dT%H%M%S"
                    ).replace(tzinfo=timezone.utc).timestamp()
                except Exception:
                    pass
            if titulo:
                resultado.append(_noticia(titulo, resumen, pub, ts))
        logger.info(f"Alpha Vantage: {len(resultado)} noticias para {simbolo}")
        return resultado
    except Exception as e:
        logger.warning(f"Alpha Vantage error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 5: Reddit
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_reddit(simbolo: str) -> list[dict]:
    client_id     = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        logger.warning("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET no configurados. Saltando Reddit.")
        return []
    try:
        # Autenticación
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        headers = {"User-Agent": "AFMA/1.0"}
        token_resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data={"grant_type": "client_credentials"},
            headers=headers,
            timeout=10,
        )
        if token_resp.status_code != 200:
            logger.warning(f"Reddit auth HTTP {token_resp.status_code}")
            return []
        token = token_resp.json().get("access_token", "")

        # Buscar en subreddits financieros relevantes
        subreddits = ["investing", "stocks", "merval", "argentina"]
        resultado  = []
        headers["Authorization"] = f"bearer {token}"

        for sub in subreddits:
            if len(resultado) >= MAX_NOTICIAS_NIVEL:
                break
            resp = requests.get(
                f"https://oauth.reddit.com/r/{sub}/search",
                params={"q": simbolo, "sort": "new", "limit": 3, "t": "week"},
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            posts = resp.json().get("data", {}).get("children", [])
            for p in posts:
                data    = p.get("data", {})
                titulo  = data.get("title", "")
                resumen = data.get("selftext", "")[:300]
                ts      = data.get("created_utc")
                if titulo:
                    resultado.append(_noticia(titulo, resumen, f"reddit/r/{sub}", ts))

        logger.info(f"Reddit: {len(resultado)} posts para {simbolo}")
        return resultado
    except Exception as e:
        logger.warning(f"Reddit error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Fuente 6: Twitter/X
# ──────────────────────────────────────────────────────────────────────────────
def _fetch_twitter(simbolo: str) -> list[dict]:
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not bearer_token:
        logger.warning("TWITTER_BEARER_TOKEN no configurado. Saltando Twitter.")
        return []
    try:
        headers = {"Authorization": f"Bearer {bearer_token}"}
        resp = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            params={
                "query":       f"${simbolo} lang:en -is:retweet",
                "max_results": MAX_NOTICIAS_NIVEL,
                "tweet.fields": "created_at,author_id",
            },
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"Twitter HTTP {resp.status_code}")
            return []
        tweets = resp.json().get("data", [])
        resultado = []
        for t in tweets:
            texto = t.get("text", "")
            ts    = None
            created = t.get("created_at")
            if created:
                try:
                    ts = datetime.fromisoformat(
                        created.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass
            if texto:
                resultado.append(_noticia(texto[:120], "", "twitter", ts))
        logger.info(f"Twitter: {len(resultado)} tweets para {simbolo}")
        return resultado
    except Exception as e:
        logger.warning(f"Twitter error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Motor Groq
# ──────────────────────────────────────────────────────────────────────────────
def _llamar_groq(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """
    Llama a Groq con reintentos para 429.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens":  800,
    }
    for intento in range(3):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                espera = 60
                logger.warning(f"Groq 429 — esperando {espera}s (intento {intento+1}/3)")
                time.sleep(espera)
                continue
            if resp.status_code != 200:
                logger.error(f"Groq HTTP {resp.status_code}: {resp.text[:200]}")
                return f"ERROR - Groq HTTP {resp.status_code}"
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Groq error intento {intento+1}: {e}")
            if intento < 2:
                time.sleep(5)
    return "ERROR - Groq no disponible después de 3 intentos."


# ──────────────────────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────────────────────
class AgenteFundamental:
    """
    Analista fundamental con cadena de fuentes y SCI como criterio de avance.
    Motor: Groq Llama 3.3 70B.
    """

    def __init__(self, api_key: str = None):
        """
        api_key: Groq API key. Si no se pasa, la lee de GROQ_API_KEY en .env.
        Se mantiene compatibilidad con el parámetro api_key del orquestador.
        """
        self.groq_api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY no configurada. "
                "Agregá GROQ_API_KEY=tu_key al .env"
            )

        self.system_prompt_template = (
            "Eres un analista fundamental financiero experto. "
            "Analizás noticias sobre {simbolo} y determinás el sentimiento del mercado.\n\n"
            "FILTRO DE RUIDO: Solo considerás noticias que mencionen explícitamente "
            "a {simbolo} o su empresa. Las demás son IRRELEVANTES.\n\n"
            "PROYECCIÓN ESTRATÉGICA: Identificá eventos globales o macroeconómicos "
            "próximos (elecciones, decisiones de tasas, eventos deportivos globales) "
            "y evaluá su impacto en {simbolo}.\n\n"
            "Formato de respuesta:\n"
            "- Fuente N: [título breve] -> Índole: POSITIVA/NEGATIVA/NEUTRAL/IRRELEVANTE\n"
            "\nCONCLUSIÓN: [POSITIVO/NEGATIVO/NEUTRAL] — [justificación en máximo 3 líneas]"
        )

        self.proxy_system_prompt = (
            "Eres un analista macroeconómico experto en Argentina. "
            "Analizás noticias del ETF ARGT como proxy del riesgo país.\n\n"
            "PROYECCIÓN ESTRATÉGICA: Identificá impacto de eventos globales "
            "(FMI, elecciones, commodities) en bonos soberanos argentinos.\n\n"
            "Formato: enumera 3 fuentes brevemente.\n"
            "CONCLUSIÓN: [POSITIVO/NEGATIVO/NEUTRAL] — [justificación en 3 líneas]"
        )

    def _formatear_noticias(self, noticias: list[dict]) -> str:
        lineas = []
        for i, n in enumerate(noticias, 1):
            lineas.append(f"- Fuente {i} ({n['publisher']}): {n['titulo']}. {n['resumen'][:200]}")
        return "\n".join(lineas)

    def analizar_activo(self, simbolo: str) -> str:
        """
        Ejecuta la cadena de fuentes con SCI como criterio de avance.
        Devuelve el análisis de sentimiento del agente.
        """
        logger.info(f"[AgenteFundamental] Iniciando análisis para {simbolo}")
        todas_las_noticias: list[dict] = []

        # ── NIVEL 1: yfinance + Finnhub en paralelo ────────────────────────
        logger.info(f"[Nivel 1] yfinance + Finnhub para {simbolo}")
        noticias_yf  = _fetch_yfinance(simbolo)
        noticias_fh  = _fetch_finnhub(simbolo)
        todas_las_noticias = noticias_yf + noticias_fh

        sci, relevantes = _calcular_sci(todas_las_noticias, simbolo)

        # ── NIVEL 2: NewsAPI + Alpha Vantage si SCI < umbral ──────────────
        if sci < SCI_UMBRAL:
            logger.info(f"[Nivel 2] SCI={sci:.2f} < {SCI_UMBRAL}. Consultando NewsAPI + Alpha Vantage...")
            noticias_na = _fetch_newsapi(simbolo)
            noticias_av = _fetch_alphavantage(simbolo)
            todas_las_noticias += noticias_na + noticias_av
            sci, relevantes = _calcular_sci(todas_las_noticias, simbolo)

        # ── NIVEL 3: Reddit + Twitter si SCI sigue bajo ───────────────────
        if sci < SCI_UMBRAL:
            logger.info(f"[Nivel 3] SCI={sci:.2f} < {SCI_UMBRAL}. Consultando Reddit + Twitter...")
            noticias_reddit  = _fetch_reddit(simbolo)
            noticias_twitter = _fetch_twitter(simbolo)
            todas_las_noticias += noticias_reddit + noticias_twitter
            sci, relevantes = _calcular_sci(todas_las_noticias, simbolo)

        # ── Fallback: proxy ARGT si no hay nada relevante ─────────────────
        if not relevantes:
            logger.info(f"Sin noticias relevantes para {simbolo}. Usando proxy ARGT...")
            noticias_argt = _fetch_yfinance("ARGT") + _fetch_finnhub("ARGT")
            if not noticias_argt:
                return "NEUTRAL - No se encontraron noticias para el activo ni para el proxy ARGT."
            prompt_noticias = self._formatear_noticias(noticias_argt[:5])
            return _llamar_groq(self.proxy_system_prompt, prompt_noticias, self.groq_api_key)

        # ── Análisis con Groq ──────────────────────────────────────────────
        noticias_a_analizar = relevantes[:MAX_NOTICIAS_NIVEL]
        prompt_sistema  = self.system_prompt_template.format(simbolo=simbolo)
        prompt_noticias = (
            f"SCI calculado: {sci:.2f} (umbral={SCI_UMBRAL})\n"
            f"Noticias relevantes encontradas: {len(relevantes)}\n\n"
            + self._formatear_noticias(noticias_a_analizar)
        )

        logger.info(
            f"[AgenteFundamental] Enviando {len(noticias_a_analizar)} noticias "
            f"a Groq. SCI={sci:.2f}"
        )
        return _llamar_groq(prompt_sistema, prompt_noticias, self.groq_api_key)


# ──────────────────────────────────────────────────────────────────────────────
# Ejecución directa para pruebas
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    agente = AgenteFundamental()

    for simbolo in ["NVDA", "GGAL", "AL30"]:
        print(f"\n{'='*50}")
        print(f" Análisis fundamental: {simbolo}")
        print("="*50)
        resultado = agente.analizar_activo(simbolo)
        print(resultado)