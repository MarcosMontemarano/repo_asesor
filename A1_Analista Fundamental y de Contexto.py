"""Agente 1: Analista Fundamental y de Contexto

Función:
    Leer noticias financieras, portales web y reportes económicos tanto locales (Argentina)
    como internacionales para extraer el sentimiento del mercado.

Skills:
    - Procesamiento de Lenguaje Natural (NLP)
    - Análisis de sentimiento enfocado en eventos macroeconómicos

Uso:
    python "A1_Analista Fundamental y de Contexto.py"
"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # type: ignore


@dataclass
class SentimentResult:
    source: str
    text: str
    sentiment: str
    score: float
    summary: Optional[str] = None
    region: Optional[str] = None
    event_type: Optional[str] = None


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def fetch_article_text(url: str, timeout: int = 12) -> str:
    if requests is None or BeautifulSoup is None:
        raise RuntimeError(
            "Faltan dependencias: instale requests y beautifulsoup4 para extraer noticias web."
        )

    response = requests.get(url, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    })
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Intentar extraer el texto más relevante de un artículo
    paragraphs = soup.find_all("p")
    if paragraphs:
        text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        return normalize_text(text)

    # Fallback: tomar el texto completo de la página
    return normalize_text(soup.get_text(separator=" ", strip=True))


def build_sentiment_pipeline():
    if pipeline is None:
        return None

    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception:
        try:
            return pipeline("sentiment-analysis")
        except Exception:
            return None


def simple_polarity(text: str) -> tuple[str, float]:
    positive_words = [
        "bueno", "positivo", "favorable", "alza", "crecimiento", "recuperación",
        "mejora", "optimista", "reducción", "disminución", "estabilidad"
    ]
    negative_words = [
        "malo", "negativo", "desfavorable", "caída", "contracción", "recesión",
        "incertidumbre", "pesimista", "aumento", "devaluación", "inflación"
    ]
    text_lower = text.lower()
    score = 0.0
    for word in positive_words:
        if word in text_lower:
            score += 1.0
    for word in negative_words:
        if word in text_lower:
            score -= 1.0

    if score > 0:
        return "POSITIVE", min(score / 5.0, 1.0)
    if score < 0:
        return "NEGATIVE", min(abs(score) / 5.0, 1.0)
    return "NEUTRAL", 0.5


def analyze_sentiment(text: str, model_pipeline=None) -> SentimentResult:
    text = normalize_text(text)
    if model_pipeline is not None:
        try:
            result = model_pipeline(text[:512])
            label = result[0]["label"].upper()
            score = float(result[0].get("score", 0.0))
            if label.startswith("5") or label.startswith("4") or "POSITIVE" in label:
                sentiment = "POSITIVE"
            elif label.startswith("1") or label.startswith("2") or "NEGATIVE" in label:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
            return SentimentResult(
                source="texto_directo",
                text=text,
                sentiment=sentiment,
                score=score,
            )
        except Exception:
            pass

    label, score = simple_polarity(text)
    return SentimentResult(
        source="texto_directo",
        text=text,
        sentiment=label,
        score=score,
    )


def detect_region_and_event(text: str) -> tuple[Optional[str], Optional[str]]:
    text_lower = text.lower()
    region = None
    event_type = None

    if "argentina" in text_lower or "banco central" in text_lower or "peso" in text_lower:
        region = "Argentina"
    elif "eurozona" in text_lower or "ue" in text_lower or "bce" in text_lower:
        region = "Eurozona"
    elif "estados unidos" in text_lower or "fed" in text_lower or "dólar" in text_lower:
        region = "EEUU"

    macro_topics = {
        "inflación": "Inflación",
        "tasa de interés": "Tasas de interés",
        "PIB": "PIB",
        "recesión": "Recesión",
        "empleo": "Empleo",
        "comercio": "Comercio internacional",
        "déficit": "Déficit fiscal",
        "deuda": "Deuda",
        "crecimiento": "Crecimiento económico",
    }
    for keyword, label in macro_topics.items():
        if keyword in text_lower:
            event_type = label
            break

    return region, event_type


def analyze_urls(urls: List[str]) -> List[SentimentResult]:
    pipeline_model = build_sentiment_pipeline()
    results: List[SentimentResult] = []

    for url in urls:
        try:
            text = fetch_article_text(url)
            sentiment = analyze_sentiment(text, model_pipeline=pipeline_model)
            region, event_type = detect_region_and_event(text)
            sentiment.source = url
            sentiment.region = region
            sentiment.event_type = event_type
            sentiment.summary = text[:250] + "..." if len(text) > 250 else text
            results.append(sentiment)
        except Exception as exc:
            results.append(SentimentResult(
                source=url,
                text="",
                sentiment="ERROR",
                score=0.0,
                summary=str(exc)
            ))

    return results


def print_summary(results: List[SentimentResult]) -> None:
    for item in results:
        print("---")
        print(f"Fuente: {item.source}")
        print(f"Sentimiento: {item.sentiment}")
        print(f"Puntaje: {item.score:.3f}")
        if item.region:
            print(f"Región: {item.region}")
        if item.event_type:
            print(f"Evento macro: {item.event_type}")
        if item.summary:
            print(f"Resumen: {item.summary}")


def main() -> int:
    sample_urls = [
        "https://www.ambito.com/economia/",
        "https://www.lanacion.com.ar/economia/",
    ]

    print("Analizando sentimiento macroeconómico de noticias financieras...")
    results = analyze_urls(sample_urls)
    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
