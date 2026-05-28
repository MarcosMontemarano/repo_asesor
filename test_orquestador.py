"""
Test de estrés del Enrutador — Nueva arquitectura
Llama extrae entidades, Python decide.

Etapa 1: tests stateless del extractor de Llama
Etapa 2: tests de la lógica de decisión Python (sin Ollama)
Etapa 3: tests de memoria conversacional end-to-end

Ejecutar:
    python test_orquestador.py
"""

import sys
import json
import re
import requests
from pathlib import Path

# Importar módulos del proyecto
sys.path.insert(0, str(Path(__file__).parent))
import ticker_resolver as tr

OLLAMA_URL   = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

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
- "texto_activo": devolvé el nombre TAL COMO lo escribió el usuario.
- "capital": SOLO dígitos. "100k" → "100000". "$50.000" → "50000". Vacío si no hay monto.
- "riesgo": "quiero algo seguro" → "bajo". "no me importa arriesgar" → "alto". Sin señal → vacío.
- "cantidad_activos": cuántos activos DISTINTOS mencionó. "SPY y AAPL" → 2.
- "intencion": analizar / distribuir / recomendar / consulta / otro

Solo JSON puro. Sin texto fuera del objeto JSON.
"""


def _extraer_numero(valor) -> str:
    if not valor:
        return ""
    return re.sub(r"[^\d]", "", str(valor))


def llamar_extractor(texto: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": EXTRACTOR_PROMPT + f"\n\nMensaje: {texto}\n\nDevolvé el JSON ahora:",
        "stream": False,
        "format": "json"
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    raw  = resp.json()["response"].strip()
    if not raw or raw == "{}":
        return {}
    datos = json.loads(raw)
    datos["capital"] = _extraer_numero(datos.get("capital", ""))
    return datos


# ──────────────────────────────────────────────────────────────────────────────
# Lógica de decisión Python (copia del orquestador para testing aislado)
# ──────────────────────────────────────────────────────────────────────────────
class ContextoTest:
    def __init__(self):
        self.capital = ""
        self.riesgo  = ""
        self.ticker_pendiente = ""

    def actualizar(self, entidades):
        c = _extraer_numero(entidades.get("capital", ""))
        r = str(entidades.get("riesgo", "")).strip().lower()
        if c: self.capital = c
        if r in ("bajo", "medio", "alto"): self.riesgo = r

    def resetear(self):
        self.capital = self.riesgo = self.ticker_pendiente = ""


def decidir(entidades: dict, contexto: ContextoTest) -> tuple[str, dict]:
    texto_activo = str(entidades.get("texto_activo", "")).strip()
    intencion    = str(entidades.get("intencion", "otro")).lower()
    cantidad     = int(entidades.get("cantidad_activos", 1) or 1)

    contexto.actualizar(entidades)

    if intencion in ("distribuir", "recomendar", "consulta"):
        return "CHAT", {"motivo": intencion}
    if cantidad > 1:
        return "CHAT", {"motivo": "multiples_activos"}

    ticker = tr.resolver_ticker(texto_activo) if texto_activo else None

    if texto_activo and not ticker:
        return "CHAT", {"motivo": "ticker_no_encontrado"}

    if (ticker and contexto.ticker_pendiente
            and ticker != contexto.ticker_pendiente
            and contexto.capital and contexto.riesgo):
        return "CHAT", {"motivo": "confirmar_contexto"}

    if not ticker:
        return "CHAT", {"motivo": "falta_ticker"}
    if not contexto.capital:
        return "CHAT", {"motivo": "falta_capital"}
    if not contexto.riesgo:
        return "CHAT", {"motivo": "falta_riesgo"}

    contexto.ticker_pendiente = ticker
    return "EXEC", {"ticker": ticker, "capital": contexto.capital, "riesgo": contexto.riesgo}


# ──────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — Extracción de entidades (Llama)
# ──────────────────────────────────────────────────────────────────────────────
CASOS_EXTRACCION = [
    # (descripcion, texto, campo_a_verificar, valor_esperado)
    ("Capital con puntos",          "tengo $100.000 para invertir",          "capital",          "100000"),
    ("Capital coloquial 100k",      "tengo 100k para TSLA",                  "capital",          "100000"),
    ("Capital en USD",              "quiero invertir USD 5000 en KO",        "capital",          "5000"),
    ("Sin capital",                 "quiero comprar AAPL",                   "capital",          ""),
    ("Riesgo inferido bajo",        "quiero algo seguro y conservador",      "riesgo",           "bajo"),
    ("Riesgo inferido alto",        "no me importa arriesgar, quiero TSLA",  "riesgo",           "alto"),
    ("Riesgo explícito medio",      "riesgo medio, 50000 en SPY",            "riesgo",           "medio"),
    ("Sin riesgo",                  "quiero comprar AAPL con 30000",         "riesgo",           ""),
    ("Texto activo YPF",            "analizá YPF con 80000",                 "texto_activo",     "YPF"),
    ("Texto activo Pampa",          "quiero invertir en Pampa Energía",      "texto_activo",     "Pampa Energía"),
    ("Texto activo Google",         "analizá Google con 120000 riesgo alto", "texto_activo",     "Google"),
    ("Sin activo",                  "tengo 50000 y no sé qué comprar",       "texto_activo",     ""),
    ("Intención analizar",          "analizá MELI con 60000",                "intencion",        "analizar"),
    ("Intención distribuir",        "cómo repartirías 100000 entre SPY y AAPL", "intencion",     "distribuir"),
    ("Intención recomendar",        "qué CEDEARs me recomendás?",            "intencion",        "recomendar"),
    ("Intención consulta gráfico",  "mostrá un gráfico de SPY",              "intencion",        "consulta"),
    ("Múltiples activos",           "SPY y AAPL con 100000",                 "cantidad_activos", 2),
    ("Un solo activo",              "analizá NVDA con 50000 riesgo alto",    "cantidad_activos", 1),
]

# ──────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — Lógica de decisión Python (sin Ollama, entidades hardcodeadas)
# ──────────────────────────────────────────────────────────────────────────────
CASOS_DECISION = [
    # (descripcion, entidades, accion_esperada, ticker_esperado)
    ("Todo completo YPF",
     {"texto_activo": "YPF", "capital": "80000", "riesgo": "medio", "cantidad_activos": 1, "intencion": "analizar"},
     "EXEC", "YPFD"),
    ("Todo completo Pampa",
     {"texto_activo": "Pampa Energía", "capital": "60000", "riesgo": "alto", "cantidad_activos": 1, "intencion": "analizar"},
     "EXEC", "PAMP"),
    ("Todo completo Google",
     {"texto_activo": "Google", "capital": "120000", "riesgo": "alto", "cantidad_activos": 1, "intencion": "analizar"},
     "EXEC", "GOOGL"),
    ("Todo completo bono AL30",
     {"texto_activo": "AL30", "capital": "20000", "riesgo": "bajo", "cantidad_activos": 1, "intencion": "analizar"},
     "EXEC", "AL30"),
    ("Falta capital",
     {"texto_activo": "AAPL", "capital": "", "riesgo": "medio", "cantidad_activos": 1, "intencion": "analizar"},
     "CHAT", None),
    ("Falta riesgo",
     {"texto_activo": "MELI", "capital": "50000", "riesgo": "", "cantidad_activos": 1, "intencion": "analizar"},
     "CHAT", None),
    ("Falta ticker",
     {"texto_activo": "", "capital": "80000", "riesgo": "medio", "cantidad_activos": 1, "intencion": "analizar"},
     "CHAT", None),
    ("Ticker fuera de catálogo",
     {"texto_activo": "PETR4", "capital": "50000", "riesgo": "medio", "cantidad_activos": 1, "intencion": "analizar"},
     "CHAT", None),
    ("Múltiples activos",
     {"texto_activo": "SPY", "capital": "100000", "riesgo": "medio", "cantidad_activos": 2, "intencion": "analizar"},
     "CHAT", None),
    ("Intención distribuir",
     {"texto_activo": "SPY", "capital": "100000", "riesgo": "medio", "cantidad_activos": 1, "intencion": "distribuir"},
     "CHAT", None),
    ("Intención recomendar",
     {"texto_activo": "", "capital": "80000", "riesgo": "bajo", "cantidad_activos": 0, "intencion": "recomendar"},
     "CHAT", None),
    ("Intención consulta",
     {"texto_activo": "SPY", "capital": "", "riesgo": "", "cantidad_activos": 1, "intencion": "consulta"},
     "CHAT", None),
    ("ETF Nasdaq → QQQ",
     {"texto_activo": "ETF del Nasdaq", "capital": "30000", "riesgo": "medio", "cantidad_activos": 1, "intencion": "analizar"},
     "EXEC", "QQQ"),
    ("Riesgo contradictorio — igual va",
     {"texto_activo": "KO", "capital": "200000", "riesgo": "bajo", "cantidad_activos": 1, "intencion": "analizar"},
     "EXEC", "KO"),
]

# ──────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — Memoria conversacional end-to-end
# ──────────────────────────────────────────────────────────────────────────────
SECUENCIAS = [
    {
        "desc": "Datos en partes — ticker, capital y riesgo en turnos separados",
        "pasos": [
            {"texto": "quiero invertir en AAPL",   "accion": "CHAT"},
            {"texto": "tengo 50000",                "accion": "CHAT"},
            {"texto": "riesgo medio",               "accion": "EXEC", "ticker": "AAPL"},
        ]
    },
    {
        "desc": "Persistencia — capital y riesgo se reutilizan en segundo análisis",
        "pasos": [
            {"texto": "analizá SPY con 100000 riesgo medio",  "accion": "EXEC", "ticker": "SPY"},
            {"texto": "ahora analizá AAPL",                   "accion": "CHAT"},  # pide confirmación
        ]
    },
    {
        "desc": "Reset explícito — nuevo análisis desde cero",
        "pasos": [
            {"texto": "analizá NVDA con 70000 riesgo alto",   "accion": "EXEC", "ticker": "NVDA"},
            {"texto": "quiero hacer otro análisis",           "accion": "RESET"},
            {"texto": "analizá GGAL",                         "accion": "CHAT"},  # sin capital ni riesgo
        ]
    },
    {
        "desc": "Distribución seguida de confirmación",
        "pasos": [
            {"texto": "cómo repartirías 100000 entre SPY, KO y GGAL?", "accion": "CHAT"},
            {"texto": "empezá por SPY",                                  "accion": "EXEC", "ticker": "SPY"},
        ]
    },
    {
        "desc": "Ticker fuera de catálogo — no ejecuta",
        "pasos": [
            {"texto": "analizá PETR4 con 50000 riesgo medio", "accion": "CHAT"},
        ]
    },
    {
        "desc": "Bono soberano completo",
        "pasos": [
            {"texto": "quiero comprar AL30 con 20000, riesgo bajo", "accion": "EXEC", "ticker": "AL30"},
        ]
    },
]

FRASES_RESET = [
    "otro análisis", "nueva inversión", "nuevo análisis",
    "empecemos de nuevo", "quiero hacer otro", "análisis distinto",
]

def _es_reset(texto: str) -> bool:
    t = texto.lower()
    return any(f in t for f in FRASES_RESET)


# ──────────────────────────────────────────────────────────────────────────────
# Runners
# ──────────────────────────────────────────────────────────────────────────────

def run_etapa1():
    print("=" * 65)
    print(" ETAPA 1 - Extraccion de entidades (Llama)")
    print("=" * 65)
    ok = fail = err = 0

    for desc, texto, campo, esperado in CASOS_EXTRACCION:
        try:
            datos = llamar_extractor(texto)
            obtenido = datos.get(campo, "")
            # Para números comparar como string
            paso = str(obtenido) == str(esperado)
            if paso:
                print(f"[OK]   {desc}")
                print(f"       {campo}='{obtenido}'")
                ok += 1
            else:
                print(f"[FAIL] {desc}")
                print(f"       Input   : \"{texto}\"")
                print(f"       {campo}: esperaba='{esperado}', obtuvo='{obtenido}'")
                print(f"       Raw: {datos}")
                fail += 1
        except Exception as e:
            print(f"[ERR]  {desc} -> {e}")
            err += 1
        print()

    print(f"  Etapa 1: {ok}/{len(CASOS_EXTRACCION)} [OK]  {fail} [FAIL]  {err} [ERR]")
    print()
    return ok, fail, err


def run_etapa2():
    print("=" * 65)
    print(" ETAPA 2 - Logica de decision Python (sin Ollama)")
    print("=" * 65)
    ok = fail = 0

    tr.inicializar()

    for desc, entidades, accion_esp, ticker_esp in CASOS_DECISION:
        ctx = ContextoTest()
        accion, datos = decidir(entidades, ctx)
        ticker_real = datos.get("ticker", "")

        accion_ok = accion == accion_esp
        ticker_ok = (ticker_real == ticker_esp) if ticker_esp else True

        if accion_ok and ticker_ok:
            detalle = f"accion={accion}" + (f", ticker={ticker_real}" if ticker_esp else "")
            print(f"[OK]   {desc}")
            print(f"       {detalle}")
            ok += 1
        else:
            print(f"[FAIL] {desc}")
            if not accion_ok:
                print(f"       Accion : esperaba={accion_esp}, obtuvo={accion}")
            if not ticker_ok:
                print(f"       Ticker : esperaba={ticker_esp}, obtuvo={ticker_real}")
            print(f"       Datos  : {datos}")
            fail += 1
        print()

    print(f"  Etapa 2: {ok}/{len(CASOS_DECISION)} [OK]  {fail} [FAIL]")
    print()
    return ok, fail


def run_etapa3():
    print("=" * 65)
    print(" ETAPA 3 - Memoria conversacional end-to-end")
    print("=" * 65)
    seq_ok = seq_fail = 0

    tr.inicializar()

    for s, sec in enumerate(SECUENCIAS, 1):
        print(f"-- Secuencia {s:02d}: {sec['desc']}")
        ctx  = ContextoTest()
        ok   = True

        for p, paso in enumerate(sec["pasos"], 1):
            texto      = paso["texto"]
            accion_esp = paso["accion"]
            ticker_esp = paso.get("ticker")

            # Simular reset
            if _es_reset(texto):
                ctx.resetear()
                if accion_esp == "RESET":
                    print(f"   [OK]   Paso {p}: \"{texto}\" -> RESET")
                    continue

            try:
                entidades = llamar_extractor(texto)
                accion, datos = decidir(entidades, ctx)
                ticker_real = datos.get("ticker", "")

                accion_ok = accion == accion_esp
                ticker_ok = (ticker_real == ticker_esp) if ticker_esp else True

                if accion_ok and ticker_ok:
                    detalle = f"accion={accion}" + (f", ticker={ticker_real}" if ticker_esp else "")
                    print(f"   [OK]   Paso {p}: \"{texto}\" -> {detalle}")
                    # Simular que el asesor respondió (para persistencia)
                    if accion == "EXEC":
                        ctx.ticker_pendiente = ticker_real
                else:
                    print(f"   [FAIL] Paso {p}: \"{texto}\"")
                    if not accion_ok:
                        print(f"          Accion : esperaba={accion_esp}, obtuvo={accion}")
                    if not ticker_ok:
                        print(f"          Ticker : esperaba={ticker_esp}, obtuvo={ticker_real}")
                    print(f"          Raw    : {entidades} -> {datos}")
                    ok = False

            except Exception as e:
                print(f"   [ERR]  Paso {p}: {e}")
                ok = False

        estado = "[OK] COMPLETA" if ok else "[FAIL] FALLIDA"
        print(f"   -> Secuencia {estado}")
        if ok:
            seq_ok += 1
        else:
            seq_fail += 1
        print()

    print(f"  Etapa 3: {seq_ok}/{len(SECUENCIAS)} secuencias [OK]  {seq_fail} [FAIL]")
    print()
    return seq_ok, seq_fail


if __name__ == "__main__":
    print()
    e1_ok, e1_fail, e1_err = run_etapa1()
    e2_ok, e2_fail         = run_etapa2()
    e3_ok, e3_fail         = run_etapa3()

    print("=" * 65)
    print(" RESUMEN FINAL")
    print("=" * 65)
    print(f"  Etapa 1 - Extraccion (Llama): {e1_ok}/{len(CASOS_EXTRACCION)} [OK]  {e1_fail} [FAIL]  {e1_err} [ERR]")
    print(f"  Etapa 2 - Decision (Python):  {e2_ok}/{len(CASOS_DECISION)} [OK]  {e2_fail} [FAIL]")
    print(f"  Etapa 3 - Memoria e2e:        {e3_ok}/{len(SECUENCIAS)} [OK]  {e3_fail} [FAIL]")
    print("=" * 65)