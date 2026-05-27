"""
Test de estrés del Enrutador (Ollama / Llama 3.2)
Sincronizado con agente_orquestador.py actual.

Etapa 1: tests stateless (un mensaje, sin historial)
Etapa 2: tests de memoria conversacional (secuencias de turnos)

Ejecutar con Ollama corriendo:
    python test_orquestador.py
"""

import requests
import json

OLLAMA_URL   = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

# ──────────────────────────────────────────────────────────────────────────────
# System prompt — copia exacta del orquestador
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
ACCIONES LOCALES (Bolsa Argentina):
  GGAL  = Grupo Financiero Galicia (banco)
  YPFD  = YPF S.A. (petróleo y gas, empresa nacional)
  BMA   = Banco Macro
  PAMP  = Pampa Energía (energía eléctrica)
  TXAR  = Ternium Argentina (acero)
  LOMA  = Loma Negra (construcción/cemento)
  SUPV  = Supervielle (banco)

CEDEARs (acciones extranjeras que cotizan en pesos en Argentina):
  SPY   = ETF S&P 500 (diversificación total del mercado USA)
  QQQ   = ETF Nasdaq 100 (tecnología USA)
  AAPL  = Apple Inc.
  MSFT  = Microsoft Corp.
  GOOGL = Alphabet (Google)
  AMZN  = Amazon
  TSLA  = Tesla
  NVDA  = NVIDIA (semiconductores/IA)
  KO    = Coca-Cola (consumo defensivo, dividendos)
  JPM   = JPMorgan Chase (banco USA)
  MELI  = MercadoLibre (tecnología latinoamericana)
  GLOB  = Globant (tecnología argentina cotizando en NYSE)

BONOS SOBERANOS ARGENTINOS (renta fija en USD):
  AL30  = Bono vence 2030, legislación local
  AL35  = Bono vence 2035, legislación local
  GD30  = Bono vence 2030, legislación Nueva York (más seguro)
  GD35  = Bono vence 2035, legislación Nueva York
  ADVERTENCIA: AL30 y GD30 son BONOS SOBERANOS, NO acciones ni CEDEARs.

PERFIL DE RIESGO POR CLASE DE ACTIVO:
  Muy bajo   → Cauciones / FCI money market
  Bajo       → Bonos (AL30, GD30)
  Medio      → CEDEARs defensivos (KO, SPY)
  Medio-Alto → Acciones locales (GGAL, YPFD, PAMP)
  Alto       → CEDEARs growth (TSLA, NVDA)

TU ROL:
Eres el enrutador de AFMA, asesor financiero de IOL Argentina.
Leé el mensaje del usuario y extraé: CAPITAL, RIESGO y TICKER.
Devolvé UN SOLO objeto JSON. Sin texto fuera del JSON.
La respuesta SIEMPRE debe contener la clave "accion". Nunca devuelvas {}.
La clave de acción se escribe EXACTAMENTE así: "accion" (sin tilde).
La clave del ticker se escribe EXACTAMENTE así: "ticker" (sin e al final).

QUÉ ES UN TICKER:
Un ticker es el código corto de un activo en la bolsa.
Cuando uses un ticker en tu mensaje, aclarás el nombre entre paréntesis.
Ejemplos de mapeo nombre → ticker:
  "YPF" o "YPF S.A."        → YPFD
  "Pampa" o "Pampa Energía" → PAMP
  "Google" o "Alphabet"     → GOOGL
  "MercadoLibre"            → MELI
  "ETF del Nasdaq"          → QQQ
  "ETF del S&P" o "S&P500"  → SPY
  "Galicia"                 → GGAL

EXTRACCIÓN DE DATOS:
CAPITAL : un número. Solo dígitos. NUNCA nombres de empresas en capital.
          Si no hay monto, capital va vacío: "".
RIESGO  : bajo / medio / alto. Sin señal clara = medio.
TICKER  : usá el catálogo y la tabla de mapeo de arriba.

REGLAS CRÍTICAS:
REGLA 1 — UN SOLO ACTIVO PARA EXEC:
  Si el usuario menciona MÁS DE UN activo, la acción es SIEMPRE CHAT.

REGLA 2 — NOMBRE DE EMPRESA CON CAPITAL Y RIESGO ES EXEC:
  Si nombra UNA empresa y tenés capital y riesgo → EXEC con ticker mapeado.
  "analizá Pampa Energía con 60000 riesgo alto" → EXEC ticker=PAMP
  "el ETF del Nasdaq con 30000 riesgo medio"   → EXEC ticker=QQQ
  "analizá Google con 120000 riesgo alto"       → EXEC ticker=GOOGL

REGLA 3 — JSON NUNCA VACÍO:
  La respuesta siempre tiene "accion". Si dudás, usá CHAT.

ACCIONES VÁLIDAS — SOLO ESTAS DOS:
"CHAT" : cuando falta CAPITAL, RIESGO o TICKER, o hay múltiples activos,
         o el usuario pide distribución, planes, gráficos, noticias,
         preguntas de seguimiento, o cualquier cosa fuera del análisis puntual.
         Formato: {"accion": "CHAT", "mensaje": "[respuesta breve]"}

"EXEC" : solo cuando tenés UN ticker + CAPITAL (número) + RIESGO confirmados.
         Formato: {"accion": "EXEC", "ticker": "[TICKER]", "capital": "[número]", "riesgo": "[nivel]"}

PROHIBIDO: RECOMENDAR, ANALIZAR, BUSCAR, DISTRIBUIR, o cualquier otra acción.
PROHIBIDO: "acción" con tilde. Siempre "accion".
PROHIBIDO: "ticket". Siempre "ticker".
PROHIBIDO: nombres de empresas en el campo capital.
PROHIBIDO: EXEC con múltiples tickers.
PROHIBIDO: devolver JSON vacío {}.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def llamar_ollama(messages: list[dict]) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "format": "json",
        "stream": False,
        "messages": messages
    }
    res  = requests.post(OLLAMA_URL, json=payload, timeout=60)
    raw  = res.json()["message"]["content"].strip()
    if not raw or raw == "{}":
        return {"accion": "CHAT", "mensaje": "[fallback: JSON vacío]"}
    datos = json.loads(raw)
    # Normalizar typos conocidos
    if "accion" not in datos:
        for v in ("acción", "Accion", "Acción", "ACTION", "action"):
            if v in datos:
                datos["accion"] = datos.pop(v)
                break
    if "ticker" not in datos:
        for v in ("ticket", "Ticker", "TICKER"):
            if v in datos:
                datos["ticker"] = datos.pop(v)
                break
    return datos


def build_messages(historial: list[str]) -> list[dict]:
    """
    Convierte el historial ["Usuario: ...", "Asesor: ..."] al formato
    de mensajes de Ollama con roles user/assistant.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for linea in historial:
        if linea.startswith("Usuario: "):
            messages.append({"role": "user", "content": linea[len("Usuario: "):]})
        elif linea.startswith("Asesor: "):
            messages.append({"role": "assistant", "content": linea[len("Asesor: "):]})
    return messages


# ──────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — Tests stateless
# ──────────────────────────────────────────────────────────────────────────────
CASOS_STATELESS = [
    # Incompleto simple
    {"desc": "Solo intención, sin datos",                   "texto": "hola, quiero empezar a invertir",                             "esperado": "CHAT"},
    {"desc": "Capital sin ticker ni riesgo",                "texto": "tengo 50000 pesos para invertir",                             "esperado": "CHAT"},
    {"desc": "Ticker sin capital",                          "texto": "quiero comprar AAPL",                                         "esperado": "CHAT"},
    {"desc": "Riesgo sin capital ni ticker",                "texto": "soy conservador y quiero algo seguro",                        "esperado": "CHAT"},
    {"desc": "Capital y riesgo sin ticker",                 "texto": "tengo 80000 y riesgo medio, qué me recomendás?",              "esperado": "CHAT"},
    # Distribución y planes
    {"desc": "Distribución bonos y CEDEARs",                "texto": "quiero invertir 100000 entre bonos argentinos y empresas internacionales", "esperado": "CHAT"},
    {"desc": "Pide porcentaje entre clases",                "texto": "con 50000, cuánto pongo en bonos y cuánto en empresas nacionales?",        "esperado": "CHAT"},
    {"desc": "Pide plan de inversión",                      "texto": "me harías un plan de inversión con 100000 pesos disponibles?",             "esperado": "CHAT"},
    {"desc": "Distribución en tres activos",                "texto": "quiero invertir en SPY, AAPL y YPF, cómo repartirías 150000?",            "esperado": "CHAT"},
    {"desc": "Diversificación sin activos específicos",     "texto": "quiero diversificar 200000 entre acciones argentinas y CEDEARs, riesgo medio", "esperado": "CHAT"},
    # Nombre natural → ticker
    {"desc": "YPF → YPFD",                                  "texto": "quiero analizar YPF con 80000 y riesgo medio",                "esperado": "EXEC", "ticker": "YPFD"},
    {"desc": "Pampa Energía → PAMP",                        "texto": "analizá Pampa Energía con 60000 riesgo alto",                 "esperado": "EXEC", "ticker": "PAMP"},
    {"desc": "MercadoLibre sin riesgo → CHAT",              "texto": "quiero entrar en MercadoLibre con 40000",                     "esperado": "CHAT"},
    {"desc": "ETF del Nasdaq → QQQ",                        "texto": "analizame el ETF del Nasdaq con 30000, riesgo medio",         "esperado": "EXEC", "ticker": "QQQ"},
    {"desc": "banco Galicia → GGAL",                        "texto": "quiero invertir en el banco Galicia, 90000 riesgo medio-alto","esperado": "EXEC", "ticker": "GGAL"},
    {"desc": "Google → GOOGL",                              "texto": "analizá Google con 120000 riesgo alto",                       "esperado": "EXEC", "ticker": "GOOGL"},
    # EXEC directo
    {"desc": "Todo junto con ticker exacto",                "texto": "analizame SPY con 100000 de capital y riesgo bajo",           "esperado": "EXEC", "ticker": "SPY"},
    {"desc": "Bono soberano completo",                      "texto": "quiero comprar AL30, tengo 20000 y asumo riesgo bajo",        "esperado": "EXEC", "ticker": "AL30"},
    {"desc": "Varios tickers → CHAT",                       "texto": "analizame SPY y AAPL con 100000 riesgo medio",                "esperado": "CHAT"},
    {"desc": "Cambio de capital con ticker",                "texto": "cambiá el capital a 75000, seguimos con GGAL riesgo alto",    "esperado": "EXEC", "ticker": "GGAL"},
    {"desc": "Ticker en minúscula",                         "texto": "analizá nvda con 50000, riesgo alto",                         "esperado": "EXEC", "ticker": "NVDA"},
    # Fuera de scope
    {"desc": "Pregunta de seguimiento",                     "texto": "por qué me recomendás esa empresa?",                         "esperado": "CHAT"},
    {"desc": "Pide gráfico",                                "texto": "podés mostrarme un gráfico de SPY de los últimos 6 meses?",   "esperado": "CHAT"},
    {"desc": "Pide noticias",                               "texto": "tenés noticias recientes sobre AAPL?",                        "esperado": "CHAT"},
    {"desc": "Pregunta de timing",                          "texto": "creés que MELI va a subir en las próximas semanas?",          "esperado": "CHAT"},
    {"desc": "Pregunta educativa",                          "texto": "qué diferencia hay entre un CEDEAR y una acción local?",      "esperado": "CHAT"},
    # Edge cases
    {"desc": "Riesgo bajo + retorno alto (contradictorio)", "texto": "quiero algo de riesgo bajo pero con mucho retorno, 200000",   "esperado": "CHAT"},
    {"desc": "Capital coloquial 100k",                      "texto": "tengo como 100k para meter en TSLA, riesgo alto",             "esperado": "EXEC", "ticker": "TSLA"},
    {"desc": "Monto en dólares",                            "texto": "quiero invertir USD 5000 en KO, riesgo bajo",                 "esperado": "EXEC", "ticker": "KO"},
    {"desc": "Ticker fuera del catálogo",                   "texto": "analizá PETR4 con 50000 riesgo medio",                        "esperado": "CHAT"},
    {"desc": "Saludo sin contenido financiero",             "texto": "hola, buen día!",                                             "esperado": "CHAT"},
]

# ──────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — Tests de memoria conversacional (secuencias de turnos)
# Formato: lista de pasos. Cada paso tiene el mensaje del usuario,
# la acción esperada, y opcionalmente el ticker esperado.
# El historial se acumula turno a turno dentro de cada secuencia.
# ──────────────────────────────────────────────────────────────────────────────
SECUENCIAS = [
    {
        "desc": "Datos en partes — capital, riesgo y ticker en turnos separados",
        "pasos": [
            {"usuario": "quiero invertir en AAPL",    "esperado": "CHAT"},
            {"usuario": "tengo 50000",                 "esperado": "CHAT"},
            {"usuario": "riesgo medio",                "esperado": "EXEC", "ticker": "AAPL"},
        ]
    },
    {
        "desc": "Corrección de dato — cambio de riesgo post-EXEC",
        "pasos": [
            {"usuario": "analizá SPY con 100000 riesgo alto",  "esperado": "EXEC", "ticker": "SPY"},
            {"usuario": "cambiá el riesgo a bajo",             "esperado": "EXEC", "ticker": "SPY"},
        ]
    },
    {
        "desc": "No repetir recomendaciones — pide otras opciones",
        "pasos": [
            {"usuario": "qué CEDEARs me recomendás con 80000 riesgo medio?", "esperado": "CHAT"},
            {"usuario": "otras opciones?",                                    "esperado": "CHAT"},
        ]
    },
    {
        "desc": "Ticker del contexto — mismo capital y riesgo para otro activo",
        "pasos": [
            {"usuario": "analizá NVDA con 70000 riesgo alto",  "esperado": "EXEC", "ticker": "NVDA"},
            {"usuario": "ahora hacé lo mismo con TSLA",        "esperado": "EXEC", "ticker": "TSLA"},
        ]
    },
    {
        "desc": "Post-EXEC sin contexto — historial borrado, pide capital de nuevo",
        "pasos": [
            {"usuario": "analizá MELI con 60000 riesgo medio", "esperado": "EXEC", "ticker": "MELI"},
            {"usuario": "ahora GGAL",                          "esperado": "CHAT"},
        ]
    },
    {
        "desc": "Distribución seguida de confirmación de ticker",
        "pasos": [
            {"usuario": "quiero invertir 100000 en SPY, KO y GGAL, cómo repartirías?", "esperado": "CHAT"},
            {"usuario": "empezá por SPY",                                               "esperado": "EXEC", "ticker": "SPY"},
        ]
    },
    {
        "desc": "Corrección de ticker a mitad de conversación",
        "pasos": [
            {"usuario": "quiero analizar Apple",              "esperado": "CHAT"},
            {"usuario": "tengo 45000",                         "esperado": "CHAT"},
            {"usuario": "mejor cambiá a Microsoft, riesgo medio", "esperado": "EXEC", "ticker": "MSFT"},
        ]
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def correr_stateless():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         ETAPA 1 — Tests stateless (mensaje único)           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    aprobados = fallados = errores = 0

    for i, caso in enumerate(CASOS_STATELESS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": caso["texto"]}
        ]
        try:
            datos  = llamar_ollama(messages)
            accion = str(datos.get("accion", "")).upper()
            ticker_esp  = caso.get("ticker")
            ticker_real = str(datos.get("ticker", "")).upper()

            accion_ok = accion == caso["esperado"]
            ticker_ok = (ticker_real == ticker_esp) if ticker_esp else True

            if accion_ok and ticker_ok:
                detalle = f"accion={accion}" + (f", ticker={ticker_real}" if ticker_esp else "")
                print(f"✅ PASS [{i:02d}] {caso['desc']}")
                print(f"        → {detalle}")
                aprobados += 1
            else:
                print(f"❌ FAIL [{i:02d}] {caso['desc']}")
                print(f"        Input: \"{caso['texto']}\"")
                if not accion_ok:
                    print(f"        Accion : esperaba={caso['esperado']}, obtuvo={accion}")
                if not ticker_ok:
                    print(f"        Ticker : esperaba={ticker_esp}, obtuvo={ticker_real}")
                print(f"        Raw    : {datos}")
                fallados += 1
        except Exception as e:
            print(f"⚠️  ERROR [{i:02d}] {caso['desc']} → {e}")
            errores += 1
        print()

    total = len(CASOS_STATELESS)
    print(f"  Etapa 1: {aprobados}/{total} ✅  |  {fallados} ❌  |  {errores} ⚠️")
    print()
    return aprobados, fallados, errores


def correr_secuencias():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       ETAPA 2 — Tests de memoria conversacional             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    seq_ok = seq_fail = 0

    for s, secuencia in enumerate(SECUENCIAS, 1):
        print(f"── Secuencia {s:02d}: {secuencia['desc']}")
        historial: list[str] = []
        pasos_ok = True

        for p, paso in enumerate(secuencia["pasos"], 1):
            historial.append(f"Usuario: {paso['usuario']}")
            messages = build_messages(historial)

            try:
                datos  = llamar_ollama(messages)
                accion = str(datos.get("accion", "")).upper()
                ticker_esp  = paso.get("ticker")
                ticker_real = str(datos.get("ticker", "")).upper()

                accion_ok = accion == paso["esperado"]
                ticker_ok = (ticker_real == ticker_esp) if ticker_esp else True

                if accion_ok and ticker_ok:
                    detalle = f"accion={accion}" + (f", ticker={ticker_real}" if ticker_esp else "")
                    print(f"   ✅ Paso {p}: \"{paso['usuario']}\" → {detalle}")
                    # Simular respuesta del asesor para continuar el historial
                    historial.append(f"Asesor: [accion={accion}]")
                else:
                    print(f"   ❌ Paso {p}: \"{paso['usuario']}\"")
                    if not accion_ok:
                        print(f"      Accion : esperaba={paso['esperado']}, obtuvo={accion}")
                    if not ticker_ok:
                        print(f"      Ticker : esperaba={ticker_esp}, obtuvo={ticker_real}")
                    print(f"      Raw    : {datos}")
                    historial.append(f"Asesor: [accion={accion}]")
                    pasos_ok = False

            except Exception as e:
                print(f"   ⚠️  Paso {p}: error → {e}")
                pasos_ok = False

        if pasos_ok:
            print(f"   → Secuencia COMPLETA ✅")
            seq_ok += 1
        else:
            print(f"   → Secuencia FALLIDA ❌")
            seq_fail += 1
        print()

    total = len(SECUENCIAS)
    print(f"  Etapa 2: {seq_ok}/{total} secuencias ✅  |  {seq_fail} ❌")
    print()
    return seq_ok, seq_fail


if __name__ == "__main__":
    print()
    e1_ok, e1_fail, e1_err = correr_stateless()
    e2_ok, e2_fail         = correr_secuencias()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    RESUMEN FINAL                            ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Etapa 1 (stateless):    {e1_ok}/{len(CASOS_STATELESS)} ✅  {e1_fail} ❌  {e1_err} ⚠️")
    print(f"║  Etapa 2 (secuencias):   {e2_ok}/{len(SECUENCIAS)} ✅  {e2_fail} ❌")
    print("╚══════════════════════════════════════════════════════════════╝")