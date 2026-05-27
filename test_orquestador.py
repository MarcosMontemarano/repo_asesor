import requests
import json

# Pegá aquí EXACTAMENTE el mismo System Prompt que usa tu bot
SYSTEM_PROMPT = """Eres el enrutador de un sistema financiero conectado a InvertirOnline (IOL). NO memorices bases de datos ni inventes a qué se dedican las empresas. Tu único trabajo es clasificar la intención del usuario, extraer datos y devolver un JSON estricto.
REGLAS INQUEBRANTABLES:
1. Si el usuario menciona un ticker exacto y real del mercado (ej. SPY, AL30, GGAL) y tienes capital y riesgo, devuelve: {"accion": "EXEC", "ticker": "[ticker]", "capital": "[capital]", "riesgo": "[riesgo]"}.
2. Si el usuario pide invertir en una clase de activo (ej. bonos, cedears, acciones, fci) pero NO da un ticker exacto, y tienes capital y riesgo, devuelve INMEDIATAMENTE: {"accion": "BUSCAR", "categoria": "[clase de activo mencionada]", "capital": "[capital]", "riesgo": "[riesgo]"}.
3. Si falta el capital o el riesgo, devuelve: {"accion": "CHAT", "mensaje": "[tu pregunta breve y natural para obtener los datos faltantes]"}.
"""

# Batería de pruebas simulando a los usuarios de IOL
casos_de_prueba = [
    {"texto": "Hola, quiero empezar a invertir", "esperado": "CHAT"},
    {"texto": "Tengo 50000 pesos y soy conservador", "esperado": "CHAT"},
    {"texto": "quiero invertir 50000 en cedears a riesgo medio", "esperado": "BUSCAR"},
    {"texto": "analizame SPY con 100000 de capital y riesgo bajo", "esperado": "EXEC"},
    {"texto": "quiero comprar AL30, tengo 20000 y asumo riesgo alto", "esperado": "EXEC"},
    {"texto": "¿Qué me recomendas en bonos? capital 100000, riesgo bajo", "esperado": "BUSCAR"}
]

def correr_tests():
    print("Iniciando pruebas de estrés del Enrutador Local...\n")
    aprobados = 0

    for i, caso in enumerate(casos_de_prueba, 1):
        payload = {
            "model": "llama3.2",
            "format": "json",
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": caso["texto"]}
            ]
        }
        
        try:
            res = requests.post("http://127.0.0.1:11434/api/chat", json=payload)
            datos_ia = json.loads(res.json()["message"]["content"])
            
            if datos_ia.get("accion") == caso["esperado"]:
                print(f"✅ PASS Test {i} | Input: '{caso['texto']}' -> Acción: {datos_ia.get('accion')}")
                aprobados += 1
            else:
                print(f"❌ FAIL Test {i} | Input: '{caso['texto']}'")
                print(f"   -> Esperaba: {caso['esperado']}, pero devolvió: {datos_ia}")
        except Exception as e:
            print(f"⚠️ ERROR de parseo en Test {i}: {e}")

    print(f"\nResultado final: {aprobados}/{len(casos_de_prueba)} pruebas aprobadas.")

if __name__ == '__main__':
    correr_tests()