"""
iol_client.py — Cliente HTTP para la API REST de IOL (InvertirOnline)

Responsabilidades:
  - Autenticación con bearer token (válido 15 min) y refresh automático
  - Obtención de precio actual de un activo por símbolo y mercado
  - Manejo de errores HTTP con logging detallado

Uso desde otros módulos:
    from iol_client import IolClient
    cliente = IolClient()          # lee IOL_USER e IOL_PASSWORD del .env
    precio = cliente.get_precio("NVDA", "bCBA")   # → float o None
"""

import os
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes de la API de IOL
# Fuente: https://api.invertironline.com/Help/Autenticacion
# ──────────────────────────────────────────────────────────────────────────────
BASE_URL   = "https://api.invertironline.com"
TOKEN_URL  = f"{BASE_URL}/token"
API_V2     = f"{BASE_URL}/api/v2"

# El bearer token dura 15 minutos según la documentación oficial.
# Usamos 14 minutos como margen de seguridad para el refresh proactivo.
TOKEN_TTL_SEGUNDOS = 14 * 60

# Mercados disponibles en IOL
MERCADO_BCBA  = "bCBA"   # Bolsa de Comercio de Buenos Aires (acciones locales y CEDEARs)
MERCADO_NYSE  = "nYSE"   # New York Stock Exchange
MERCADO_NASDAQ = "nASDAQ"


class IolAuthError(Exception):
    """Error de autenticación con la API de IOL."""


class IolClient:
    """
    Cliente para la API REST de IOL con manejo automático de tokens.

    El token se renueva automáticamente cuando está por vencer,
    sin necesidad de reiniciar el bot ni volver a hacer login manual.
    """

    def __init__(self, username: str = None, password: str = None):
        self._username     = username or os.getenv("IOL_USER", "")
        self._password     = password or os.getenv("IOL_PASSWORD", "")
        self._access_token : str   = ""
        self._refresh_token: str   = ""
        self._token_ts     : float = 0.0   # timestamp del último login/refresh

        if not self._username or not self._password:
            raise IolAuthError(
                "Credenciales de IOL no configuradas. "
                "Definí IOL_USER e IOL_PASSWORD en tu .env"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Autenticación
    # ──────────────────────────────────────────────────────────────────────────

    def _login(self) -> None:
        """
        Obtiene el primer par de tokens usando usuario y contraseña.
        Llamado automáticamente en la primera request.
        """
        logger.info("IOL: realizando login...")
        resp = requests.post(
            TOKEN_URL,
            data={
                "username":   self._username,
                "password":   self._password,
                "grant_type": "password",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )

        if resp.status_code != 200:
            raise IolAuthError(
                f"Login fallido — HTTP {resp.status_code}: {resp.text[:200]}"
            )

        datos = resp.json()
        self._access_token  = datos["access_token"]
        self._refresh_token = datos["refresh_token"]
        self._token_ts      = time.time()
        logger.info("IOL: login exitoso. Token válido por 15 minutos.")

    def _refresh(self) -> None:
        """
        Renueva el bearer token usando el refresh token.
        Si falla, hace login completo como fallback.
        """
        logger.info("IOL: refrescando token...")
        try:
            resp = requests.post(
                TOKEN_URL,
                data={
                    "refresh_token": self._refresh_token,
                    "grant_type":    "refresh_token",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )

            if resp.status_code != 200:
                logger.warning(
                    f"Refresh fallido (HTTP {resp.status_code}). "
                    "Intentando login completo..."
                )
                self._login()
                return

            datos = resp.json()
            self._access_token  = datos["access_token"]
            self._refresh_token = datos["refresh_token"]
            self._token_ts      = time.time()
            logger.info("IOL: token refrescado exitosamente.")

        except Exception as e:
            logger.warning(f"Error en refresh: {e}. Intentando login completo...")
            self._login()

    def _asegurar_token(self) -> None:
        """
        Garantiza que hay un token válido antes de cada request.
        Hace login si no hay token, o refresh si está por vencer.
        """
        if not self._access_token:
            self._login()
            return

        segundos_transcurridos = time.time() - self._token_ts
        if segundos_transcurridos >= TOKEN_TTL_SEGUNDOS:
            self._refresh()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._access_token}"}

    # ──────────────────────────────────────────────────────────────────────────
    # Endpoints de mercado
    # ──────────────────────────────────────────────────────────────────────────

    def get_precio(self, simbolo: str, mercado: str = MERCADO_BCBA) -> float | None:
        """
        Devuelve el último precio de un activo en ARS.

        Args:
            simbolo : ticker del activo (ej: "NVDA", "GGAL", "AL30")
            mercado : mercado donde cotiza (default: bCBA)

        Returns:
            float con el último precio, o None si no se puede obtener.
        """
        self._asegurar_token()

        url = f"{API_V2}/titulos/{simbolo}/cotizacion"
        params = {"mercado": mercado}

        try:
            resp = requests.get(
                url,
                params=params,
                headers=self._headers(),
                timeout=10,
            )

            if resp.status_code == 401:
                # Token rechazado — forzar re-login y reintentar una vez
                logger.warning("IOL: token rechazado (401). Re-autenticando...")
                self._login()
                resp = requests.get(
                    url,
                    params=params,
                    headers=self._headers(),
                    timeout=10,
                )

            if resp.status_code == 404:
                logger.warning(
                    f"IOL: símbolo '{simbolo}' no encontrado en mercado '{mercado}'."
                )
                return None

            if resp.status_code != 200:
                logger.error(
                    f"IOL: error HTTP {resp.status_code} para {simbolo}/{mercado}. "
                    f"Body: {resp.text[:200]}"
                )
                return None

            datos = resp.json()

            # La API devuelve el precio en el campo "ultimoPrecio"
            precio = datos.get("ultimoPrecio")
            if precio is None or precio == 0:
                logger.warning(
                    f"IOL: 'ultimoPrecio' ausente o cero para {simbolo}. "
                    f"Respuesta completa: {datos}"
                )
                return None

            logger.info(f"IOL: precio de {simbolo} ({mercado}) = {precio}")
            return float(precio)

        except requests.exceptions.Timeout:
            logger.error(f"IOL: timeout al consultar {simbolo}.")
            return None
        except Exception as e:
            logger.error(f"IOL: error inesperado al consultar {simbolo}: {e}")
            return None

    def get_precio_con_fallback(self, simbolo: str) -> float | None:
        """
        Intenta obtener el precio primero en bCBA (CEDEARs y acciones locales).
        Si falla, intenta en NYSE y NASDAQ como fallback para acciones USA puras.

        Devuelve el primer precio válido encontrado, o None.
        """
        mercados = [MERCADO_BCBA, MERCADO_NYSE, MERCADO_NASDAQ]
        for mercado in mercados:
            precio = self.get_precio(simbolo, mercado)
            if precio:
                return precio
        logger.warning(
            f"IOL: no se pudo obtener precio de {simbolo} "
            f"en ningún mercado ({', '.join(mercados)})."
        )
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Singleton — una sola instancia por proceso del bot
# ──────────────────────────────────────────────────────────────────────────────
_cliente_iol: IolClient | None = None


def get_cliente() -> IolClient | None:
    """
    Devuelve la instancia singleton del cliente IOL.
    Si las credenciales no están configuradas, devuelve None
    sin romper el flujo del bot.
    """
    global _cliente_iol
    if _cliente_iol is None:
        try:
            _cliente_iol = IolClient()
        except IolAuthError as e:
            logger.warning(f"Cliente IOL no disponible: {e}")
            return None
    return _cliente_iol


def obtener_precio(simbolo: str) -> float:
    """
    Función de conveniencia para obtener el precio de un activo.
    Devuelve 0.0 si el cliente no está disponible o el precio no se obtiene.
    Nunca lanza excepciones — el flujo del bot nunca se interrumpe por precio.
    """
    cliente = get_cliente()
    if not cliente:
        return 0.0
    precio = cliente.get_precio_con_fallback(simbolo)
    return precio if precio else 0.0