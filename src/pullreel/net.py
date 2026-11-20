"""HTTP clients that trust the operating system's certificate store.

Company proxies and antivirus "HTTPS scanning" re-sign TLS traffic with their own root certificate, which Windows
and macOS trust but Python's bundled list (certifi) does not: every API call then fails with
CERTIFICATE_VERIFY_FAILED. `truststore` makes Python verify against the OS store, like a browser does.
"""

from __future__ import annotations

import functools
import ssl
from typing import Any

import httpx


@functools.cache
def ssl_context() -> ssl.SSLContext | bool:
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # truststore missing or unsupported platform: fall back to httpx's default verification
        return True


def async_client(**kwargs: Any) -> httpx.AsyncClient:
    kwargs.setdefault("verify", ssl_context())
    return httpx.AsyncClient(**kwargs)


def get(url: str, **kwargs: Any) -> httpx.Response:
    kwargs.setdefault("verify", ssl_context())
    return httpx.get(url, **kwargs)


def describe(exc: BaseException) -> str:
    """Human cause of a network failure (the SSL reason is often buried two levels down)."""
    text = str(exc) or type(exc).__name__
    if "CERTIFICATE_VERIFY_FAILED" in text:
        return ("TLS certificate rejected - a proxy or antivirus is intercepting HTTPS and its certificate is not "
                "trusted by this computer")  # fmt: skip
    return text.splitlines()[0][:200]


def post(url: str, **kwargs: Any) -> httpx.Response:
    kwargs.setdefault("verify", ssl_context())
    return httpx.post(url, **kwargs)
