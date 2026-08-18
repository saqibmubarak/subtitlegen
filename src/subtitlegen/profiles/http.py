from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class HttpClient(Protocol):
    def get_json(self, url: str) -> Any:
        """GET a URL and decode JSON."""

    def get_text(self, url: str) -> str:
        """GET a URL and decode text."""


class UrllibHttpClient:
    """Short-timeout HTTP client used by automatic profile sources."""

    def __init__(self, timeout: float = 4.0, user_agent: str = "subtitlegen/0.1") -> None:
        if timeout <= 0:
            raise ValueError("HTTP timeout must be positive")
        if not user_agent.strip():
            raise ValueError("user agent must not be blank")
        self._timeout = timeout
        self._user_agent = user_agent

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_text(url))

    def get_text(self, url: str) -> str:
        request = Request(
            url,
            headers={"User-Agent": self._user_agent, "Accept": "*/*"},
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
                charset = (
                    response.headers.get_content_charset() if response.headers else None
                )
        except HTTPError as error:
            raise RuntimeError(f"HTTP {error.code} for {url}") from error
        except (OSError, URLError, TimeoutError) as error:
            raise RuntimeError(f"HTTP request failed for {url}") from error
        text: str = raw.decode(charset or "utf-8", errors="replace")
        return text
