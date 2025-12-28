"""Xtream Codes API client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from util import safe_urlopen


@dataclass(slots=True)
class XtreamClient:
    """Client for Xtream Codes API.

    Handles authentication and API calls to Xtream-compatible IPTV providers.
    """

    base_url: str
    username: str
    password: str

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/player_api.php?username={self.username}&password={self.password}"

    def _fetch(self, url: str, timeout: int = 30) -> str:
        with safe_urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")

    def _api(self, action: str | None = None, **params: Any) -> Any:
        url = self.api_url
        if action:
            url += f"&action={action}"
        for k, v in params.items():
            url += f"&{k}={v}"
        return json.loads(self._fetch(url))

    def get_live_categories(self) -> list[dict[str, Any]]:
        return self._api("get_live_categories")

    def get_live_streams(self, category_id: int | None = None) -> list[dict[str, Any]]:
        if category_id:
            return self._api("get_live_streams", category_id=category_id)
        return self._api("get_live_streams")

    def get_vod_categories(self) -> list[dict[str, Any]]:
        return self._api("get_vod_categories")

    def get_vod_streams(self, category_id: int | None = None) -> list[dict[str, Any]]:
        if category_id:
            return self._api("get_vod_streams", category_id=category_id)
        return self._api("get_vod_streams")

    def get_series_categories(self) -> list[dict[str, Any]]:
        return self._api("get_series_categories")

    def get_series(self, category_id: int | None = None) -> list[dict[str, Any]]:
        if category_id:
            return self._api("get_series", category_id=category_id)
        return self._api("get_series")

    def get_series_info(self, series_id: int) -> dict[str, Any]:
        return self._api("get_series_info", series_id=series_id)

    def get_vod_info(self, vod_id: int) -> dict[str, Any]:
        return self._api("get_vod_info", vod_id=vod_id)

    def build_stream_url(self, stream_type: str, stream_id: int, ext: str = "") -> str:
        base = f"{self.base_url}/{stream_type}/{self.username}/{self.password}/{stream_id}"
        return f"{base}.{ext}" if ext else base

    @property
    def epg_url(self) -> str:
        return f"{self.base_url}/xmltv.php?username={self.username}&password={self.password}"
