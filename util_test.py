"""Tests for util.py."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest


@pytest.fixture
def util_module():
    """Import util module."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import util

    return util


class _FakeRequest:
    """Minimal request object for testing redirect handler."""

    def __init__(self, url: str):
        self.full_url = url
        self.headers: dict[str, str] = {}
        self.data = None
        self.origin_req_host = "original.com"

    def get_method(self) -> str:
        return "GET"


class TestSafeRedirectHandler:
    def test_handler_allows_http(self, util_module):
        handler = util_module._SafeRedirectHandler()
        req = _FakeRequest("http://original.com")
        result = handler.redirect_request(
            req,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="http://redirect.com/path",
        )
        assert result is not None

    def test_handler_allows_https(self, util_module):
        handler = util_module._SafeRedirectHandler()
        req = _FakeRequest("https://original.com")
        result = handler.redirect_request(
            req,
            fp=None,
            code=302,
            msg="Found",
            headers={},
            newurl="https://secure.com/path",
        )
        assert result is not None

    def test_handler_rejects_file_scheme(self, util_module):
        handler = util_module._SafeRedirectHandler()
        req = _FakeRequest("http://original.com")
        with pytest.raises(urllib.error.URLError, match="Unsafe redirect scheme"):
            handler.redirect_request(
                req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl="file:///etc/passwd",
            )

    def test_handler_rejects_data_scheme(self, util_module):
        handler = util_module._SafeRedirectHandler()
        req = _FakeRequest("http://original.com")
        with pytest.raises(urllib.error.URLError, match="Unsafe redirect scheme"):
            handler.redirect_request(
                req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl="data:text/html,<script>alert(1)</script>",
            )

    def test_handler_rejects_javascript_scheme(self, util_module):
        handler = util_module._SafeRedirectHandler()
        req = _FakeRequest("http://original.com")
        with pytest.raises(urllib.error.URLError, match="Unsafe redirect scheme"):
            handler.redirect_request(
                req,
                fp=None,
                code=302,
                msg="Found",
                headers={},
                newurl="javascript:alert(1)",
            )
