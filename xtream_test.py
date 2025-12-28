"""Tests for xtream.py - Xtream Codes API client."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


sys.path.insert(0, str(Path(__file__).parent))
from xtream import XtreamClient


class TestXtreamClient:
    """Tests for XtreamClient."""

    def test_api_url_property(self):
        client = XtreamClient("http://example.com", "user", "pass")
        assert client.api_url == "http://example.com/player_api.php?username=user&password=pass"

    def test_epg_url_property(self):
        client = XtreamClient("http://example.com", "user", "pass")
        assert client.epg_url == "http://example.com/xmltv.php?username=user&password=pass"

    def test_build_stream_url_live_no_ext(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("live", 123)
        assert url == "http://example.com/live/user/pass/123"

    def test_build_stream_url_live_with_ext(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("live", 123, "m3u8")
        assert url == "http://example.com/live/user/pass/123.m3u8"

    def test_build_stream_url_movie(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("movie", 456, "mkv")
        assert url == "http://example.com/movie/user/pass/456.mkv"

    def test_build_stream_url_series(self):
        client = XtreamClient("http://example.com", "user", "pass")
        url = client.build_stream_url("series", 789, "mp4")
        assert url == "http://example.com/series/user/pass/789.mp4"


class TestXtreamClientApi:
    """Tests for XtreamClient API methods with mocked network."""

    @pytest.fixture
    def client(self):
        return XtreamClient("http://example.com", "user", "pass")

    @pytest.fixture
    def mock_urlopen(self):
        with patch("xtream.safe_urlopen") as mock:
            yield mock

    def _setup_response(self, mock_urlopen, data):
        """Helper to setup mock response."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

    def test_get_live_categories(self, client, mock_urlopen):
        categories = [{"category_id": "1", "category_name": "News"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_live_categories()

        assert result == categories
        mock_urlopen.assert_called_once()
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_categories" in url

    def test_get_live_streams(self, client, mock_urlopen):
        streams = [{"stream_id": 1, "name": "CNN"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_live_streams()

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_streams" in url

    def test_get_live_streams_with_category(self, client, mock_urlopen):
        streams = [{"stream_id": 1, "name": "CNN"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_live_streams(category_id=5)

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_live_streams" in url
        assert "category_id=5" in url

    def test_get_vod_categories(self, client, mock_urlopen):
        categories = [{"category_id": "10", "category_name": "Movies"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_vod_categories()

        assert result == categories
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_categories" in url

    def test_get_vod_streams(self, client, mock_urlopen):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_vod_streams()

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_streams" in url

    def test_get_vod_streams_with_category(self, client, mock_urlopen):
        streams = [{"stream_id": 100, "name": "Movie 1"}]
        self._setup_response(mock_urlopen, streams)

        result = client.get_vod_streams(category_id=10)

        assert result == streams
        url = mock_urlopen.call_args[0][0]
        assert "category_id=10" in url

    def test_get_series_categories(self, client, mock_urlopen):
        categories = [{"category_id": "20", "category_name": "Drama"}]
        self._setup_response(mock_urlopen, categories)

        result = client.get_series_categories()

        assert result == categories
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series_categories" in url

    def test_get_series(self, client, mock_urlopen):
        series = [{"series_id": 200, "name": "Show 1"}]
        self._setup_response(mock_urlopen, series)

        result = client.get_series()

        assert result == series
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series" in url

    def test_get_series_with_category(self, client, mock_urlopen):
        series = [{"series_id": 200, "name": "Show 1"}]
        self._setup_response(mock_urlopen, series)

        result = client.get_series(category_id=20)

        assert result == series
        url = mock_urlopen.call_args[0][0]
        assert "category_id=20" in url

    def test_get_series_info(self, client, mock_urlopen):
        info = {"info": {"name": "Show 1"}, "episodes": {"1": []}}
        self._setup_response(mock_urlopen, info)

        result = client.get_series_info(series_id=200)

        assert result == info
        url = mock_urlopen.call_args[0][0]
        assert "action=get_series_info" in url
        assert "series_id=200" in url

    def test_get_vod_info(self, client, mock_urlopen):
        info = {"info": {"name": "Movie 1", "plot": "A story"}}
        self._setup_response(mock_urlopen, info)

        result = client.get_vod_info(vod_id=100)

        assert result == info
        url = mock_urlopen.call_args[0][0]
        assert "action=get_vod_info" in url
        assert "vod_id=100" in url
