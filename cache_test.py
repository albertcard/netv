"""Tests for cache.py."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def cache_module(tmp_path: Path):
    """Import cache module with temp directories."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import cache

    # Patch paths to temp locations
    original_server_settings = cache.SERVER_SETTINGS_FILE
    original_users_dir = cache.USERS_DIR
    original_cache_dir = cache.CACHE_DIR
    cache.SERVER_SETTINGS_FILE = tmp_path / "server_settings.json"
    cache.USERS_DIR = tmp_path / "users"
    cache.USERS_DIR.mkdir(exist_ok=True)
    cache.CACHE_DIR = tmp_path / "cache"
    cache.CACHE_DIR.mkdir(exist_ok=True)

    # Clear memory cache
    cache._cache.clear()

    yield cache

    cache.SERVER_SETTINGS_FILE = original_server_settings
    cache.USERS_DIR = original_users_dir
    cache.CACHE_DIR = original_cache_dir
    cache._cache.clear()


class TestFileCache:
    def test_save_and_load_file_cache(self, cache_module):
        cache_module.save_file_cache("test", {"key": "value"})
        result = cache_module.load_file_cache("test")
        assert result is not None
        data, ts = result
        assert data == {"key": "value"}
        assert ts > 0

    def test_load_nonexistent_cache(self, cache_module):
        assert cache_module.load_file_cache("nonexistent") is None

    def test_load_corrupted_cache(self, cache_module):
        path = cache_module.CACHE_DIR / "corrupted.json"
        path.write_text("not valid json")
        assert cache_module.load_file_cache("corrupted") is None


class TestMemoryCache:
    def test_get_cache_returns_reference(self, cache_module):
        cache = cache_module.get_cache()
        cache["test"] = 123
        assert cache_module.get_cache()["test"] == 123

    def test_clear_all_caches_preserves_epg(self, cache_module):
        cache = cache_module.get_cache()
        cache["epg"] = {"data": "epg"}
        cache["live"] = {"data": "live"}
        cache_module.clear_all_caches()
        assert "epg" in cache
        assert "live" not in cache


class TestCachedInfo:
    def test_get_cached_info_calls_fetch(self, cache_module):
        fetch_fn = mock.Mock(return_value={"result": 42})
        result = cache_module.get_cached_info("test_key", fetch_fn)
        assert result == {"result": 42}
        fetch_fn.assert_called_once()

    def test_get_cached_info_uses_memory_cache(self, cache_module):
        fetch_fn = mock.Mock(return_value={"result": 1})
        cache_module.get_cached_info("key1", fetch_fn)
        cache_module.get_cached_info("key1", fetch_fn)
        # Only called once - second call uses memory cache
        fetch_fn.assert_called_once()

    def test_get_cached_info_force_bypasses_memory(self, cache_module):
        fetch_fn = mock.Mock(return_value={"result": 1})
        cache_module.get_cached_info("key2", fetch_fn)
        cache_module.get_cached_info("key2", fetch_fn, force=True)
        assert fetch_fn.call_count == 2


class TestSettings:
    def test_load_settings_defaults(self, cache_module):
        settings = cache_module.load_server_settings()
        assert settings["sources"] == []
        assert settings["transcode_mode"] == "auto"
        assert settings["transcode_hw"] == "nvidia"
        assert settings["probe_movies"] is True

    def test_save_and_load_settings(self, cache_module):
        settings = {"sources": [{"id": "s1", "name": "Test"}], "custom": True}
        cache_module.save_server_settings(settings)
        loaded = cache_module.load_server_settings()
        assert loaded["sources"] == [{"id": "s1", "name": "Test"}]
        assert loaded["custom"] is True


class TestUserSettings:
    def test_load_user_settings_defaults(self, cache_module):
        settings = cache_module.load_user_settings("testuser")
        assert settings["guide_filter"] == []
        assert settings["captions_enabled"] is True
        assert settings["watch_history"] == {}

    def test_save_and_load_user_settings(self, cache_module):
        settings = {"guide_filter": ["cat1", "cat2"], "captions_enabled": False}
        cache_module.save_user_settings("testuser", settings)
        loaded = cache_module.load_user_settings("testuser")
        assert loaded["guide_filter"] == ["cat1", "cat2"]
        assert loaded["captions_enabled"] is False

    def test_watch_position_save_and_get(self, cache_module):
        cache_module.save_watch_position("user1", "http://video.url", 120.5, 3600.0)
        entry = cache_module.get_watch_position("user1", "http://video.url")
        assert entry is not None
        assert entry["position"] == 120.5
        assert entry["duration"] == 3600.0

    def test_watch_position_resets_at_95_percent(self, cache_module):
        # Save at 96% watched
        cache_module.save_watch_position("user1", "http://video.url", 960.0, 1000.0)
        entry = cache_module.get_watch_position("user1", "http://video.url")
        assert entry is None  # Should be reset


class TestSource:
    def test_source_dataclass(self, cache_module):
        source = cache_module.Source(
            id="test",
            name="Test Source",
            type="xtream",
            url="http://example.com",
        )
        assert source.id == "test"
        assert source.username == ""
        assert source.epg_timeout == 120
        assert source.epg_enabled is True

    def test_get_sources_empty(self, cache_module):
        sources = cache_module.get_sources()
        assert sources == []

    def test_get_sources_from_settings(self, cache_module):
        settings = {
            "sources": [
                {
                    "id": "s1",
                    "name": "Source 1",
                    "type": "m3u",
                    "url": "http://example.com/playlist.m3u",
                }
            ]
        }
        cache_module.save_settings(settings)
        sources = cache_module.get_sources()
        assert len(sources) == 1
        assert sources[0].id == "s1"
        assert sources[0].type == "m3u"


class TestUpdateSourceEpgUrl:
    def test_update_source_epg_url(self, cache_module):
        settings = {"sources": [{"id": "s1", "name": "S1", "type": "m3u", "url": "http://x"}]}
        cache_module.save_settings(settings)
        cache_module.update_source_epg_url("s1", "http://epg.example.com")
        loaded = cache_module.load_settings()
        assert loaded["sources"][0]["epg_url"] == "http://epg.example.com"

    def test_update_source_epg_url_not_overwrite(self, cache_module):
        settings = {
            "sources": [
                {
                    "id": "s1",
                    "name": "S1",
                    "type": "m3u",
                    "url": "http://x",
                    "epg_url": "http://existing",
                }
            ]
        }
        cache_module.save_settings(settings)
        cache_module.update_source_epg_url("s1", "http://new")
        loaded = cache_module.load_settings()
        assert loaded["sources"][0]["epg_url"] == "http://existing"

    def test_update_source_epg_url_empty_noop(self, cache_module):
        settings = {"sources": [{"id": "s1", "name": "S1", "type": "m3u", "url": "http://x"}]}
        cache_module.save_settings(settings)
        cache_module.update_source_epg_url("s1", "")
        loaded = cache_module.load_settings()
        assert "epg_url" not in loaded["sources"][0]
