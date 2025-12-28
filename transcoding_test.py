"""Tests for transcoding.py.

State transition scenarios:

| # | From -> To            | Expected Behavior                      | CC |
|---|-----------------------|----------------------------------------|----|
| 1 | START -> JUMP         | Kill, delete, start at X, extract CC   | Y  |
| 2 | START -> DEAD         | ffmpeg killed, segments remain         | -  |
| 3 | JUMP -> JUMP          | Kill, delete, start at new X, extract  | Y  |
| 4 | JUMP -> DEAD          | ffmpeg killed, segments remain         | -  |
| 5 | JUMP -> RESUME        | Return cached, existing CC             | Y  |
| 6 | DEAD(off=0) -> RESUME | Append, no new CC                      | !  |
| 7 | DEAD(off>0) -> RESUME | Return cached, existing CC             | Y  |
| 8 | DEAD -> JUMP          | Start at X, extract CC                 | Y  |
| 9 | RESUME -> JUMP        | Kill, delete, start at X, extract CC   | Y  |
| 10| RESUME -> DEAD        | ffmpeg killed                          | -  |

CC legend: Y=works, !=partial (existing content only), -=n/a (no playback)
"""

import json
import pathlib
import tempfile
import time
from unittest import mock

import pytest


pytest_plugins = ("pytest_asyncio",)


def make_playlist(output_dir: str, num_segments: int) -> float:
    """Create fake HLS playlist with segments. Returns duration."""
    playlist = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:6\n"
    for i in range(num_segments):
        playlist += f"#EXTINF:6.0,\nseg{i:03d}.ts\n"
        (pathlib.Path(output_dir) / f"seg{i:03d}.ts").write_bytes(b"x" * 2000)
    (pathlib.Path(output_dir) / "stream.m3u8").write_text(playlist)
    return num_segments * 6.0


def make_vtt(output_dir: str, index: int = 0):
    """Create fake VTT subtitle file."""
    (pathlib.Path(output_dir) / f"sub{index}.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nTest\n"
    )


def make_session_json(output_dir: str, session_id: str, url: str, seek_offset: float = 0):
    """Create session.json for recovery."""
    info = {
        "session_id": session_id,
        "url": url,
        "is_vod": True,
        "started": time.time(),
        "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
        "duration": 2876.0,
        "seek_offset": seek_offset,
    }
    (pathlib.Path(output_dir) / "session.json").write_text(json.dumps(info))


# Import after defining helpers to avoid import errors during collection
@pytest.fixture
def transcoding_module():
    """Import transcoding module with mocked settings."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import transcoding

    transcoding.init(
        lambda: {
            "transcode_hw": "software",
            "probe_movies": True,
            "vod_transcode_cache_mins": 60,
        }
    )
    yield transcoding
    # Cleanup sessions after test
    with transcoding._transcode_lock:
        transcoding._transcode_sessions.clear()
        transcoding._vod_url_to_session.clear()


class TestScenarioA:
    """A: Active session (ffmpeg running) -> return it, CC works."""

    @pytest.mark.asyncio
    async def test_active_session_returned(self, transcoding_module):
        with tempfile.TemporaryDirectory() as output_dir:
            make_playlist(output_dir, 5)
            make_vtt(output_dir)

            # Active process (returncode is None)
            mock_process = mock.Mock()
            mock_process.returncode = None

            existing = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 1260,
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_a"] = existing

            result = await transcoding_module._handle_existing_vod_session(
                "test_a", "http://example.com/video.mkv", "software", False
            )

            assert result is not None, "Should return session"
            assert result["session_id"] == "test_a"
            assert result["seek_offset"] == 1260
            assert len(result["subtitles"]) > 0, "Should have subtitles"


class TestScenarioB:
    """B: Dead session, seek_offset=0, segments exist -> append."""

    @pytest.mark.asyncio
    async def test_dead_session_no_seek_appends(self, transcoding_module):
        with tempfile.TemporaryDirectory() as output_dir:
            hls_duration = make_playlist(output_dir, 10)  # 60s
            make_vtt(output_dir)

            # Dead process
            mock_process = mock.Mock()
            mock_process.returncode = -9

            existing = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 0,  # No prior jump
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_b"] = existing

            with mock.patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = mock.AsyncMock()
                mock_proc.returncode = None
                mock_proc.pid = 12345
                mock_exec.return_value = mock_proc

                # Create next segment so resume loop doesn't wait
                def create_segment(*args, **kwargs):
                    (pathlib.Path(output_dir) / "seg010.ts").write_bytes(b"x" * 2000)
                    return mock_proc

                mock_exec.side_effect = create_segment

                with mock.patch.object(transcoding_module, "_wait_for_playlist", return_value=True):
                    result = await transcoding_module._handle_existing_vod_session(
                        "test_b", "http://example.com/video.mkv", "software", False
                    )

            assert result is not None, "Should return session after append"
            # Check ffmpeg was called with append_list
            cmd = mock_exec.call_args[0]
            cmd_str = " ".join(cmd)
            assert "append_list" in cmd_str, "Should use append_list"
            assert f"-ss {hls_duration}" in cmd_str or f"-ss {int(hls_duration)}" in cmd_str


class TestScenarioC:
    """C: Dead session, seek_offset>0, segments exist -> return cached, CC works."""

    @pytest.mark.asyncio
    async def test_dead_session_with_seek_returns_cached(self, transcoding_module):
        with tempfile.TemporaryDirectory() as output_dir:
            hls_duration = make_playlist(output_dir, 20)  # 120s
            make_vtt(output_dir)

            mock_process = mock.Mock()
            mock_process.returncode = -9  # Dead

            existing = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 1260,  # Prior jump
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_c"] = existing

            # Should NOT start new ffmpeg
            with mock.patch("asyncio.create_subprocess_exec") as mock_exec:
                result = await transcoding_module._handle_existing_vod_session(
                    "test_c", "http://example.com/video.mkv", "software", False
                )

            assert result is not None, "Should return cached session"
            assert result["session_id"] == "test_c"
            assert result["seek_offset"] == 1260
            assert result["transcoded_duration"] == hls_duration
            assert len(result["subtitles"]) > 0, "Should have subtitles from cache"
            mock_exec.assert_not_called()  # No new ffmpeg


class TestScenarioD:
    """D: Dead session, no segments -> return None (triggers fresh start)."""

    @pytest.mark.asyncio
    async def test_dead_session_no_segments_returns_none(self, transcoding_module):
        with tempfile.TemporaryDirectory() as output_dir:
            # No segments, just empty dir

            mock_process = mock.Mock()
            mock_process.returncode = -9

            existing = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 0,
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_d"] = existing
                transcoding_module._vod_url_to_session["http://example.com/video.mkv"] = "test_d"

            result = await transcoding_module._handle_existing_vod_session(
                "test_d", "http://example.com/video.mkv", "software", False
            )

            assert result is None, "Should return None to trigger fresh start"


class TestScenarioF:
    """F: Jump (seek_transcode) -> kill old, start fresh, CC works."""

    @pytest.mark.asyncio
    async def test_seek_extracts_subtitles(self, transcoding_module):
        with tempfile.TemporaryDirectory() as output_dir:
            make_playlist(output_dir, 10)
            make_vtt(output_dir)
            make_session_json(output_dir, "test_f", "http://example.com/video.mkv")

            mock_process = mock.Mock()
            mock_process.returncode = None
            mock_process.kill = mock.Mock()
            mock_process.wait = mock.AsyncMock()

            session = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 0,
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_f"] = session

            with mock.patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = mock.AsyncMock()
                mock_proc.returncode = None
                mock_proc.pid = 12345
                mock_exec.return_value = mock_proc

                with mock.patch.object(transcoding_module, "_wait_for_playlist", return_value=True):
                    result = await transcoding_module.seek_transcode("test_f", 1260.0)

            assert result["ok"] is True
            assert result["time"] == 1260.0

            # Check ffmpeg was called with subtitle extraction
            cmd = mock_exec.call_args[0]
            cmd_str = " ".join(cmd)
            assert "-ss 1260" in cmd_str, "Should seek to position"
            assert "sub0.vtt" in cmd_str, "Should extract subtitles"
            assert "-c:s webvtt" in cmd_str, "Should convert to webvtt"
            # Subtitle timestamps should be offset to start at 0
            assert "-output_ts_offset -1260" in cmd_str, "Should offset timestamps"


class TestCCExtraction:
    """Verify CC is extracted in fresh starts."""

    def test_build_cmd_includes_subtitles(self, transcoding_module):
        subtitles = [
            transcoding_module.SubtitleStream(index=2, lang="eng", name="English"),
        ]

        cmd = transcoding_module.build_hls_ffmpeg_cmd(
            "http://example.com/video.mkv",
            "software",
            "/tmp/test",
            is_vod=True,
            subtitles=subtitles,
        )

        cmd_str = " ".join(cmd)
        assert "-map 0:2" in cmd_str
        assert "sub0.vtt" in cmd_str
        assert "-c:s webvtt" in cmd_str

    def test_build_cmd_no_subtitles_when_none(self, transcoding_module):
        cmd = transcoding_module.build_hls_ffmpeg_cmd(
            "http://example.com/video.mkv",
            "software",
            "/tmp/test",
            is_vod=True,
            subtitles=None,
        )

        cmd_str = " ".join(cmd)
        assert "sub0.vtt" not in cmd_str
        assert "-c:s" not in cmd_str


class TestCompoundScenarios:
    """Compound scenarios: sequences of operations."""

    @pytest.mark.asyncio
    async def test_jump_then_resume(self, transcoding_module):
        """Jump -> close -> Resume: Should return cached with CC."""
        with tempfile.TemporaryDirectory() as output_dir:
            # After jump: seek_offset=1260, some segments transcoded
            make_playlist(output_dir, 20)  # 120s transcoded
            make_vtt(output_dir)  # CC from jump

            mock_process = mock.Mock()
            mock_process.returncode = -9  # Dead (user closed browser)

            existing = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 1260,  # From prior jump
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_jr"] = existing

            result = await transcoding_module._handle_existing_vod_session(
                "test_jr", "http://example.com/video.mkv", "software", False
            )

            assert result is not None
            assert result["seek_offset"] == 1260
            assert result["transcoded_duration"] == 120.0
            assert len(result["subtitles"]) > 0, "CC should work from cached VTT"

    @pytest.mark.asyncio
    async def test_resume_then_jump(self, transcoding_module):
        """Start -> close -> Resume -> Jump: Jump should work with CC."""
        with tempfile.TemporaryDirectory() as output_dir:
            # After resume (no prior jump): seek_offset=0
            make_playlist(output_dir, 50)  # 300s transcoded
            make_vtt(output_dir)
            make_session_json(output_dir, "test_rj", "http://example.com/video.mkv", seek_offset=0)

            mock_process = mock.Mock()
            mock_process.returncode = None  # Active after resume
            mock_process.kill = mock.Mock()
            mock_process.wait = mock.AsyncMock()

            session = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 0,
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_rj"] = session

            # Now jump to 1260
            with mock.patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = mock.AsyncMock()
                mock_proc.returncode = None
                mock_proc.pid = 12345
                mock_exec.return_value = mock_proc

                with mock.patch.object(transcoding_module, "_wait_for_playlist", return_value=True):
                    result = await transcoding_module.seek_transcode("test_rj", 1260.0)

            assert result["ok"] is True
            cmd_str = " ".join(mock_exec.call_args[0])
            assert "sub0.vtt" in cmd_str, "Jump should extract CC"

    @pytest.mark.asyncio
    async def test_jump_then_jump(self, transcoding_module):
        """Jump -> Jump: Second jump should work with CC."""
        with tempfile.TemporaryDirectory() as output_dir:
            # After first jump: seek_offset=1260
            make_playlist(output_dir, 10)
            make_vtt(output_dir)
            make_session_json(
                output_dir, "test_jj", "http://example.com/video.mkv", seek_offset=1260
            )

            mock_process = mock.Mock()
            mock_process.returncode = None
            mock_process.kill = mock.Mock()
            mock_process.wait = mock.AsyncMock()

            session = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 1260,
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_jj"] = session

            # Second jump to 2000
            with mock.patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = mock.AsyncMock()
                mock_proc.returncode = None
                mock_proc.pid = 12345
                mock_exec.return_value = mock_proc

                with mock.patch.object(transcoding_module, "_wait_for_playlist", return_value=True):
                    result = await transcoding_module.seek_transcode("test_jj", 2000.0)

            assert result["ok"] is True
            assert result["time"] == 2000.0
            cmd_str = " ".join(mock_exec.call_args[0])
            assert "-ss 2000" in cmd_str
            assert "sub0.vtt" in cmd_str, "Second jump should extract CC"

    @pytest.mark.asyncio
    async def test_jump_resume_jump(self, transcoding_module):
        """Jump -> close -> Resume -> Jump: Full cycle."""
        with tempfile.TemporaryDirectory() as output_dir:
            # State after Jump -> Resume: cached session with seek_offset=1260
            make_playlist(output_dir, 20)
            make_vtt(output_dir)
            make_session_json(
                output_dir, "test_jrj", "http://example.com/video.mkv", seek_offset=1260
            )

            mock_process = mock.Mock()
            mock_process.returncode = -9  # Dead from resume

            existing = {
                "dir": output_dir,
                "process": mock_process,
                "seek_offset": 1260,
                "url": "http://example.com/video.mkv",
                "is_vod": True,
                "subtitles": [{"index": 2, "lang": "eng", "name": "English"}],
                "duration": 2876.0,
            }

            with transcoding_module._transcode_lock:
                transcoding_module._transcode_sessions["test_jrj"] = existing

            # Resume returns cached
            result = await transcoding_module._handle_existing_vod_session(
                "test_jrj", "http://example.com/video.mkv", "software", False
            )
            assert result is not None
            assert result["seek_offset"] == 1260

            # Now user jumps to 2000
            existing["process"] = mock.Mock()
            existing["process"].returncode = None  # Simulate active after load
            existing["process"].kill = mock.Mock()
            existing["process"].wait = mock.AsyncMock()

            with mock.patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = mock.AsyncMock()
                mock_proc.returncode = None
                mock_proc.pid = 12345
                mock_exec.return_value = mock_proc

                with mock.patch.object(transcoding_module, "_wait_for_playlist", return_value=True):
                    result = await transcoding_module.seek_transcode("test_jrj", 2000.0)

            assert result["ok"] is True
            cmd_str = " ".join(mock_exec.call_args[0])
            assert "sub0.vtt" in cmd_str, "Jump after resume should extract CC"


class TestSeriesProbeCache:
    """Tests for series probe cache persistence and invalidation."""

    def test_series_cache_hit(self, transcoding_module):
        """Series probe cache should return cached result (exact episode match)."""
        media_info = transcoding_module.MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
            audio_channels=2,
            audio_sample_rate=48000,
            duration=2876.0,
        )
        subs = [transcoding_module.SubtitleStream(index=2, lang="eng", name="English")]

        # Populate cache: series 1234, episode 101 (new structure with name/episodes)
        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache[1234] = {
                "name": "Test Series",
                "episodes": {101: (time.time(), media_info, subs)},
            }

        # Should hit cache without calling ffprobe
        with mock.patch("subprocess.run") as mock_run:
            result_info, result_subs = transcoding_module.probe_media(
                "http://example.com/ep1.mkv", series_id=1234, episode_id=101
            )

        mock_run.assert_not_called()
        assert result_info is not None
        assert result_info.video_codec == "h264"
        assert len(result_subs) == 1
        # MRU should be updated to 101
        assert transcoding_module._series_probe_cache[1234].get("mru") == 101

        # Cleanup
        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache.clear()

    def test_series_cache_fallback(self, transcoding_module):
        """Series probe cache should fall back to most recent episode."""
        media_info = transcoding_module.MediaInfo(
            video_codec="hevc",
            audio_codec="aac",
            pix_fmt="yuv420p",
        )

        # Populate cache: series 1234, episode 101 only (new structure)
        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache[1234] = {
                "name": "Test Series",
                "mru": 101,
                "episodes": {101: (time.time(), media_info, [])},
            }

        # Request episode 102 (not cached) - should fall back to MRU (101)
        with mock.patch("subprocess.run") as mock_run:
            result_info, _ = transcoding_module.probe_media(
                "http://example.com/ep2.mkv", series_id=1234, episode_id=102
            )

        mock_run.assert_not_called()
        assert result_info is not None
        assert result_info.video_codec == "hevc"
        # MRU stays at 101 (fallback doesn't change MRU)
        assert transcoding_module._series_probe_cache[1234].get("mru") == 101

        # Cleanup
        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache.clear()

    def test_series_cache_invalidation(self, transcoding_module):
        """Invalidating series cache should clear MRU pointer."""
        media_info = transcoding_module.MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
        )

        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache[5678] = {
                "name": "Test",
                "mru": 101,
                "episodes": {101: (time.time(), media_info, [])},
            }

        assert 5678 in transcoding_module._series_probe_cache
        assert transcoding_module._series_probe_cache[5678].get("mru") == 101

        # Mock save to avoid file I/O
        with mock.patch.object(transcoding_module, "_save_series_probe_cache"):
            transcoding_module.invalidate_series_probe_cache(5678)

        # Series entry is deleted entirely
        assert 5678 not in transcoding_module._series_probe_cache

        # Cleanup
        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache.clear()

    def test_series_cache_invalidation_episode(self, transcoding_module):
        """Invalidating specific episode should keep other episodes."""
        media_info = transcoding_module.MediaInfo(
            video_codec="h264",
            audio_codec="aac",
            pix_fmt="yuv420p",
        )

        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache[5678] = {
                "name": "Test",
                "episodes": {
                    101: (time.time(), media_info, []),
                    102: (time.time(), media_info, []),
                },
            }

        # Mock save to avoid file I/O
        with mock.patch.object(transcoding_module, "_save_series_probe_cache"):
            transcoding_module.invalidate_series_probe_cache(5678, episode_id=101)

        assert 5678 in transcoding_module._series_probe_cache
        episodes = transcoding_module._series_probe_cache[5678]["episodes"]
        assert 101 not in episodes
        assert 102 in episodes

        # Cleanup
        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache.clear()

    def test_series_cache_save_load(self, transcoding_module):
        """Series cache should persist to and load from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = pathlib.Path(tmpdir) / "series_probe_cache.json"

            # Temporarily override cache file path
            original_file = transcoding_module._SERIES_PROBE_CACHE_FILE
            transcoding_module._SERIES_PROBE_CACHE_FILE = cache_file

            try:
                media_info = transcoding_module.MediaInfo(
                    video_codec="hevc",
                    audio_codec="eac3",
                    pix_fmt="yuv420p10le",
                    audio_channels=6,
                    audio_sample_rate=48000,
                    subtitle_codecs=["subrip"],
                    duration=3600.0,
                )
                subs = [transcoding_module.SubtitleStream(index=3, lang="spa", name="Spanish")]

                with transcoding_module._probe_lock:
                    transcoding_module._series_probe_cache[9999] = {
                        "name": "Test Show",
                        "episodes": {201: (time.time(), media_info, subs)},
                    }

                # Save
                transcoding_module._save_series_probe_cache()
                assert cache_file.exists()

                # Clear and reload
                with transcoding_module._probe_lock:
                    transcoding_module._series_probe_cache.clear()

                transcoding_module._load_series_probe_cache()

                assert 9999 in transcoding_module._series_probe_cache
                episodes = transcoding_module._series_probe_cache[9999]["episodes"]
                assert 201 in episodes
                _, loaded_info, loaded_subs = episodes[201]
                assert loaded_info.video_codec == "hevc"
                assert loaded_info.audio_codec == "eac3"
                assert len(loaded_subs) == 1
                assert loaded_subs[0].lang == "spa"
                assert transcoding_module._series_probe_cache[9999]["name"] == "Test Show"
            finally:
                transcoding_module._SERIES_PROBE_CACHE_FILE = original_file
                with transcoding_module._probe_lock:
                    transcoding_module._series_probe_cache.clear()

    def test_series_cache_populated_on_probe(self, transcoding_module):
        """Fresh probe with series_id should populate series cache."""
        probe_output = json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p"},
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "channels": 2,
                        "sample_rate": "48000",
                    },
                ],
                "format": {"duration": "1800.0"},
            }
        )

        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=probe_output)
            with mock.patch.object(transcoding_module, "_save_series_probe_cache"):
                result_info, _ = transcoding_module.probe_media(
                    "http://example.com/ep2.mkv", series_id=4321, episode_id=301, series_name="Test"
                )

        assert result_info is not None
        assert 4321 in transcoding_module._series_probe_cache
        episodes = transcoding_module._series_probe_cache[4321]["episodes"]
        assert 301 in episodes
        # Fresh probe sets MRU to current episode
        assert transcoding_module._series_probe_cache[4321].get("mru") == 301

        # Cleanup
        with transcoding_module._probe_lock:
            transcoding_module._series_probe_cache.clear()
            transcoding_module._probe_cache.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
