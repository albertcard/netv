"""Tests for epg.py."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest


@pytest.fixture
def epg_module():
    """Import epg module."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import epg

    return epg


class TestParseEpgTime:
    def test_parse_basic_time(self, epg_module):
        result = epg_module.parse_epg_time("20241130120000")
        assert result.year == 2024
        assert result.month == 11
        assert result.day == 30
        assert result.hour == 12
        assert result.minute == 0
        assert result.second == 0

    def test_parse_time_with_positive_offset(self, epg_module):
        result = epg_module.parse_epg_time("20241130120000 +0530")
        assert result.hour == 12
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(hours=5, minutes=30)

    def test_parse_time_with_negative_offset(self, epg_module):
        result = epg_module.parse_epg_time("20241130120000-0500")
        assert result.utcoffset() == timedelta(hours=-5)

    def test_parse_time_with_zero_offset(self, epg_module):
        result = epg_module.parse_epg_time("20241130120000 +0000")
        assert result.utcoffset() == timedelta(0)

    def test_parse_invalid_returns_now(self, epg_module):
        result = epg_module.parse_epg_time("invalid")
        assert (datetime.now(UTC) - result).total_seconds() < 2


class TestProgram:
    def test_program_dataclass(self, epg_module):
        prog = epg_module.Program(
            channel_id="ch1",
            title="Test Show",
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            stop=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
            desc="Description",
        )
        assert prog.channel_id == "ch1"
        assert prog.title == "Test Show"
        assert prog.source_id == ""


class TestEPGData:
    def test_epg_data_defaults(self, epg_module):
        epg = epg_module.EPGData()
        assert epg.channels == {}
        assert epg.icons == {}
        assert epg.programs == {}


class TestGetProgramsInRange:
    def test_programs_in_range(self, epg_module):
        epg = epg_module.EPGData()
        epg.programs["ch1"] = [
            epg_module.Program(
                "ch1",
                "Show 1",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
            ),
            epg_module.Program(
                "ch1",
                "Show 2",
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            ),
            epg_module.Program(
                "ch1",
                "Show 3",
                datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
            ),
        ]

        # Get programs from 10:30 to 11:30
        result = epg_module.get_programs_in_range(
            epg,
            "ch1",
            datetime(2024, 1, 1, 10, 30, tzinfo=UTC),
            datetime(2024, 1, 1, 11, 30, tzinfo=UTC),
        )
        assert len(result) == 2
        assert result[0].title == "Show 1"
        assert result[1].title == "Show 2"

    def test_programs_empty_channel(self, epg_module):
        epg = epg_module.EPGData()
        result = epg_module.get_programs_in_range(
            epg,
            "nonexistent",
            datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        assert result == []

    def test_programs_prefer_source(self, epg_module):
        epg = epg_module.EPGData()
        epg.programs["ch1"] = [
            epg_module.Program(
                "ch1",
                "Show A",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                source_id="source1",
            ),
            epg_module.Program(
                "ch1",
                "Show B",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                source_id="source2",
            ),
        ]

        result = epg_module.get_programs_in_range(
            epg,
            "ch1",
            datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            preferred_source_id="source2",
        )
        assert len(result) == 1
        assert result[0].title == "Show B"


class TestSanitizeEpgXml:
    def test_sanitize_extracts_channels_and_programmes(self, epg_module):
        xml = """
        <tv>
        <channel id="ch1"><display-name>Channel 1</display-name></channel>
        <corrupted>
        <programme start="20240101120000 +0000" stop="20240101130000 +0000" channel="ch1">
            <title>Test</title>
        </programme>
        """
        result = epg_module._sanitize_epg_xml(xml)
        assert "<channel" in result
        assert "<programme" in result
        assert '<?xml version="1.0"?>' in result
