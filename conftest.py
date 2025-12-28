"""Pytest configuration for iptv tests."""

import warnings


# Suppress unawaited coroutine warnings from AsyncMock in tests.
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
