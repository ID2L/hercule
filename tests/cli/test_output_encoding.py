"""Tests for CLI output stream hardening.

The CLI prints emoji status markers. On a stream whose encoding cannot represent them
(the Windows legacy code page, or any redirected stream whose preferred encoding is not
UTF-8), writing one raises UnicodeEncodeError on the very first message — killing the
command before it does any work. These tests pin the behaviour that prevents it.
"""

import io
import sys

import pytest

from hercule.cli.main import harden_output_streams


class FakeStream:
    """Minimal text stream recording how it was reconfigured."""

    def __init__(self, *, tty: bool, raises: type[Exception] | None = None):
        self._tty = tty
        self._raises = raises
        self.calls: list[dict[str, str]] = []

    def isatty(self) -> bool:
        return self._tty

    def reconfigure(self, **kwargs: str) -> None:
        if self._raises is not None:
            raise self._raises("stream refuses reconfiguration")
        self.calls.append(kwargs)


class StreamWithoutReconfigure:
    """Stand-in for a stream replaced by a test runner, which has no reconfigure()."""

    def isatty(self) -> bool:
        return False


class TestHardenOutputStreams:
    """Test cases for harden_output_streams()."""

    def test_redirected_stream_switches_to_utf8(self, monkeypatch):
        """A pipe or file is switched to UTF-8, which can always encode the markers."""
        stdout, stderr = FakeStream(tty=False), FakeStream(tty=False)
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)

        harden_output_streams()

        assert stdout.calls == [{"encoding": "utf-8"}]
        assert stderr.calls == [{"encoding": "utf-8"}]

    def test_interactive_console_keeps_encoding_and_gets_error_handler(self, monkeypatch):
        """A console keeps its encoding, so unsupported glyphs degrade instead of becoming mojibake."""
        stdout = FakeStream(tty=True)
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", FakeStream(tty=True))

        harden_output_streams()

        assert stdout.calls == [{"errors": "replace"}]

    def test_stream_without_reconfigure_is_skipped(self, monkeypatch):
        """Streams replaced by a test runner have no reconfigure(); startup must not fail."""
        monkeypatch.setattr(sys, "stdout", StreamWithoutReconfigure())
        monkeypatch.setattr(sys, "stderr", StreamWithoutReconfigure())

        harden_output_streams()  # must not raise

    @pytest.mark.parametrize("error", [OSError, ValueError])
    def test_refused_reconfiguration_is_tolerated(self, monkeypatch, error):
        """A stream that rejects reconfiguration is left as-is rather than crashing the CLI."""
        monkeypatch.setattr(sys, "stdout", FakeStream(tty=False, raises=error))
        monkeypatch.setattr(sys, "stderr", FakeStream(tty=True, raises=error))

        harden_output_streams()  # must not raise

    def test_emoji_survives_a_hardened_legacy_stream(self, monkeypatch):
        """End-to-end: a cp1252 stream can carry the CLI markers once hardened."""
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)

        with pytest.raises(UnicodeEncodeError):
            stream.write("\N{BAR CHART}")
            stream.flush()

        harden_output_streams()

        stream.write("\N{BAR CHART}")
        stream.flush()
        assert "\N{BAR CHART}".encode() in raw.getvalue()
