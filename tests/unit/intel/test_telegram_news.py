"""Tests for the Telegram channel news collector — parsing, matching, rows."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphavedha.intel.collectors.telegram_news import (
    TelegramMessage,
    configured_channels,
    message_rows,
    parse_channel_messages,
)
from alphavedha.intel.symbols import match_symbols

_PREVIEW_HTML = """
<html><body>
<section class="tgme_channel_history js-message_history">

<div class="tgme_widget_message_wrap js-widget_message_wrap">
 <div class="tgme_widget_message text_not_supported_wrap js-widget_message"
      data-post="moneycontrolcom/424242" data-view="abc">
  <div class="tgme_widget_message_bubble">
   <div class="tgme_widget_message_text js-message_text" dir="auto">HDFC Bank Q1
net profit rises 12% <b>YoY</b>; asset quality stable<br/>Details:
<a href="https://www.moneycontrol.com/x">moneycontrol.com</a></div>
   <div class="tgme_widget_message_footer compact js-message_footer">
    <span class="tgme_widget_message_meta">
     <a class="tgme_widget_message_date" href="https://t.me/moneycontrolcom/424242">
      <time datetime="2026-07-07T08:42:13+00:00" class="time">08:42</time>
     </a>
    </span>
   </div>
  </div>
 </div>
</div>

<div class="tgme_widget_message_wrap js-widget_message_wrap">
 <div class="tgme_widget_message" data-post="moneycontrolcom/424243">
  <div class="tgme_widget_message_bubble">
   <a class="tgme_widget_message_photo_wrap" href="https://t.me/moneycontrolcom/424243">
    <div class="tgme_widget_message_photo"></div>
   </a>
   <time datetime="2026-07-07T08:50:00+00:00" class="time">08:50</time>
  </div>
 </div>
</div>

<div class="tgme_widget_message_wrap js-widget_message_wrap">
 <div class="tgme_widget_message" data-post="moneycontrolcom/424244">
  <div class="tgme_widget_message_bubble">
   <div class="tgme_widget_message_text js-message_text" dir="auto">Tata Steel and
JSW Steel rally as M&amp;M unveils new SUV</div>
   <time datetime="2026-07-07T09:05:30+00:00" class="time">09:05</time>
  </div>
 </div>
</div>

</section>
</body></html>
"""


class TestParseChannelMessages:
    def test_parses_text_messages_and_skips_photo_only(self) -> None:
        messages = parse_channel_messages(_PREVIEW_HTML, "moneycontrolcom")
        assert [m.msg_id for m in messages] == [424242, 424244]

    def test_message_fields(self) -> None:
        msg = parse_channel_messages(_PREVIEW_HTML, "moneycontrolcom")[0]
        assert msg.channel == "moneycontrolcom"
        assert msg.url == "https://t.me/moneycontrolcom/424242"
        assert msg.posted_at == datetime(2026, 7, 7, 8, 42, 13, tzinfo=UTC)
        assert msg.posted_at.tzinfo is not None

    def test_inline_markup_flattened_and_br_becomes_newline(self) -> None:
        msg = parse_channel_messages(_PREVIEW_HTML, "moneycontrolcom")[0]
        assert "YoY" in msg.text  # <b> content kept
        assert "\n" in msg.text  # <br/> preserved as newline
        assert "<" not in msg.text  # no leftover tags

    def test_html_entities_decoded(self) -> None:
        msg = parse_channel_messages(_PREVIEW_HTML, "moneycontrolcom")[1]
        assert "M&M" in msg.text

    def test_empty_html(self) -> None:
        assert parse_channel_messages("", "x") == []
        assert parse_channel_messages("<html><body></body></html>", "x") == []


class TestMatchSymbols:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("HDFC Bank Q1 profit rises", ["HDFCBANK"]),
            ("TCS wins mega deal from European client", ["TCS"]),
            ("Reliance Industries and Infosys lead gains", ["INFY", "RELIANCE"]),
            ("L&T bags Middle East order", ["LT"]),
            ("Zomato parent Eternal expands Blinkit", ["ETERNAL"]),
            ("Gold prices steady ahead of Fed meet", []),
        ],
    )
    def test_matches(self, text: str, expected: list[str]) -> None:
        assert match_symbols(text) == expected

    def test_ticker_requires_exact_uppercase(self) -> None:
        assert match_symbols("Lt Gen Sharma appointed to board") == []
        assert match_symbols("tcs lowercase mention") == []

    def test_word_boundary_prevents_substring_hits(self) -> None:
        assert match_symbols("Titanium dioxide imports surge") == []

    def test_longer_alias_of_other_symbol_wins(self) -> None:
        # "SBI Life" must tag SBILIFE only — the inner "SBI" span is contained.
        assert match_symbols("SBI Life Q1 premium up 14%") == ["SBILIFE"]
        assert match_symbols("SBI cuts lending rates") == ["SBIN"]

    def test_empty_text(self) -> None:
        assert match_symbols("") == []


class TestMessageRows:
    def _msg(self, text: str) -> TelegramMessage:
        return TelegramMessage(
            channel="moneycontrolcom",
            msg_id=1,
            text=text,
            posted_at=datetime(2026, 7, 7, 8, 42, 13, tzinfo=UTC),
            url="https://t.me/moneycontrolcom/1",
        )

    def test_one_row_per_matched_symbol(self) -> None:
        rows = message_rows([self._msg("Tata Steel and JSW Steel rally on China cues")])
        assert [r["symbol"] for r in rows] == ["JSWSTEEL", "TATASTEEL"]
        assert rows[0]["text_hash"] == rows[1]["text_hash"]

    def test_row_shape(self) -> None:
        row = message_rows([self._msg("Infosys guidance raised\nFull details inside")])[0]
        assert row["source"] == "TELEGRAM"
        assert row["category"] == "news"
        assert row["headline"] == "Infosys guidance raised"  # first line only
        assert row["url"] == "https://t.me/moneycontrolcom/1"
        filed_at = row["filed_at"]
        assert isinstance(filed_at, datetime)
        assert filed_at.utcoffset() is not None
        assert filed_at.utcoffset().total_seconds() == 5.5 * 3600  # IST

    def test_unmatched_message_dropped(self) -> None:
        assert message_rows([self._msg("Crude oil slips below $80")]) == []


class TestConfiguredChannels:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_NEWS_CHANNELS", raising=False)
        assert configured_channels() == ["moneycontrolcom", "ndtvprofitnews", "cnbc_tv18"]

    def test_env_override_strips_handles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_NEWS_CHANNELS", "@foo, bar ,")
        assert configured_channels() == ["foo", "bar"]
