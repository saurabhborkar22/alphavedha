"""Tests for ASM/GSM surveillance list collector."""

from __future__ import annotations

from datetime import date

from alphavedha.intel.collectors.surveillance import (
    _parse_asm_response,
    _parse_gsm_response,
)


class TestParseAsmResponse:
    def test_parses_longterm_data(self) -> None:
        data = {
            "longterm": {
                "data": [
                    {
                        "symbol": "ABC",
                        "asmSurvIndicator": "Stage I",
                        "survCode": "LTASM - I (13)",
                        "asmTime": "19-Jun-2026",
                    }
                ]
            },
            "shortterm": {"data": []},
        }
        rows = _parse_asm_response(data)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "ABC.NS"
        assert rows[0]["list_name"] == "LTASM-Stage I"
        assert rows[0]["added_on"] == date.today()

    def test_parses_shortterm_data(self) -> None:
        data = {
            "longterm": {"data": []},
            "shortterm": {
                "data": [
                    {
                        "symbol": "XYZ",
                        "asmSurvIndicator": "Stage II",
                        "survCode": "STASM - II",
                    }
                ]
            },
        }
        rows = _parse_asm_response(data)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "XYZ.NS"
        assert rows[0]["list_name"] == "STASM-Stage II"

    def test_skips_empty_symbol(self) -> None:
        data = {
            "longterm": {"data": [{"symbol": "", "asmSurvIndicator": "Stage I", "survCode": ""}]},
            "shortterm": {"data": []},
        }
        rows = _parse_asm_response(data)
        assert len(rows) == 0

    def test_handles_missing_sections(self) -> None:
        rows = _parse_asm_response({})
        assert rows == []

    def test_handles_non_dict_section(self) -> None:
        data = {"longterm": "not a dict", "shortterm": {"data": []}}
        rows = _parse_asm_response(data)
        assert rows == []

    def test_multiple_items(self) -> None:
        data = {
            "longterm": {
                "data": [
                    {"symbol": "A", "asmSurvIndicator": "Stage I", "survCode": ""},
                    {"symbol": "B", "asmSurvIndicator": "Stage II", "survCode": ""},
                ]
            },
            "shortterm": {
                "data": [
                    {"symbol": "C", "asmSurvIndicator": "Stage I", "survCode": ""},
                ]
            },
        }
        rows = _parse_asm_response(data)
        assert len(rows) == 3
        symbols = {r["symbol"] for r in rows}
        assert symbols == {"A.NS", "B.NS", "C.NS"}

    def test_truncates_long_list_name(self) -> None:
        data = {
            "longterm": {
                "data": [
                    {
                        "symbol": "ABC",
                        "asmSurvIndicator": "Very Long Stage Name Here",
                        "survCode": "",
                    }
                ]
            },
            "shortterm": {"data": []},
        }
        rows = _parse_asm_response(data)
        assert len(rows[0]["list_name"]) <= 20


class TestParseGsmResponse:
    def test_parses_gsm_items(self) -> None:
        data = [
            {"symbol": "DEF", "gsmSurvIndicator": "Stage III"},
        ]
        rows = _parse_gsm_response(data)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "DEF.NS"
        assert rows[0]["list_name"] == "GSM-Stage III"
        assert rows[0]["added_on"] == date.today()

    def test_skips_empty_symbol(self) -> None:
        rows = _parse_gsm_response([{"symbol": "", "gsmSurvIndicator": "Stage I"}])
        assert len(rows) == 0

    def test_handles_non_list(self) -> None:
        rows = _parse_gsm_response({"not": "a list"})
        assert rows == []

    def test_handles_empty_list(self) -> None:
        rows = _parse_gsm_response([])
        assert rows == []

    def test_default_list_name_without_indicator(self) -> None:
        rows = _parse_gsm_response([{"symbol": "GHI", "gsmSurvIndicator": ""}])
        assert len(rows) == 1
        assert rows[0]["list_name"] == "GSM"
