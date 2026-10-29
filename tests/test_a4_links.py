"""Link checker: extraction, relative links and failure rules (no network)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("check_links", ROOT / "a4" / "check_links.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_extract_strips_punctuation_and_separates_relative_links():
    m = _mod()
    urls, rel = m.extract("See https://example.org/a. And [x](docs/a.md#sec), [y](https://example.org/b), [z](#top), <https://example.org/c>;")
    assert urls == ["https://example.org/a", "https://example.org/b", "https://example.org/c"] and rel == ["docs/a.md#sec"]


def test_check_reports_missing_files_and_bad_status(tmp_path):
    m = _mod()
    (tmp_path / "a4").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ok.md").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("[a](docs/ok.md) [b](docs/missing.md) https://good.example https://bad.example https://down.example", encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("url: https://good.example\n", encoding="utf-8")
    codes = {"https://good.example": "200", "https://bad.example": "404", "https://down.example": "URLError"}
    res = m.check(tmp_path, codes.__getitem__)
    bad = m.failures(res)
    assert {b[1] for b in bad} == {"https://bad.example", "https://down.example", "docs/missing.md"}
    assert len([r for r in res["urls"] if r[1] == "https://good.example"]) == 2


def test_status_without_network_is_a_failure_name():
    m = _mod()
    assert not m.fetch_status("http://127.0.0.1:9/", timeout=1).isdigit()
