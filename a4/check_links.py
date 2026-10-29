"""Check every URL in README.md, CITATION.cff and a4/*.md and every repo-relative Markdown link: ``python a4/check_links.py``.

URLs are requested without authentication (HEAD, then GET). Exit code 0 only if every URL answered with status < 400 and
every relative link points to an existing file. An unreachable URL (no network, timeout) counts as a failure.
"""
from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
URL = re.compile(r"https?://[^\s<>\")\]]+")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def files(root: Path = ROOT) -> List[Path]:
    return [p for p in [root / "README.md", root / "CITATION.cff", *sorted((root / "a4").glob("*.md")), *sorted((root / "a4" / "tables").glob("*.md"))] if p.is_file()]


def extract(text: str) -> Tuple[List[str], List[str]]:
    """(URLs, relative links) of a text; trailing punctuation is stripped from URLs."""
    urls = sorted({u.rstrip(".,;:'") for u in URL.findall(text)})
    rel = sorted({m for m in LINK.findall(text) if not m.startswith(("http://", "https://", "mailto:", "#"))})
    return urls, rel


def fetch_status(url: str, timeout: float = 20.0) -> str:
    """HTTP status code as text, or the error name."""
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": "link-check/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return str(r.status)
        except urllib.error.HTTPError as exc:
            if method == "GET" or exc.code not in (403, 405, 400, 501):
                return str(exc.code)
        except Exception as exc:  # noqa: BLE001 - network problems are reported, not raised
            if method == "GET":
                return type(exc).__name__
    return "unknown"


def check(root: Path = ROOT, status: Callable[[str], str] = fetch_status) -> Dict[str, List[Tuple[str, str, str]]]:
    """Results as {'urls': [(file, url, status)], 'relative': [(file, link, 'ok'|'missing')]}."""
    out: Dict[str, List[Tuple[str, str, str]]] = {"urls": [], "relative": []}
    cache: Dict[str, str] = {}
    for p in files(root):
        urls, rel = extract(p.read_text(encoding="utf-8"))
        for u in urls:
            cache.setdefault(u, status(u))
            out["urls"].append((p.relative_to(root).as_posix(), u, cache[u]))
        for link in rel:
            target = (p.parent / link.split("#")[0]).resolve()
            out["relative"].append((p.relative_to(root).as_posix(), link, "ok" if target.exists() else "missing"))
    return out


def failures(result: Dict[str, List[Tuple[str, str, str]]]) -> List[Tuple[str, str, str]]:
    bad = [r for r in result["urls"] if not (r[2].isdigit() and int(r[2]) < 400)]
    return bad + [r for r in result["relative"] if r[2] != "ok"]


def main() -> int:
    result = check()
    for kind in ("urls", "relative"):
        for f, target, st in result[kind]:
            print(f"{st:>6}  {f}  {target}")
    bad = failures(result)
    print(f"\n{len(result['urls'])} URL(s), {len(result['relative'])} relative link(s), {len(bad)} failure(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
