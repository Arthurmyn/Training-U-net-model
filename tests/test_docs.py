"""Documentation checks: README links, commands, generated tables, citation file, private paths."""
from __future__ import annotations

import importlib
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _markdown_files():
    return [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))


def test_relative_links_point_to_existing_files():
    missing = []
    for md in _markdown_files():
        for target in re.findall(r"\]\(([^)\s]+)\)", md.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (md.parent / target.split("#")[0]).resolve()
            if not path.exists():
                missing.append(f"{md.name}: {target}")
    assert not missing, missing


def test_readme_tables_match_the_result_files():
    from train.readme_tables import main
    assert main([]) == 0


def test_readme_abstract_numbers_are_in_the_generated_tables():
    blocks = "\n".join(re.findall(r"<!-- BEGIN:\w+ -->\n(.*?)\n<!-- END:\w+ -->", README, re.S))
    abstract = README.split("## 1. Abstract")[1].split("## 2.")[0]
    for number in re.findall(r"\b0\.\d{3}\b|\[0\.\d{3}, 0\.\d{3}\]", abstract):
        assert number in blocks, f"{number} from the abstract is not in the generated result tables"


def test_every_python_command_in_the_readme_exists():
    pytest.importorskip("torch")
    modules = sorted(set(re.findall(r"python -m ([\w.]+)", README)) - {"pytest"})
    assert {"audit.run", "train.run_all", "train.collect", "train.readme_tables"} <= set(modules)
    for name in modules:
        mod = importlib.import_module(name)
        assert hasattr(mod, "main"), f"{name} has no main()"
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])
        assert exc.value.code in (0, None), name


def test_readme_config_and_data_paths_exist():
    for rel in re.findall(r"`((?:configs|outputs|results|docs|requirements)/[\w./-]+\.(?:yaml|json|csv|md|txt))`", README):
        assert (ROOT / rel).exists(), rel
    from audit.config import load_config
    cfg = load_config(ROOT / "config.yaml")
    assert {"kvasir", "cvc"} <= set(cfg.datasets)
    pytest.importorskip("torch")
    from train.config import load_train_config
    eq = load_train_config(ROOT / "configs" / "equal_size.yaml")
    assert eq.seeds == [1, 2, 3, 4, 5] and eq.runs_dir.name == "runs_equal"


def test_citation_file_has_required_fields_and_no_invented_identifiers():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0" and cff["title"] and cff["authors"] and cff["message"]
    assert cff["repository-code"].startswith("https://github.com/")
    for forbidden in ("doi", "version", "date-released"):
        assert forbidden not in cff, f"{forbidden} must not be invented"
    assert cff.get("license") == "MIT" and "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")   # declared license must match the file
    assert all("family-names" in a and "given-names" in a for a in cff["authors"])


def test_no_private_paths_in_tracked_text_files():
    try:
        files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    except Exception:
        pytest.skip("not a git checkout")
    bad = []
    for rel in files:
        if rel.endswith((".png", ".pt", ".zip")):
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if re.search(r"[A-Za-z]:\\Users\\|/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/", text):
            bad.append(rel)
    assert not bad, bad
