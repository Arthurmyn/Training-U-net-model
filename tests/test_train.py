"""Tests for the training package: model, metric, augmentation, data, a tiny CPU training run, analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")

from audit.config import load_config  # noqa: E402
from audit.run import run_audit, write_outputs  # noqa: E402
from train import analyze as analysis  # noqa: E402
from train.augment import augment  # noqa: E402
from train.config import load_train_config  # noqa: E402
from train.engine import train_run  # noqa: E402
from train.evaluate import evaluate_run  # noqa: E402
from train.metrics import bce_dice_loss, dice_per_image  # noqa: E402
from train.model import UNet  # noqa: E402
from train.verify_data import verify  # noqa: E402

from .conftest import make_dataset  # noqa: E402

BIG = 20.0     # logits that become probabilities of ~1 / ~0


def logits_from(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.where(mask > 0, BIG, -BIG).astype(np.float32))[None, None]


def test_unet_output_shape_and_parameters():
    m = UNet(3, 4)
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 1, 32, 32)
    assert sum(p.numel() for p in UNet(3, 32).parameters()) > 7_000_000


def test_dice_per_image_values():
    g = torch.zeros(1, 1, 8, 8)
    g[..., 2:6, 2:6] = 1
    assert dice_per_image(logits_from(g[0, 0].numpy()), g).item() == pytest.approx(1.0)
    other = torch.zeros(1, 1, 8, 8)
    other[..., 0:2, 0:2] = 1
    assert dice_per_image(logits_from(other[0, 0].numpy()), g).item() == 0.0
    half = torch.zeros(1, 1, 8, 8)
    half[..., 2:6, 2:4] = 1                                     # 8 px inside a 16 px target: 2*8 / (8+16)
    assert dice_per_image(logits_from(half[0, 0].numpy()), g).item() == pytest.approx(2 * 8 / 24)
    empty = torch.zeros(1, 1, 8, 8)
    assert dice_per_image(logits_from(empty[0, 0].numpy()), empty).item() == 1.0       # both empty
    assert dice_per_image(logits_from(empty[0, 0].numpy()), g).item() == 0.0           # only prediction empty
    assert bce_dice_loss(logits_from(g[0, 0].numpy()), g).item() < 0.01


def test_augmentation_keeps_image_and_mask_aligned_and_binary():
    torch.manual_seed(0)
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:20, 4:12] = 1
    img = mask.repeat(3, 1, 1).clone()                          # image is the mask itself
    flips = {"hflip": 1.0, "vflip": 1.0, "p_affine": 0.0}
    a_img, a_mask = augment(img, mask, flips)
    assert torch.equal(a_img[0], a_mask[0]) and not torch.equal(a_mask, mask)
    strong = {"hflip": 0.5, "vflip": 0.5, "p_affine": 1.0, "rotation_deg": 15, "scale": [0.9, 1.1]}
    for _ in range(20):
        a_img, a_mask = augment(img, mask, strong)
        assert set(a_mask.unique().tolist()) <= {0.0, 1.0}
        inter = ((a_img[0] > 0.5) & (a_mask[0] > 0.5)).sum().item()
        union = ((a_img[0] > 0.5) | (a_mask[0] > 0.5)).sum().item()
        assert union == 0 or inter / union > 0.8               # nearest vs bilinear differ only at the border


def _project(tmp: Path) -> Path:
    """Two tiny datasets, the audit (so splits exist) and a train config; returns train_config.yaml."""
    for key in ("a", "b"):
        make_dataset(tmp / "data" / key, n_good=24, defects=False)
    audit_cfg = {"seed": 5, "output_dir": "outputs", "preprocessing": {"size": 32},
                 "datasets": {k: {"name": k.upper(), "root": f"data/{k}", "image_dir": "images", "mask_dir": "masks", "expected_count": 24,
                                  "split": {"method": "random", "sizes": {"train": 16, "val": 4, "test": 4}}} for k in ("a", "b")}}
    (tmp / "config.yaml").write_text(yaml.safe_dump(audit_cfg))
    write_outputs(run_audit(load_config(tmp / "config.yaml"), progress=False))
    train_cfg = {"audit_config": "config.yaml", "runs_dir": "runs", "results_dir": "results", "sources": ["a", "b"], "seeds": [1, 2],
                 "model": {"base_channels": 4},
                 "training": {"batch_size": 4, "max_epochs": 2, "patience": 5, "lr": 0.003, "device": "cpu", "num_workers": 0, "threshold": 0.5},
                 "augmentation": {"hflip": 0.5, "vflip": 0.5, "p_affine": 0.5, "rotation_deg": 10, "scale": [0.9, 1.1]},
                 "analysis": {"bootstrap_samples": 300, "seed": 1, "ci_level": 0.95, "practical_thresholds": [0.05, 0.03, 0.10]}}
    p = tmp / "train_config.yaml"
    p.write_text(yaml.safe_dump(train_cfg))
    return p


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return _project(tmp_path_factory.mktemp("proj"))


def test_train_evaluate_resume_and_determinism(project, tmp_path):
    cfg = load_train_config(project)
    d = train_run(cfg, "a", 1, progress=False)
    assert (d / "best.pt").is_file() and (d / "last.pt").is_file()
    meta = json.loads((d / "run.json").read_text())
    assert meta["status"] == "finished" and meta["n_train"] == 16 and meta["n_val"] == 4 and meta["source"] == "a"
    metrics = pd.read_csv(d / "metrics.csv")
    assert metrics["epoch"].tolist() == [1, 2]

    csv = evaluate_run(cfg, d, progress=False, examples=False)
    r = pd.read_csv(csv)
    assert set(r["eval_set"]) == {"a_val", "a_test", "b_external"}
    assert (r[r.eval_set == "a_test"].shape[0], r[r.eval_set == "b_external"].shape[0]) == (4, 24)
    assert r["dice"].between(0, 1).all() and set(r["kind"]) == {"in_domain", "external"}
    ext = r[r.eval_set == "b_external"]
    assert ext["target_split"].value_counts().to_dict() == {"train": 16, "val": 4, "test": 4}   # B's own split labels
    assert (d / "eval_summary.json").is_file()

    assert train_run(cfg, "a", 1, progress=False) == d                      # finished run is skipped, not retrained
    assert pd.read_csv(d / "metrics.csv").shape[0] == 2

    # resume: pretend the run was interrupted after epoch 1
    one = cfg.override(**{"training.max_epochs": 1, "runs_dir": "runs_resume"})
    d2 = train_run(one, "a", 3, progress=False)
    meta2 = json.loads((d2 / "run.json").read_text())
    meta2["status"] = "running"
    (d2 / "run.json").write_text(json.dumps(meta2))
    two = cfg.override(**{"training.max_epochs": 2, "runs_dir": "runs_resume"})
    train_run(two, "a", 3, progress=False)
    assert pd.read_csv(d2 / "metrics.csv")["epoch"].tolist() == [1, 2]

    # same seed, new folder -> identical first epoch (CPU)
    d3 = train_run(cfg.override(**{"runs_dir": "runs_again"}), "a", 1, progress=False)
    assert pd.read_csv(d3 / "metrics.csv")["train_loss"].iloc[0] == pytest.approx(metrics["train_loss"].iloc[0], abs=1e-6)


def test_verify_data_detects_changed_file(tmp_path):
    cfg = load_train_config(_project(tmp_path))
    assert verify(cfg, progress=False) is True
    target = sorted((tmp_path / "data" / "a" / "images").iterdir())[0]
    Image.new("RGB", (48, 32), (1, 2, 3)).save(target)
    assert verify(cfg, progress=False) is False


def test_cluster_bootstrap_uses_clusters():
    rng = np.random.default_rng(0)
    values = np.repeat(rng.normal(0.7, 0.1, 10), 20)           # 10 sequences x 20 near-identical frames
    clusters = np.repeat(np.arange(10), 20)
    by_cluster = analysis.cluster_bootstrap_means(values, clusters, 2000, np.random.default_rng(1))
    by_frame = analysis.cluster_bootstrap_means(values, np.arange(200), 2000, np.random.default_rng(1))
    assert by_cluster.std() > 2 * by_frame.std()               # ignoring sequences would give far too narrow intervals
    assert analysis.cluster_bootstrap_means(np.full(30, 0.5), np.arange(30), 100, rng).std() == 0


def test_analysis_on_synthetic_results(project):
    cfg = load_train_config(project).override(**{"runs_dir": "runs_fake", "results_dir": "results_fake"})
    rng = np.random.default_rng(3)
    for src, tgt in (("a", "b"), ("b", "a")):
        for seed in (1, 2):
            rows = []
            def add(eval_set, kind, target, n, mean, parts, offset=0):
                for i in range(n):
                    rows.append({"run": f"{src}_seed{seed}", "source": src, "seed": seed, "eval_set": eval_set, "kind": kind, "target": target,
                                 "dice": float(np.clip(rng.normal(mean, 0.03), 0, 1)), "gt_area": 0.1, "pred_area": 0.1, "stem": f"{target}{offset + i}",
                                 "image_file": f"{target}{offset + i}.png", "sequence_id": None, "target_split": parts[i]})
            add(f"{src}_val", "in_domain", src, 4, 0.80, ["val"] * 4, offset=12)
            add(f"{src}_test", "in_domain", src, 8, 0.80, ["test"] * 8, offset=16)      # same images as the external set's test part
            add(f"{tgt}_external", "external", tgt, 24, 0.60, ["train"] * 12 + ["val"] * 4 + ["test"] * 8)
            d = cfg.runs_dir / f"{src}_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(d / "per_image_dice.csv", index=False)
    summary = analysis.analyze(cfg)
    t2 = pd.read_csv(cfg.results_dir / "tables" / "dice_drop_external.csv")
    assert len(t2) == 2 and t2["drop (in-domain - external)"].between(0.1, 0.3).all()
    assert t2["reading at 0.05"].str.startswith("drop > 0.05").all()
    t3 = pd.read_csv(cfg.results_dir / "tables" / "paired_same_target_images.csv")
    assert len(t3) == 2 and (t3["images"] == 8).all() and t3["difference (own - other)"].between(0.1, 0.3).all()
    text = summary.read_text()
    assert "Cause" in text and (cfg.results_dir / "fig_dice_distributions.png").is_file()
    assert str(cfg.base_dir) not in text


# ---------------------------------------------------------------- equal-size training: config inheritance, subset, min_epochs
def test_config_inheritance_and_project_root(tmp_path):
    root = _project(tmp_path)
    (tmp_path / "configs").mkdir()
    child = {"project_root": "..", "base": "train_config.yaml", "runs_dir": "runs_equal", "seeds": [1, 2, 3, 4, 5],
             "training": {"min_epochs": 7}}
    (tmp_path / "configs" / "equal.yaml").write_text(yaml.safe_dump(child))
    cfg = load_train_config(tmp_path / "configs" / "equal.yaml")
    assert cfg.base_dir == tmp_path.resolve() and cfg.runs_dir == (tmp_path / "runs_equal").resolve()
    assert cfg.seeds == [1, 2, 3, 4, 5]                                                   # lists are replaced
    assert cfg.section("training")["min_epochs"] == 7 and cfg.section("training")["batch_size"] == 4   # dicts are merged
    assert cfg.audit.datasets["a"].root == (tmp_path / "data" / "a").resolve()
    assert root.exists()


def test_fixed_subset_is_deterministic_reduces_only_train_and_is_validated(tmp_path):
    from train.data import load_items
    from train.subset import make_train_subset, select_subset
    files = [f"{i}.png" for i in range(1, 41)]
    a, b, c = select_subset(files, 10, 42), select_subset(list(reversed(files)), 10, 42), select_subset(files, 10, 43)
    assert a == b and a != c and len(set(a)) == 10 and set(a) <= set(files)
    with pytest.raises(ValueError):
        select_subset(files, 41, 1)
    cfg = load_train_config(_project(tmp_path)).override(**{"training.train_subset": {"a": {"size": 8, "seed": 42}}})
    parts_full, ext_full = load_items(cfg.override(**{"training.train_subset": {}}), "a")
    with pytest.raises(FileNotFoundError):
        load_items(cfg, "a")                                                              # subset list not written yet
    path = make_train_subset(cfg, "a", 8, 42)
    parts, ext = load_items(cfg, "a")
    assert len(parts["train"]) == 8 and len(parts_full["train"]) == 16
    assert {i.stem for i in parts["train"]} <= {i.stem for i in parts_full["train"]}
    assert [i.stem for i in parts["val"]] == [i.stem for i in parts_full["val"]] and len(parts["test"]) == len(parts_full["test"])
    assert len(ext) == len(ext_full) == 24
    body = json.loads(path.read_text())
    body["images"][0] = "not_in_train.png"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        load_items(cfg, "a")                                                              # edited list no longer matches the split


def test_min_epochs_delays_early_stopping(tmp_path):
    cfg = load_train_config(_project(tmp_path)).override(**{"training.max_epochs": 6, "training.patience": 1, "training.lr": 1e-9,
                                                            "training.min_epochs": 4, "runs_dir": "runs_min"})
    d = train_run(cfg, "a", 1, progress=False)
    assert json.loads((d / "run.json").read_text())["epochs_run"] >= 4                    # patience 1 alone would stop after epoch 2
    d0 = train_run(cfg.override(**{"training.min_epochs": 0, "runs_dir": "runs_nomin"}), "a", 1, progress=False)
    assert json.loads((d0 / "run.json").read_text())["epochs_run"] <= json.loads((d / "run.json").read_text())["epochs_run"]


# ---------------------------------------------------------------- end to end: collect everything in one folder (tiny CPU run)
def test_collect_builds_the_whole_folder(tmp_path):
    from train.collect import collect
    from train.evaluate import evaluate_run
    cfg_file = _project(tmp_path)
    cfg = load_train_config(cfg_file).override(**{"seeds": [1, 2], "training.max_epochs": 2})
    for source in ("a", "b"):
        for seed in (1, 2):
            evaluate_run(cfg, train_run(cfg, source, seed, progress=False), progress=False, examples=False)
    out = collect(cfg, cfg_file, tmp_path / "final", skip_inference=False, with_examples=True, progress=False)
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    for need in ("README.md", "MANIFEST.json", "METRICS.md", "tables/gap_summary.csv", "tables/decision_rule.csv", "tables/metrics_by_condition.csv",
                 "per_image/per_image_metrics_extended.csv", "figures/fig01_study_design.png", "figures/fig03_dice_by_condition.png",
                 "figures/fig05_all_metrics.png", "figures/fig06_gap_all_metrics.png", "figures/fig07_dice_by_polyp_size.png",
                 "figures/fig08_examples_E2.png", "figures/fig08_examples_E4.png", "figures/figA1_training_curves.png", "figures/figA2_delta_sensitivity.png",
                 "runs/a_seed1/metrics.csv", "runs/b_seed2/per_image_dice.csv", "config/train_config.yaml", "config/splits/a_split.json", "audit/audit_report.md"):
        assert need in names, need
    assert not any(n.endswith(".pt") for n in names)                       # weights only with --with-weights
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["n_files"] == len(names) - 1 and set(manifest["runs"]) == {"a_seed1", "a_seed2", "b_seed1", "b_seed2"}
    gap = pd.read_csv(out / "tables" / "gap_summary.csv")
    assert set(gap["n_seeds"]) == {2}
    readme = (out / "README.md").read_text()
    assert "Key numbers" in readme and str(tmp_path) not in readme and str(tmp_path) not in (out / "MANIFEST.json").read_text()


# ---------------------------------------------------------------- GPU check
def test_gpu_check_reports_cpu_only_clearly(tmp_path, capsys):
    from train.check_gpu import check, report
    cfg = load_train_config(_project(tmp_path)).override(**{"training.device": "cpu"})
    info = check(cfg, steps=2)
    assert info["device_type"] == "cpu" and info["seconds_per_step"] > 0 and set(info["per_source"]) == {"a", "b"}
    assert info["per_source"]["a"]["train_images"] == 16 and info["per_source"]["a"]["steps_per_epoch"] == 4
    assert report(info) == 1                                                              # CPU only -> exit code 1
    out = capsys.readouterr().out
    assert "NO GPU WILL BE USED" in out and "SETUP_ru.md" in out


def test_forcing_an_unavailable_device_fails_loudly(tmp_path):
    from train.engine import select_device
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            select_device("cuda")
    assert select_device("cpu").type == "cpu"
