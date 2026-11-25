"""Tiện ích dùng chung: cấu hình, đường dẫn, lưu hình/bảng, sổ đăng ký kết quả."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("x5")


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # cho phép chạy chế độ kiểm thử với dữ liệu tổng hợp (không dùng cho báo cáo)
    if os.environ.get("X5_SMOKE") == "1":
        cfg["paths"] = {k: v.replace("data/", "data_smoke/").replace("results", "results_smoke")
                        .replace("models", "models_smoke") for k, v in cfg["paths"].items()}
        cfg["causal"]["n_bootstrap"] = 20
        cfg["milp"]["n_bootstrap"] = 10
        cfg["spo"]["epochs"] = 5
        cfg["mlops"]["latency_requests"] = 200
        cfg["router"]["review_capacity"] = 50
    for k, v in cfg["paths"].items():
        p = ROOT / v
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = p
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------- hình & bảng
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10,
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


PALETTE = ["#2A6F97", "#E07A5F", "#3D9970", "#8E7DBE", "#F2CC8F", "#6C757D", "#C44536", "#1B998B"]


def save_fig(fig, cfg: dict, name: str, caption: str) -> Path:
    path = cfg["paths"]["figures"] / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    _register(cfg, "figures", name, {"file": path.name, "caption": caption})
    return path


def save_table(df: pd.DataFrame, cfg: dict, name: str, caption: str, index: bool = False,
               float_fmt: int = 4) -> Path:
    path = cfg["paths"]["tables"] / f"{name}.csv"
    out = df.copy()
    for c in out.select_dtypes(include=[np.floating]).columns:
        out[c] = out[c].round(float_fmt)
    out.to_csv(path, index=index, encoding="utf-8-sig")
    _register(cfg, "tables", name, {"file": path.name, "caption": caption})
    return path


def _register(cfg, kind, name, meta):
    reg_path = cfg["paths"]["results"] / "registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    reg.setdefault(kind, {})[name] = meta
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def save_metrics(cfg: dict, stage: str, metrics: dict) -> None:
    """Lưu chỉ số của từng giai đoạn vào results/metrics.json (nguồn số liệu duy nhất cho báo cáo)."""
    path = cfg["paths"]["results"] / "metrics.json"
    allm = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    allm[stage] = _to_py(metrics)
    path.write_text(json.dumps(allm, ensure_ascii=False, indent=2), encoding="utf-8")


def load_metrics(cfg: dict) -> dict:
    path = cfg["paths"]["results"] / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _to_py(o):
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o


def sha256_file(path: str | os.PathLike, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def timer(label: str, store: dict | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    log.info(f"[time] {label}: {dt:.1f}s")
    if store is not None:
        store[label] = dt
