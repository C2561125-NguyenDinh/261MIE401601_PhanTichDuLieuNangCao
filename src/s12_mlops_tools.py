"""Giai đoạn bổ trợ 12 - Hiện thực tầng MLOps bằng công cụ chuẩn của đề cương.

  (A) MLflow: ghi tham số, chỉ số, tạo tác và đăng ký mô hình vào Model Registry (backend SQLite cục bộ).
  (B) Evidently: báo cáo trôi dạt dữ liệu Train so với OOT (HTML) và đối chiếu với PSI tự cài đặt (giai đoạn 8).
  (C) FastAPI: kiểm thử dịch vụ chấm điểm serve/api.py bằng TestClient: tính đúng (so với chấm điểm theo lô),
      lũy đẳng, xử lý lỗi và độ trễ qua HTTP.
Dashboard giám sát Streamlit nằm ở dashboard/app.py (đọc trực tiếp các bảng trong results/).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from s2_features import load_campaign  # noqa: E402
from utils import PALETTE, load_config, load_metrics, log, save_fig, save_metrics, save_table, set_seed, setup_mpl  # noqa: E402

warnings.filterwarnings("ignore")


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, (int, float, str, bool)):
            out[key] = v
    return out


# ------------------------------------------------------------------ (A) MLflow
def mlflow_tracking(cfg):
    import mlflow
    from mlflow.tracking import MlflowClient
    M = load_metrics(cfg)
    db = ROOT / "mlops" / "mlflow.db"
    db.parent.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment("x5-uplift-decision-system")
    raw = {k: v for k, v in load_config().items() if k != "paths"}
    params = {k: v for k, v in _flat(raw).items()}
    gates = pd.read_csv(cfg["paths"]["tables"] / "t48_release_gates.csv", encoding="utf-8-sig")
    t28 = pd.read_csv(cfg["paths"]["tables"] / "t28_policy_comparison_locked.csv", encoding="utf-8-sig").set_index("Chính sách")
    t28o = pd.read_csv(cfg["paths"]["tables"] / "t28_policy_comparison_oot.csv", encoding="utf-8-sig").set_index("Chính sách")
    lat = pd.read_csv(cfg["paths"]["tables"] / "t43_latency.csv", encoding="utf-8-sig").set_index("Bước")
    cal = pd.read_csv(cfg["paths"]["tables"] / "t12_calibration_results.csv", encoding="utf-8-sig")
    metrics = {
        "milp_profit_dr_locked": float(t28.loc["MILP (SCIP)", "Lợi nhuận tăng thêm DR (RUB)"]),
        "milp_profit_dr_oot": float(t28o.loc["MILP (SCIP)", "Lợi nhuận tăng thêm DR (RUB)"]),
        "latency_p95_ms": float(lat.loc["Đầu cuối (E2E)", "P95 (ms)"]),
        "release_gates_passed": float(M["mlops"]["gates_pass"]),
    }
    for k in ["auc_locked", "auc_oot", "ece_locked", "ece_oot"]:
        if k in M.get("forecast", {}):
            metrics[k] = float(M["forecast"][k])
    for _, r in cal.iterrows():
        nm = "".join(ch for ch in str(r.iloc[0]) + "_" + str(r.iloc[1]) if ch.isalnum() or ch == "_")[:60]
        for c in ["ECE", "AUC", "Brier"]:
            if c in cal.columns and pd.notna(r[c]):
                metrics[f"cal_{nm}_{c}".lower()] = float(r[c])
    with mlflow.start_run(run_name=f"pipeline-{cfg['model_version']}") as run:
        for i in range(0, len(params), 90):
            mlflow.log_params(dict(list(params.items())[i:i + 90]))
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"campaign_id": cfg["campaign_id"], "model_version": cfg["model_version"],
                         "gates": f"{M['mlops']['gates_pass']}/{M['mlops']['gates_total']}"})
        for f in ["t28_policy_comparison_locked.csv", "t48_release_gates.csv", "t41_artifact_checksums.csv",
                  "t42_version_matrix.csv", "t44_psi_drift.csv"]:
            mlflow.log_artifact(str(cfg["paths"]["tables"] / f), artifact_path="tables")
        for f in sorted(cfg["paths"]["models"].glob("*.joblib")):
            mlflow.log_artifact(str(f), artifact_path="models")
        import mlflow.lightgbm
        import joblib
        fm = joblib.load(cfg["paths"]["models"] / "forecast.joblib")
        info = mlflow.lightgbm.log_model(fm["gbm"], name="forecast_gbm", registered_model_name="x5_forecast_gbm")
        run_id = run.info.run_id
    client = MlflowClient()
    mv = client.get_latest_versions("x5_forecast_gbm")[-1]
    client.set_registered_model_alias("x5_forecast_gbm", "champion", mv.version)
    r = client.get_run(run_id)
    arts = client.list_artifacts(run_id, "tables") + client.list_artifacts(run_id, "models")
    tab = pd.DataFrame([
        {"Hạng mục": "Experiment", "Giá trị": "x5-uplift-decision-system"},
        {"Hạng mục": "Run ID", "Giá trị": run_id[:12]},
        {"Hạng mục": "Số tham số đã ghi", "Giá trị": str(len(r.data.params))},
        {"Hạng mục": "Số chỉ số đã ghi", "Giá trị": str(len(r.data.metrics))},
        {"Hạng mục": "Số tạo tác (bảng, mô hình)", "Giá trị": str(len(arts))},
        {"Hạng mục": "Mô hình đăng ký", "Giá trị": f"x5_forecast_gbm, phiên bản {mv.version}, bí danh champion"},
        {"Hạng mục": "Backend", "Giá trị": "SQLite cục bộ (mlops/mlflow.db)"},
    ])
    save_table(tab, cfg, "tx41_mlflow_run", "Thông tin lần chạy được ghi vào MLflow Tracking và Model Registry")
    return {"run_id": run_id, "n_params": len(r.data.params), "n_metrics": len(r.data.metrics),
            "n_artifacts": len(arts), "model_version": int(mv.version)}


# ------------------------------------------------------------------ (B) Evidently
def evidently_drift(cfg, plt):
    from evidently import Report
    from evidently.presets import DataDriftPreset
    camp = load_campaign(cfg)
    import joblib
    feats = joblib.load(cfg["paths"]["models"] / "uplift_learners.joblib").feats
    rng = np.random.default_rng(cfg["seed"])
    ref = camp[camp.split == "train"][feats].sample(10000, random_state=cfg["seed"]).reset_index(drop=True)
    cur = camp[camp.split == "oot"][feats].sample(10000, random_state=cfg["seed"]).reset_index(drop=True)
    rep = Report([DataDriftPreset()])
    snap = rep.run(cur, ref)
    out = ROOT / "mlops" / "evidently_drift_train_vs_oot.html"
    snap.save_html(str(out))
    d = snap.dict()
    rows = []
    for mt in d.get("metrics", []):
        name = str(mt.get("metric_name", mt.get("metric_id", "")))
        val = mt.get("value")
        if name.startswith("ValueDrift") and isinstance(val, (int, float)):
            import re
            col = re.search(r"column=([^,)]+)", name)
            meth = re.search(r"method=(.+?),threshold=", name) or re.search(r"method=([^,)]+)", name)
            thr = re.search(r"threshold=([0-9.eE-]+)", name)
            rows.append({"Đặc trưng": col.group(1) if col else name, "Phép kiểm định": meth.group(1) if meth else "",
                         "Ngưỡng": thr.group(1) if thr else "", "Giá trị": float(val)})
        if name.startswith("DriftedColumnsCount"):
            drifted = val
    ev = pd.DataFrame(rows)
    # quy ước Evidently: với p-value (K-S, chi-square) trôi dạt khi giá trị < ngưỡng; với khoảng cách (Wasserstein...) khi > ngưỡng
    def drift_flag(r):
        try:
            th = float(r["Ngưỡng"])
        except Exception:
            th = 0.1
        is_p = any(k in r["Phép kiểm định"].lower() for k in ["k-s", "ks", "chi", "z-test", "p_value", "fisher"])
        return r["Giá trị"] < th if is_p else r["Giá trị"] > th
    ev["Trôi dạt (Evidently)"] = ev.apply(drift_flag, axis=1)
    psi = pd.read_csv(cfg["paths"]["tables"] / "t44_psi_drift.csv", encoding="utf-8-sig")
    pcol = [c for c in psi.columns if "OOT" in c or "oot" in c]
    fcol = psi.columns[0]
    if pcol:
        psi = psi[[fcol, pcol[0]]].rename(columns={fcol: "Đặc trưng", pcol[0]: "PSI Train→OOT"})
        ev = ev.merge(psi, on="Đặc trưng", how="left")
        ev["Cảnh báo PSI (≥ 0,10)"] = ev["PSI Train→OOT"] >= cfg["mlops"]["psi_warn"]
    ev = ev.sort_values("Trôi dạt (Evidently)", ascending=False)
    save_table(ev, cfg, "tx42_evidently_drift",
               "Kiểm định trôi dạt dữ liệu Train so với OOT bằng Evidently (DataDriftPreset) và đối chiếu với PSI tự cài đặt")
    n_ev = int(ev["Trôi dạt (Evidently)"].sum())
    agree = None
    if "Cảnh báo PSI (≥ 0,10)" in ev.columns:
        sub = ev.dropna(subset=["PSI Train→OOT"])
        agree = float((sub["Trôi dạt (Evidently)"] == sub["Cảnh báo PSI (≥ 0,10)"]).mean())
    return {"n_features": len(ev), "n_drift_evidently": n_ev, "agreement_with_psi": agree, "html": str(out.relative_to(ROOT))}


# ------------------------------------------------------------------ (C) FastAPI
def fastapi_tests(cfg, plt):
    from fastapi.testclient import TestClient
    from serve.api import Scorer, create_app
    sc = Scorer(cfg)
    client = TestClient(create_app(sc))
    camp = sc.store
    rng = np.random.default_rng(cfg["seed"])
    ids = camp.index[camp.split == "locked"].to_numpy()
    sample = rng.choice(ids, 1000, replace=False)
    # tính đúng: so với chấm điểm theo lô và với điểm V̂ của giai đoạn 5
    batch = sc.score_ids(sample)
    ds = pd.read_parquet(cfg["paths"]["processed"] / "decision_scores.parquet").loc[sample]
    lat, mism_v, mism_a = [], 0, 0
    for cid in sample:
        t0 = time.perf_counter()
        r = client.post("/score", json={"client_id": str(cid)})
        lat.append((time.perf_counter() - t0) * 1000)
        js = r.json()
        if abs(js["value_hat"] - batch.loc[cid, "value_hat"]) > 1e-6:
            mism_v += 1
        if js["action"] != batch.loc[cid, "action"]:
            mism_a += 1
    lat = np.array(lat)
    max_dv = float(np.abs(batch["value_hat"].values - ds["value_hat"].values).max())
    r1 = client.post("/score", json={"client_id": str(sample[0])}).json()
    r2 = client.post("/score", json={"client_id": str(sample[0])}).json()
    nf = client.post("/score", json={"client_id": "khong_ton_tai"}).status_code
    bad = client.post("/score", json={}).status_code
    hb = client.get("/health").json()
    t0 = time.perf_counter(); bres = client.post("/score_batch", json={"client_ids": [str(c) for c in sample]}); tb = time.perf_counter() - t0
    tests = pd.DataFrame([
        {"Kiểm thử": "GET /health trả về phiên bản và mã băm mô hình", "Kết quả": "PASS" if hb.get("status") == "ok" else "FAIL",
         "Chi tiết": f"{hb.get('model_version')}, sha256 {hb.get('uplift_model_sha256')}"},
        {"Kiểm thử": "V̂ qua API trùng chấm điểm theo lô (1.000 khách)", "Kết quả": "PASS" if mism_v == 0 else "FAIL", "Chi tiết": f"{mism_v} khác biệt"},
        {"Kiểm thử": "Hành động qua API trùng chấm điểm theo lô", "Kết quả": "PASS" if mism_a == 0 else "FAIL", "Chi tiết": f"{mism_a} khác biệt"},
        {"Kiểm thử": "V̂ của dịch vụ trùng V̂ của giai đoạn 5", "Kết quả": "PASS" if max_dv < 1e-6 else "FAIL", "Chi tiết": f"sai lệch lớn nhất {max_dv:.2e}"},
        {"Kiểm thử": "Gọi lặp cùng khách cho cùng khóa lũy đẳng", "Kết quả": "PASS" if r1["idempotency_key"] == r2["idempotency_key"] else "FAIL", "Chi tiết": r1["idempotency_key"][:16] + "…"},
        {"Kiểm thử": "client_id không tồn tại trả về 404", "Kết quả": "PASS" if nf == 404 else "FAIL", "Chi tiết": f"HTTP {nf}"},
        {"Kiểm thử": "Yêu cầu thiếu dữ liệu trả về 422", "Kết quả": "PASS" if bad == 422 else "FAIL", "Chi tiết": f"HTTP {bad}"},
        {"Kiểm thử": "POST /score_batch 1.000 khách", "Kết quả": "PASS" if bres.status_code == 200 and len(bres.json()) == 1000 else "FAIL", "Chi tiết": f"{tb * 1000:.0f} ms cho cả lô"},
    ])
    save_table(tests, cfg, "tx43_fastapi_tests", "Kiểm thử dịch vụ chấm điểm FastAPI (TestClient, 1.000 khách Locked Test)")
    lt = pd.DataFrame([{"Đường gọi": "POST /score (từng khách, qua HTTP nội bộ)", "P50 (ms)": np.percentile(lat, 50),
                        "P95 (ms)": np.percentile(lat, 95), "P99 (ms)": np.percentile(lat, 99), "Max (ms)": lat.max()},
                       {"Đường gọi": "POST /score_batch (1.000 khách)", "P50 (ms)": tb * 1000, "P95 (ms)": np.nan, "P99 (ms)": np.nan, "Max (ms)": np.nan}])
    save_table(lt, cfg, "tx44_fastapi_latency", "Độ trễ của dịch vụ FastAPI (bao gồm tuần tự hóa JSON và tính toàn bộ sáu learner)")
    return {"tests_pass": int((tests["Kết quả"] == "PASS").sum()), "tests_total": len(tests),
            "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)), "batch_ms": tb * 1000}


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    m = {}
    log.info("MLflow ..."); m["mlflow"] = mlflow_tracking(cfg)
    log.info("Evidently ..."); m["evidently"] = evidently_drift(cfg, plt)
    log.info("FastAPI ..."); m["fastapi"] = fastapi_tests(cfg, plt)
    save_metrics(cfg, "mlops_tools", m)
    return m


if __name__ == "__main__":
    run(load_config())
