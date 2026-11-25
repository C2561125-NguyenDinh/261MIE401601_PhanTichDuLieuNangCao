"""Dịch vụ chấm điểm và định tuyến hành động (FastAPI).

Chạy:  uvicorn serve.api:app --port 8000
  GET  /health            -> trạng thái, phiên bản mô hình, mã băm mô hình
  POST /score             -> {"client_id": "..."}  hoặc {"features": {...}}: trả về xác suất mua, chi tiêu kỳ vọng,
                             uplift, giá trị V̂, hành động, mã lý do và khóa lũy đẳng
  POST /score_batch       -> {"client_ids": [...]}
Quyết định trực tuyến dùng giá đối ngẫu λ của ràng buộc ngân sách từ lần giải MILP gần nhất:
gửi khi V̂ − λ·c > 0 (tương đương nghiệm MILP khi chỉ ngân sách là ràng buộc chặt, mục 4.5.3).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from s2_features import feature_cols, load_campaign  # noqa: E402
from s7_router import ActionRouter, idem_key  # noqa: E402
from utils import load_config, load_metrics, sha256_file  # noqa: E402

LEARNERS = ["S-learner", "T-learner", "X-learner", "R-learner", "DR-learner", "Class Transformation"]


class Scorer:
    def __init__(self, cfg=None):
        self.cfg = cfg or load_config()
        md, proc = self.cfg["paths"]["models"], self.cfg["paths"]["processed"]
        self.M = load_metrics(self.cfg)
        self.fm = joblib.load(md / "forecast.joblib")
        self.L = joblib.load(md / "uplift_learners.joblib")
        self.sp = joblib.load(md / "spo.joblib")
        self.best = self.M["causal"]["best_learner"]
        camp = load_campaign(self.cfg)
        self.store = camp
        fc = pd.read_parquet(proc / "forecast_campaign.parquet")
        self.spend_cap = float(fc["spend_hat"].quantile(self.cfg["business"]["aov_cap_quantile"]))
        duals = pd.read_csv(self.cfg["paths"]["tables"] / "t29_shadow_prices.csv", encoding="utf-8-sig")
        self.lam = float(duals.set_index(duals.columns[0]).iloc[:, 0].get("budget", 0.0))
        self.router = ActionRouter(self.cfg, self.M["router"]["thresholds"])
        self.model_hash = sha256_file(md / "uplift_learners.joblib")[:12]

    def score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        b = self.cfg["business"]
        F = self.fm["features"]
        p_raw = self.fm["gbm"].predict_proba(X[F])[:, 1]
        p_cal = self.fm["calibrator"].predict(p_raw)
        spend = np.expm1(self.fm["mean_model"].predict(X[F]))
        P = self.L.predict_all(X)
        tau, tau_std = P[self.best].values, P[LEARNERS].std(axis=1).values
        p1 = np.clip(P["_mu0"].values + tau, 0, 1)
        base_cols = [c for c in self.sp["cols"] if c not in ("spend_hat", "p1_hat", "p_buy_cal")]
        Xd = X[base_cols].copy()
        Xd["spend_hat"], Xd["p1_hat"], Xd["p_buy_cal"] = spend, p1, p_cal
        Xd = Xd[self.sp["cols"]].fillna(self.sp["med"])
        Xs = (np.sign(Xd) * np.log1p(np.abs(Xd)) - self.sp["mu"]) / self.sp["sd"]
        v_hat = -(np.c_[Xs.values, np.ones(len(Xs))] @ self.sp["theta"])
        cost = b["contact_cost"] + p1 * b["incentive_cost"]
        x = (v_hat - self.lam * cost > 0).astype(int)
        df = pd.DataFrame({"tau_hat": tau, "tau_std": tau_std, "x": x, "recency": X["recency"].values,
                           "sum_all": X["sum_all"].values, "has_tx": X["has_tx"].values}, index=X.index)
        routed = self.router.route(df)
        date = self.M["features"]["cutoffs"]["DECISION"]
        return pd.DataFrame({"p_buy": p_cal, "spend_hat": spend, "tau_hat": tau, "tau_std": tau_std,
                             "value_hat": v_hat, "cost": cost, "selected": x, "action": routed["action"].values,
                             "reasons": routed["reasons"].values,
                             "idempotency_key": [idem_key(self.cfg["campaign_id"], c, self.cfg["model_version"], date)
                                                 for c in X.index]}, index=X.index)

    def score_ids(self, ids):
        X = self.store.loc[list(ids)]
        return self.score_frame(X)


from pydantic import BaseModel  # noqa: E402


class One(BaseModel):
    client_id: str | None = None
    features: dict | None = None


class Batch(BaseModel):
    client_ids: list[str]


def create_app(scorer: Scorer | None = None):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="X5 uplift decision service", version="1.0.0")
    state = {"scorer": scorer}

    def sc():
        if state["scorer"] is None:
            state["scorer"] = Scorer()
        return state["scorer"]

    @app.get("/health")
    def health():
        s = sc()
        return {"status": "ok", "model_version": s.cfg["model_version"], "uplift_model_sha256": s.model_hash,
                "best_learner": s.best, "budget_dual_lambda": s.lam}

    @app.post("/score")
    def score(req: One):
        s = sc()
        if req.client_id is not None:
            if req.client_id not in s.store.index:
                raise HTTPException(404, "client_id không có trong feature store")
            r = s.score_ids([req.client_id]).iloc[0]
        elif req.features is not None:
            r = s.score_frame(pd.DataFrame([req.features], index=["adhoc"])).iloc[0]
        else:
            raise HTTPException(422, "cần client_id hoặc features")
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.to_dict().items()}

    @app.post("/score_batch")
    def score_batch(req: Batch):
        s = sc()
        ids = [i for i in req.client_ids if i in s.store.index]
        out = s.score_ids(ids).reset_index().rename(columns={"index": "client_id"})
        return out.to_dict(orient="records")

    return app


app = create_app()  # mô hình được nạp ở lần gọi đầu tiên
