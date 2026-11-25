"""Giai đoạn 3 - Dự báo nâng cao, hiệu chỉnh xác suất và định lượng bất định.

(a) Xác suất mua hàng trong H ngày tới (phân loại) -> hiệu chỉnh Platt / Isotonic, ECE, Brier.
(b) Chi tiêu kỳ vọng khi mua (hồi quy) -> Conformalized Quantile Regression (CQR), độ phủ khoảng.
Kết quả (b) đi vào hàm giá trị của bài toán tối ưu.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, mean_absolute_error,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from s2_features import feature_cols, load_campaign, load_snapshots
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl


# ------------------------------------------------------------------ metrics
def ece_score(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        gap = abs(y[mask].mean() - p[mask].mean())
        ece += mask.mean() * gap
        mce = max(mce, gap)
    return ece, mce


def cls_metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ece, mce = ece_score(y, p)
    return {"AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p), "LogLoss": log_loss(y, p),
            "Brier": brier_score_loss(y, p), "ECE": ece, "MCE": mce, "mean_p": p.mean(), "base_rate": y.mean()}


def _log1p_pos(X):
    return np.sign(X) * np.log1p(np.abs(X))


class Calibrator:
    def __init__(self, method):
        self.method = method

    def fit(self, p, y):
        if self.method == "platt":
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            self.m = LogisticRegression(C=1e6).fit(z, y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        return self

    def predict(self, p):
        if self.method == "none":
            return p
        if self.method == "platt":
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            return self.m.predict_proba(z)[:, 1]
        return self.m.predict(p)


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    fc = load_snapshots(cfg)
    feats = feature_cols(fc)
    tr = fc[(fc.snapshot == "S1") & (fc.split == "train")]
    va = fc[(fc.snapshot == "S2") & (fc.split == "val")]
    lk = fc[(fc.snapshot == "S2") & (fc.split == "locked")]
    oot = fc[(fc.snapshot == "S3") & (fc.split.isin(["locked", "oot"]))]
    sets = {"Validation (S2)": va, "Locked Test (S2)": lk, "Out-of-Time (S3)": oot}
    log.info(f"Dự báo: train={len(tr)}, val={len(va)}, locked={len(lk)}, oot={len(oot)}")
    m: dict = {"n": {"train": len(tr), "val": len(va), "locked": len(lk), "oot": len(oot)}}

    # ---------------- mô hình
    base_feats = ["recency", "n_tx_14d", "n_tx_all"]
    models = {}
    models["Baseline (Logit-RFM)"] = make_pipeline(FunctionTransformer(_log1p_pos), StandardScaler(),
                                                   LogisticRegression(max_iter=500)).fit(
        tr[base_feats].fillna(0), tr["y_buy"])
    models["Logistic (đầy đủ)"] = make_pipeline(FunctionTransformer(_log1p_pos), StandardScaler(),
                                                LogisticRegression(max_iter=1000, C=0.5)).fit(
        tr[feats].fillna(0), tr["y_buy"])
    gbm = lgb.LGBMClassifier(**cfg["lgbm"], random_state=cfg["seed"])
    gbm.fit(tr[feats], tr["y_buy"], eval_set=[(va[feats], va["y_buy"])],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    models["LightGBM"] = gbm
    m["lgbm_best_iter"] = int(gbm.best_iteration_ or cfg["lgbm"]["n_estimators"])

    def predict(name, df):
        mdl = models[name]
        X = df[base_feats].fillna(0) if name.startswith("Baseline") else (df[feats].fillna(0) if name.startswith("Logistic") else df[feats])
        return mdl.predict_proba(X)[:, 1]

    rows = []
    for name in models:
        for sname, df in sets.items():
            r = cls_metrics(df["y_buy"].values, predict(name, df))
            rows.append({"Mô hình": name, "Tập": sname, **r})
    t = pd.DataFrame(rows)
    save_table(t, cfg, "t10_forecast_model_comparison", "So sánh mô hình dự báo xác suất mua hàng (chưa hiệu chỉnh)")
    m["model_comparison"] = t.to_dict(orient="records")

    # ---------------- hiệu chỉnh: chọn phương pháp bằng 2-fold trên Validation
    p_va = predict("LightGBM", va)
    y_va = va["y_buy"].values
    rng = np.random.default_rng(cfg["seed"])
    fold = rng.integers(0, 2, len(va))
    sel = []
    for meth in ["none", "platt", "isotonic"]:
        briers, eces = [], []
        for k in [0, 1]:
            c = Calibrator(meth).fit(p_va[fold != k], y_va[fold != k]) if meth != "none" else Calibrator("none")
            pk = c.predict(p_va[fold == k])
            briers.append(brier_score_loss(y_va[fold == k], pk))
            eces.append(ece_score(y_va[fold == k], pk)[0])
        sel.append({"Phương pháp": meth, "Brier (CV 2-fold)": np.mean(briers), "ECE (CV 2-fold)": np.mean(eces)})
    sel = pd.DataFrame(sel)
    best = sel.sort_values("ECE (CV 2-fold)").iloc[0]["Phương pháp"]
    save_table(sel, cfg, "t11_calibration_selection", "Lựa chọn phương pháp hiệu chỉnh bằng kiểm định chéo 2-fold trên tập Validation")
    m["calibration_selected"] = best
    cals = {meth: (Calibrator(meth).fit(p_va, y_va) if meth != "none" else Calibrator("none"))
            for meth in ["none", "platt", "isotonic"]}

    rows = []
    for meth, c in cals.items():
        for sname, df in sets.items():
            if sname.startswith("Validation"):
                continue
            r = cls_metrics(df["y_buy"].values, c.predict(predict("LightGBM", df)))
            rows.append({"Hiệu chỉnh": meth, "Tập": sname, **r})
    tc = pd.DataFrame(rows)
    save_table(tc, cfg, "t12_calibration_results", "Hiệu năng LightGBM trước và sau hiệu chỉnh trên Locked Test và Out-of-Time")
    m["calibration_results"] = tc.to_dict(orient="records")

    # ---------------- hình: reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (sname, df) in zip(axes, [("Locked Test (S2)", lk), ("Out-of-Time (S3)", oot)]):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for i, (meth, c) in enumerate(cals.items()):
            p = c.predict(predict("LightGBM", df))
            y = df["y_buy"].values
            q = pd.qcut(p, 15, duplicates="drop")
            g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).mean()
            ax.plot(g["p"], g["y"], "o-", color=PALETTE[i], label=f"{meth} (ECE={ece_score(y, p)[0]:.4f})")
        ax.set_xlabel("Xác suất dự báo"); ax.set_ylabel("Tần suất quan sát"); ax.set_title(sname); ax.legend()
    save_fig(fig, cfg, "f10_reliability", "Biểu đồ độ tin cậy (reliability diagram) trước và sau hiệu chỉnh")

    # histogram xác suất
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (meth, c) in enumerate(cals.items()):
        ax.hist(c.predict(predict("LightGBM", lk)), bins=60, alpha=.45, color=PALETTE[i], label=meth, density=True)
    ax.set_title("Phân phối xác suất dự báo trên Locked Test"); ax.legend()
    save_fig(fig, cfg, "f11_prob_hist", "Phân phối xác suất dự báo theo phương pháp hiệu chỉnh")

    # ROC
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, name in enumerate(models):
        fpr, tpr, _ = roc_curve(oot["y_buy"], predict(name, oot))
        ax.plot(fpr, tpr, color=PALETTE[i], label=f"{name} (AUC={roc_auc_score(oot['y_buy'], predict(name, oot)):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()
    ax.set_title("Đường ROC trên Out-of-Time (S3)")
    save_fig(fig, cfg, "f12_roc_oot", "Đường ROC của các mô hình dự báo trên tập Out-of-Time")

    # tầm quan trọng đặc trưng
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=feats).sort_values()
    imp = imp / imp.sum()
    fig, ax = plt.subplots(figsize=(7, 8))
    imp.tail(25).plot.barh(ax=ax, color=PALETTE[0]); ax.set_title("Tầm quan trọng đặc trưng (gain) - LightGBM")
    save_fig(fig, cfg, "f13_forecast_importance", "Tầm quan trọng đặc trưng của mô hình dự báo mua hàng")
    save_table(imp.sort_values(ascending=False).rename("gain_share").reset_index().rename(columns={"index": "Đặc trưng"}),
               cfg, "t13_forecast_importance", "Tỷ trọng gain của các đặc trưng trong mô hình dự báo")

    # ---------------- giá trị quyết định: net benefit (decision curve)
    best_cal = cals[best]
    p_lk = best_cal.predict(predict("LightGBM", lk))
    y_lk = lk["y_buy"].values
    ths = np.linspace(0.05, 0.95, 19)
    rows = []
    for th in ths:
        sel_ = p_lk >= th
        tp, fp = (sel_ & (y_lk == 1)).sum(), (sel_ & (y_lk == 0)).sum()
        n = len(y_lk)
        nb = tp / n - fp / n * th / (1 - th)
        nb_all = y_lk.mean() - (1 - y_lk.mean()) * th / (1 - th)
        rows.append({"Ngưỡng": th, "Net benefit mô hình": nb, "Net benefit 'chọn tất cả'": nb_all, "Tỷ lệ được chọn": sel_.mean()})
    dc = pd.DataFrame(rows)
    save_table(dc, cfg, "t14_decision_curve", "Phân tích đường cong quyết định (net benefit) trên Locked Test")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(dc["Ngưỡng"], dc["Net benefit mô hình"], color=PALETTE[0], label="Mô hình đã hiệu chỉnh")
    ax.plot(dc["Ngưỡng"], dc["Net benefit 'chọn tất cả'"], color=PALETTE[1], label="Chọn tất cả")
    ax.axhline(0, color="k", lw=1, label="Không chọn ai")
    ax.set_ylim(-0.05, max(dc["Net benefit mô hình"].max(), 0.05) * 1.15)
    ax.set_xlabel("Ngưỡng xác suất"); ax.set_ylabel("Net benefit"); ax.legend(); ax.set_title("Đường cong quyết định")
    save_fig(fig, cfg, "f14_decision_curve", "Đường cong quyết định (net benefit) của mô hình dự báo")

    # ---------------- (b) chi tiêu khi mua: CQR
    alpha = cfg["conformal"]["alpha"]
    trb, vab = tr[tr.y_buy == 1], va[va.y_buy == 1]
    lkb, ootb = lk[lk.y_buy == 1], oot[oot.y_buy == 1]
    ylog = lambda d: np.log1p(d["y_spend"].clip(lower=0).values)
    params = {k: v for k, v in cfg["lgbm"].items()}
    q_models = {}
    for q in [alpha / 2, 0.5, 1 - alpha / 2]:
        mdl = lgb.LGBMRegressor(objective="quantile", alpha=q, **params, random_state=cfg["seed"])
        mdl.fit(trb[feats], ylog(trb))
        q_models[q] = mdl
    mean_model = lgb.LGBMRegressor(**params, random_state=cfg["seed"]).fit(trb[feats], ylog(trb))
    lo_v, hi_v = q_models[alpha / 2].predict(vab[feats]), q_models[1 - alpha / 2].predict(vab[feats])
    scores = np.maximum(lo_v - ylog(vab), ylog(vab) - hi_v)
    n = len(scores)
    qhat = np.quantile(scores, min(1, np.ceil((n + 1) * (1 - alpha)) / n))
    m["cqr_qhat"] = qhat
    rows = []
    for sname, d in [("Locked Test (S2)", lkb), ("Out-of-Time (S3)", ootb)]:
        y = ylog(d)
        lo_raw, hi_raw = q_models[alpha / 2].predict(d[feats]), q_models[1 - alpha / 2].predict(d[feats])
        lo, hi = lo_raw - qhat, hi_raw + qhat
        pred = np.expm1(mean_model.predict(d[feats]))
        for meth, (a, b) in {"Quantile thô": (lo_raw, hi_raw), "CQR": (lo, hi)}.items():
            rows.append({"Tập": sname, "Phương pháp": meth, "Độ phủ": np.mean((y >= a) & (y <= b)),
                         "Độ rộng TB (RUB)": np.mean(np.expm1(b) - np.expm1(a)),
                         "MAE điểm (RUB)": mean_absolute_error(d["y_spend"], pred),
                         "MAE baseline AOV (RUB)": mean_absolute_error(d["y_spend"], d["aov_all"].fillna(d["aov_all"].median()) * np.maximum(1, d["n_tx_14d"])),
                         "Độ phủ mục tiêu": 1 - alpha})
    tq = pd.DataFrame(rows)
    save_table(tq, cfg, "t15_cqr_results", "Định lượng bất định chi tiêu bằng Conformalized Quantile Regression")
    m["cqr_results"] = tq.to_dict(orient="records")

    # độ phủ theo thập phân vị dự báo
    y = ylog(lkb)
    lo = q_models[alpha / 2].predict(lkb[feats]) - qhat
    hi = q_models[1 - alpha / 2].predict(lkb[feats]) + qhat
    med = q_models[0.5].predict(lkb[feats])
    dec = pd.qcut(med, 10, labels=False, duplicates="drop")
    cov = pd.DataFrame({"dec": dec, "cov": (y >= lo) & (y <= hi), "w": np.expm1(hi) - np.expm1(lo)}).groupby("dec").mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(cov.index + 1, cov["cov"], color=PALETTE[2]); ax[0].axhline(1 - alpha, color="k", ls="--")
    ax[0].set_ylim(0.5, 1); ax[0].set_title("Độ phủ CQR theo thập phân vị dự báo"); ax[0].set_xlabel("Thập phân vị")
    ax[1].bar(cov.index + 1, cov["w"], color=PALETTE[3]); ax[1].set_title("Độ rộng khoảng (RUB)"); ax[1].set_xlabel("Thập phân vị")
    save_fig(fig, cfg, "f15_cqr_coverage", "Độ phủ và độ rộng khoảng dự báo CQR theo thập phân vị")
    order = np.argsort(med)[:: max(1, len(med) // 300)]
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(order))
    ax.fill_between(xs, np.expm1(lo[order]), np.expm1(hi[order]), color=PALETTE[0], alpha=.25, label="Khoảng CQR 90%")
    ax.plot(xs, np.expm1(med[order]), color=PALETTE[0], label="Trung vị dự báo")
    ax.scatter(xs, np.expm1(y[order]), s=6, color=PALETTE[1], label="Thực tế")
    ax.set_yscale("log"); ax.set_xlabel("Khách hàng (sắp theo dự báo)"); ax.set_ylabel(f"Chi tiêu {cfg['temporal']['horizon_days']} ngày (RUB)"); ax.legend()
    ax.set_title("Khoảng dự báo chi tiêu có bảo đảm độ phủ (mẫu Locked Test)")
    save_fig(fig, cfg, "f16_cqr_intervals", "Khoảng dự báo chi tiêu CQR so với giá trị thực tế")

    # ---------------- chấm điểm tập chiến dịch tại thời điểm quyết định
    camp = load_campaign(cfg)
    out = pd.DataFrame(index=camp.index)
    out["p_buy_raw"] = gbm.predict_proba(camp[feats])[:, 1]
    out["p_buy_cal"] = best_cal.predict(out["p_buy_raw"].values)
    out["spend_hat"] = np.expm1(mean_model.predict(camp[feats]))
    out["spend_lo"] = np.expm1(q_models[alpha / 2].predict(camp[feats]) - qhat)
    out["spend_hi"] = np.expm1(q_models[1 - alpha / 2].predict(camp[feats]) + qhat)
    out.to_parquet(cfg["paths"]["processed"] / "forecast_campaign.parquet")
    m["campaign_p_buy_mean"] = out["p_buy_cal"].mean()
    m["campaign_spend_hat_mean"] = out["spend_hat"].mean()
    # liên hệ giữa xác suất mua dự báo và target chiến dịch ở nhóm đối chứng
    ctrl = camp.treatment_flg == 0
    m["campaign_ctrl_auc_vs_target"] = roc_auc_score(camp.loc[ctrl, "target"], out.loc[ctrl, "p_buy_cal"])

    joblib.dump({"gbm": gbm, "calibrator": best_cal, "q_models": q_models, "mean_model": mean_model,
                 "qhat": qhat, "features": feats}, cfg["paths"]["models"] / "forecast.joblib")
    save_metrics(cfg, "forecast", m)
    return m


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
