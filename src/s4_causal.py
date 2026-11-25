"""Giai đoạn 4 - Phân tích nhân quả và mô hình hóa tác động biên (Uplift Modeling).

- Kiểm tra cân bằng ngẫu nhiên hóa (SMD).
- ATE / ATT / ATC: chênh lệch trung bình (Neyman), AIPW cross-fitting, Double ML (DoubleML IRM).
- Kiểm định phản bác: placebo treatment, biến gây nhiễu ngẫu nhiên, ổn định mẫu con.
- CATE: S-, T-, X-, R-, DR-learner và Class Transformation; đánh giá Qini / AUUC / uplift@k ngoài mẫu.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from s2_features import feature_cols, load_campaign
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ uplift metrics
def qini_curve(y, t, u):
    """Đường Qini (Radcliffe): Q(k) = Y_T(k) - Y_C(k) * N_T(k)/N_C(k)."""
    o = np.argsort(-u, kind="mergesort")
    y, t = y[o], t[o]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt - np.where(nc > 0, yc * nt / nc, 0)
    x = np.arange(1, len(y) + 1)
    return np.r_[0, x], np.r_[0, q]


def uplift_curve(y, t, u):
    o = np.argsort(-u, kind="mergesort")
    y, t = y[o], t[o]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        up = (np.where(nt > 0, yt / nt, 0) - np.where(nc > 0, yc / nc, 0)) * (nt + nc)
    x = np.arange(1, len(y) + 1)
    return np.r_[0, x], np.r_[0, up]


def qini_auc(y, t, u, normalize=True):
    """Diện tích giữa đường Qini và đường ngẫu nhiên; chuẩn hóa theo đường Qini lý tưởng nếu normalize."""
    x, q = qini_curve(y, t, u)
    rand = x * q[-1] / x[-1]
    area = np.trapezoid(q - rand, x)
    if not normalize:
        return area / (x[-1] ** 2)
    # đường lý tưởng: đặt các phản hồi can thiệp lên đầu, phản hồi đối chứng xuống cuối
    perfect_u = y * t - y * (1 - t)
    xp, qp = qini_curve(y, t, perfect_u.astype(float) + 1e-9 * np.random.default_rng(0).random(len(y)))
    area_p = np.trapezoid(qp - xp * qp[-1] / xp[-1], xp)
    return area / area_p if area_p != 0 else np.nan


def auuc(y, t, u):
    x, up = uplift_curve(y, t, u)
    rand = x * up[-1] / x[-1]
    return np.trapezoid(up - rand, x) / (x[-1] ** 2)


def uplift_at_k(y, t, u, k):
    n = int(len(u) * k)
    o = np.argsort(-u)[:n]
    yt, tt = y[o], t[o]
    if tt.sum() == 0 or (1 - tt).sum() == 0:
        return np.nan
    return yt[tt == 1].mean() - yt[tt == 0].mean()


def eval_uplift(y, t, u):
    return {"Qini_norm": qini_auc(y, t, u), "Qini_area": qini_auc(y, t, u, normalize=False),
            "AUUC": auuc(y, t, u), "Uplift@10%": uplift_at_k(y, t, u, .1),
            "Uplift@20%": uplift_at_k(y, t, u, .2), "Uplift@30%": uplift_at_k(y, t, u, .3)}


# ------------------------------------------------------------------ learners
def lgb_clf(cfg, **kw):
    p = {**cfg["lgbm"], **kw}
    return lgb.LGBMClassifier(**p, random_state=cfg["seed"])


def lgb_reg(cfg, **kw):
    p = {**cfg["lgbm"], **kw}
    return lgb.LGBMRegressor(**p, random_state=cfg["seed"])


class UpliftLearners:
    """Tập hợp các meta-learner, cùng huấn luyện trên tập train."""

    def __init__(self, cfg, feats, e):
        self.cfg, self.feats, self.e = cfg, feats, e
        self.slow = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500)

    def fit(self, X, y, t):
        cfg, F = self.cfg, self.feats
        Xt = X[F].copy()
        # S-learner
        Xs = Xt.assign(_T=t)
        self.s = lgb_clf(cfg).fit(Xs, y)
        # T-learner
        self.m1 = lgb_clf(cfg).fit(Xt[t == 1], y[t == 1])
        self.m0 = lgb_clf(cfg).fit(Xt[t == 0], y[t == 0])
        # X-learner
        d1 = y[t == 1] - self.m0.predict_proba(Xt[t == 1])[:, 1]
        d0 = self.m1.predict_proba(Xt[t == 0])[:, 1] - y[t == 0]
        self.x1 = lgb_reg(cfg, **self.slow).fit(Xt[t == 1], d1)
        self.x0 = lgb_reg(cfg, **self.slow).fit(Xt[t == 0], d0)
        # cross-fitting cho R- và DR-learner
        kf = KFold(self.cfg["causal"]["n_folds"], shuffle=True, random_state=cfg["seed"])
        mu0, mu1, mhat = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
        for tr, te in kf.split(Xt):
            a = lgb_clf(cfg).fit(Xt.iloc[tr][t[tr] == 1], y[tr][t[tr] == 1])
            b = lgb_clf(cfg).fit(Xt.iloc[tr][t[tr] == 0], y[tr][t[tr] == 0])
            c = lgb_clf(cfg).fit(Xt.iloc[tr], y[tr])
            mu1[te] = a.predict_proba(Xt.iloc[te])[:, 1]
            mu0[te] = b.predict_proba(Xt.iloc[te])[:, 1]
            mhat[te] = c.predict_proba(Xt.iloc[te])[:, 1]
        e = self.e
        psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self.psi_train = psi
        self.dr = lgb_reg(cfg, **self.slow).fit(Xt, psi)
        rt = t - e
        self.r = lgb_reg(cfg, **self.slow).fit(Xt, (y - mhat) / rt, sample_weight=rt ** 2)
        # Class Transformation (Jaskowski & Jaroszewicz)
        z = (y * t + (1 - y) * (1 - t)).astype(int)
        self.ct = lgb_clf(cfg).fit(Xt, z)
        return self

    def predict_all(self, X):
        Xt = X[self.feats]
        p1s = self.s.predict_proba(Xt.assign(_T=1))[:, 1]
        p0s = self.s.predict_proba(Xt.assign(_T=0))[:, 1]
        mu1, mu0 = self.m1.predict_proba(Xt)[:, 1], self.m0.predict_proba(Xt)[:, 1]
        e = self.e
        out = pd.DataFrame(index=X.index)
        out["S-learner"] = p1s - p0s
        out["T-learner"] = mu1 - mu0
        out["X-learner"] = e * self.x0.predict(Xt) + (1 - e) * self.x1.predict(Xt)
        out["R-learner"] = self.r.predict(Xt)
        out["DR-learner"] = self.dr.predict(Xt)
        out["Class Transformation"] = 2 * self.ct.predict_proba(Xt)[:, 1] - 1
        out["_mu0"], out["_mu1"] = mu0, mu1
        return out


# ------------------------------------------------------------------ ATE
def neyman(y, t):
    a, b = y[t == 1], y[t == 0]
    ate = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return ate, se


def aipw(cfg, X, y, t, feats, e_known=None):
    kf = KFold(cfg["causal"]["n_folds"], shuffle=True, random_state=cfg["seed"])
    mu0, mu1, eh = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for tr, te in kf.split(X):
        Xa, Xb = X.iloc[tr][feats], X.iloc[te][feats]
        mu1[te] = lgb_clf(cfg).fit(Xa[t[tr] == 1], y[tr][t[tr] == 1]).predict_proba(Xb)[:, 1]
        mu0[te] = lgb_clf(cfg).fit(Xa[t[tr] == 0], y[tr][t[tr] == 0]).predict_proba(Xb)[:, 1]
        eh[te] = lgb_clf(cfg, n_estimators=150).fit(Xa, t[tr]).predict_proba(Xb)[:, 1]
    clip = cfg["causal"]["propensity_clip"]
    eh = np.clip(eh, clip, 1 - clip)
    psi = mu1 - mu0 + t * (y - mu1) / eh - (1 - t) * (y - mu0) / (1 - eh)
    ate, se = psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))
    p = t.mean()
    att_psi = (t * (y - mu0) - (1 - t) * eh / (1 - eh) * (y - mu0)) / p
    atc_psi = (t * (1 - eh) / eh * (y - mu1) - (1 - t) * (y - mu1)) / (1 - p) * -1
    atc_psi = ((1 - t) * (mu1 - y) + t * (1 - eh) / eh * (y - mu1)) / (1 - p)
    return {"ATE": (ate, se), "ATT": (att_psi.mean(), att_psi.std(ddof=1) / np.sqrt(len(y))),
            "ATC": (atc_psi.mean(), atc_psi.std(ddof=1) / np.sqrt(len(y))), "e_hat": eh}


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    camp = load_campaign(cfg)
    feats = feature_cols(camp)
    y_all, t_all = camp["target"].values.astype(float), camp["treatment_flg"].values.astype(float)
    m: dict = {}

    # ---------------- cân bằng ngẫu nhiên hóa
    rows = []
    for c in feats:
        a, b = camp.loc[camp.treatment_flg == 1, c], camp.loc[camp.treatment_flg == 0, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"Đặc trưng": c, "TB can thiệp": a.mean(), "TB đối chứng": b.mean(),
                     "SMD": (a.mean() - b.mean()) / sd if sd > 0 else 0})
    bal = pd.DataFrame(rows)
    bal["|SMD|"] = bal["SMD"].abs()
    save_table(bal.sort_values("|SMD|", ascending=False), cfg, "t16_balance_smd",
               "Kiểm tra cân bằng hiệp biến giữa nhóm can thiệp và đối chứng (SMD)")
    m["max_abs_smd"] = bal["|SMD|"].max()
    m["n_smd_gt_01"] = int((bal["|SMD|"] > 0.1).sum())
    fig, ax = plt.subplots(figsize=(6.5, 9))
    b2 = bal.sort_values("|SMD|")
    ax.scatter(b2["SMD"], range(len(b2)), color=PALETTE[0])
    ax.set_yticks(range(len(b2))); ax.set_yticklabels(b2["Đặc trưng"], fontsize=7)
    ax.axvline(0.1, color="r", ls="--"); ax.axvline(-0.1, color="r", ls="--"); ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Chênh lệch trung bình chuẩn hóa (SMD)"); ax.set_title("Love plot - cân bằng ngẫu nhiên hóa")
    save_fig(fig, cfg, "f17_love_plot", "Love plot: SMD của các hiệp biến giữa hai nhóm")

    # ---------------- ATE / ATT / ATC
    dev = camp[camp.split.isin(["train", "val"])]
    yd, td = dev["target"].values.astype(float), dev["treatment_flg"].values.astype(float)
    rows = []
    for sname, d in [("Toàn bộ", camp), ("Train+Val", dev), ("Locked Test", camp[camp.split == "locked"]),
                     ("Out-of-Time", camp[camp.split == "oot"])]:
        a, se = neyman(d["target"].values.astype(float), d["treatment_flg"].values.astype(float))
        rows.append({"Ước lượng": "Chênh lệch trung bình (Neyman)", "Tập": sname, "Tham số": "ATE",
                     "Giá trị": a, "SE": se, "KTC95_dưới": a - 1.96 * se, "KTC95_trên": a + 1.96 * se, "n": len(d)})
    log.info("AIPW cross-fitting ...")
    ai = aipw(cfg, dev, yd, td, feats)
    for k in ["ATE", "ATT", "ATC"]:
        a, se = ai[k]
        rows.append({"Ước lượng": "AIPW (cross-fitting, LightGBM)", "Tập": "Train+Val", "Tham số": k,
                     "Giá trị": a, "SE": se, "KTC95_dưới": a - 1.96 * se, "KTC95_trên": a + 1.96 * se, "n": len(dev)})
    try:
        import doubleml as dml
        log.info("DoubleML IRM ...")
        X = dev[feats].fillna(dev[feats].median()).values
        for score in ["ATE", "ATTE"]:
            data = dml.DoubleMLData.from_arrays(X, yd, td)
            irm = dml.DoubleMLIRM(data, ml_g=lgb_clf(cfg, n_estimators=200), ml_m=lgb_clf(cfg, n_estimators=150),
                                  n_folds=cfg["causal"]["n_folds"], score=score, trimming_threshold=cfg["causal"]["propensity_clip"])
            irm.fit()
            a, se = float(irm.coef[0]), float(irm.se[0])
            rows.append({"Ước lượng": "Double ML (DoubleML IRM)", "Tập": "Train+Val",
                         "Tham số": "ATE" if score == "ATE" else "ATT", "Giá trị": a, "SE": se,
                         "KTC95_dưới": a - 1.96 * se, "KTC95_trên": a + 1.96 * se, "n": len(dev)})
    except Exception as ex:  # pragma: no cover
        log.warning(f"DoubleML lỗi: {ex}")
    ate_t = pd.DataFrame(rows)
    save_table(ate_t, cfg, "t17_ate_estimates", "Ước lượng tác động trung bình ATE / ATT / ATC bằng nhiều phương pháp")
    m["ate_table"] = ate_t.to_dict(orient="records")
    m["propensity_hat_mean"] = float(ai["e_hat"].mean())
    m["propensity_hat_std"] = float(ai["e_hat"].std())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    lab = ate_t["Ước lượng"].str.replace(" (cross-fitting, LightGBM)", "") + " | " + ate_t["Tập"] + " | " + ate_t["Tham số"]
    ax.errorbar(ate_t["Giá trị"], range(len(ate_t)), xerr=1.96 * ate_t["SE"], fmt="o", color=PALETTE[0], capsize=4)
    ax.set_yticks(range(len(ate_t))); ax.set_yticklabels(lab, fontsize=8); ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Tác động lên xác suất mua (điểm xác suất)"); ax.set_title("Forest plot ước lượng tác động trung bình")
    save_fig(fig, cfg, "f18_ate_forest", "Forest plot các ước lượng ATE/ATT/ATC với KTC 95%")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(ai["e_hat"], bins=60, color=PALETTE[3])
    ax.axvline(td.mean(), color="k", ls="--", label=f"Tỷ lệ can thiệp thực = {td.mean():.3f}")
    ax.set_title("Phân phối điểm xu hướng ước lượng (overlap)"); ax.legend()
    save_fig(fig, cfg, "f19_propensity_overlap", "Phân phối điểm xu hướng ước lượng - kiểm tra overlap")

    # ---------------- phản bác
    rng = np.random.default_rng(cfg["seed"])
    ref = []
    base_ate, base_se = neyman(yd, td)
    plac = [neyman(yd, rng.permutation(td))[0] for _ in range(cfg["causal"]["n_bootstrap"])]
    ref.append({"Kiểm định": "Placebo treatment (hoán vị T)", "Kỳ vọng": "≈ 0",
                "Kết quả": np.mean(plac), "Độ lệch chuẩn": np.std(plac),
                "p-value (2 phía)": np.mean(np.abs(plac) >= abs(base_ate))})
    devr = dev.copy(); devr["_rand_cc"] = rng.normal(size=len(devr))
    a2 = aipw(cfg, devr, yd, td, feats + ["_rand_cc"])["ATE"]
    ref.append({"Kiểm định": "Thêm biến gây nhiễu ngẫu nhiên (AIPW)", "Kỳ vọng": f"≈ {ai['ATE'][0]:.4f}",
                "Kết quả": a2[0], "Độ lệch chuẩn": a2[1], "p-value (2 phía)": np.nan})
    subs = []
    for _ in range(cfg["causal"]["n_bootstrap"]):
        idx = rng.choice(len(yd), len(yd) // 2, replace=False)
        subs.append(neyman(yd[idx], td[idx])[0])
    ref.append({"Kiểm định": "Ổn định mẫu con 50% (Neyman)", "Kỳ vọng": f"≈ {base_ate:.4f}",
                "Kết quả": np.mean(subs), "Độ lệch chuẩn": np.std(subs), "p-value (2 phía)": np.nan})
    boots = []
    for _ in range(cfg["causal"]["n_bootstrap"]):
        idx = rng.integers(0, len(yd), len(yd))
        boots.append(neyman(yd[idx], td[idx])[0])
    ref.append({"Kiểm định": "Bootstrap ATE (Neyman)", "Kỳ vọng": f"≈ {base_ate:.4f}", "Kết quả": np.mean(boots),
                "Độ lệch chuẩn": np.std(boots), "p-value (2 phía)": np.mean(np.array(boots) <= 0) * 2})
    ref = pd.DataFrame(ref)
    save_table(ref, cfg, "t18_refutation", "Các kiểm định phản bác (refutation tests) cho ước lượng nhân quả")
    m["refutation"] = ref.to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(plac, bins=40, alpha=.6, color=PALETTE[5], label="Placebo (T hoán vị)")
    ax.hist(boots, bins=40, alpha=.6, color=PALETTE[0], label="Bootstrap ATE thật")
    ax.axvline(0, color="k", lw=1); ax.legend(); ax.set_title("Phân phối placebo so với bootstrap ATE")
    save_fig(fig, cfg, "f20_placebo", "Kiểm định placebo: phân phối ATE khi hoán vị can thiệp so với ATE thật")

    # ---------------- CATE learners
    tr = camp[camp.split == "train"]
    e = float(tr["treatment_flg"].mean())
    log.info("Huấn luyện 6 meta-learner ...")
    L = UpliftLearners(cfg, feats, e).fit(tr, tr["target"].values.astype(float), tr["treatment_flg"].values.astype(float))
    preds = L.predict_all(camp)
    learners = ["S-learner", "T-learner", "X-learner", "R-learner", "DR-learner", "Class Transformation"]
    rows = []
    for split in ["val", "locked", "oot"]:
        d = camp.split == split
        for ln in learners:
            r = eval_uplift(y_all[d], t_all[d], preds.loc[d, ln].values)
            rows.append({"Learner": ln, "Tập": split, **r})
    ut = pd.DataFrame(rows)
    save_table(ut, cfg, "t19_uplift_learners", "So sánh các meta-learner uplift trên Validation, Locked Test và Out-of-Time")
    m["uplift_table"] = ut.to_dict(orient="records")
    best = ut[ut["Tập"] == "val"].sort_values("Qini_norm", ascending=False).iloc[0]["Learner"]
    m["best_learner"] = best
    log.info(f"Learner tốt nhất (Qini trên Validation): {best}")

    # Qini curves
    for split, fname in [("locked", "f21_qini_locked"), ("oot", "f22_qini_oot")]:
        d = (camp.split == split).values
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, ln in enumerate(learners):
            x, q = qini_curve(y_all[d], t_all[d], preds.loc[d, ln].values)
            ax.plot(x / x[-1], q, color=PALETTE[i], lw=2 if ln == best else 1.2,
                    label=f"{ln} (Qini={qini_auc(y_all[d], t_all[d], preds.loc[d, ln].values):.3f})")
        ax.plot([0, 1], [0, q[-1]], "k--", lw=1, label="Ngẫu nhiên")
        ax.set_xlabel("Tỷ lệ khách được nhắm (theo uplift giảm dần)"); ax.set_ylabel("Số lượt mua tăng thêm (Qini)")
        ax.set_title(f"Đường Qini - {'Locked Test' if split == 'locked' else 'Out-of-Time'}"); ax.legend(fontsize=8)
        save_fig(fig, cfg, fname, f"Đường Qini của các meta-learner trên tập {'Locked Test' if split == 'locked' else 'Out-of-Time'}")

    # uplift theo thập phân vị cho learner tốt nhất
    rows = []
    for split in ["locked", "oot"]:
        d = camp[camp.split == split].copy()
        d["u"] = preds.loc[d.index, best]
        d["dec"] = pd.qcut(d["u"].rank(method="first"), 10, labels=False)
        for k, g in d.groupby("dec"):
            a, se = neyman(g["target"].values.astype(float), g["treatment_flg"].values.astype(float))
            rows.append({"Tập": split, "Thập phân vị (1=uplift cao nhất)": 10 - k, "n": len(g),
                         "Uplift dự báo TB": g["u"].mean(), "Uplift quan sát": a, "SE": se,
                         "Tỷ lệ mua T": g.loc[g.treatment_flg == 1, "target"].mean(),
                         "Tỷ lệ mua C": g.loc[g.treatment_flg == 0, "target"].mean()})
    dec = pd.DataFrame(rows).sort_values(["Tập", "Thập phân vị (1=uplift cao nhất)"])
    save_table(dec, cfg, "t20_uplift_deciles", f"Uplift dự báo và quan sát theo thập phân vị ({best})")
    m["deciles"] = dec.to_dict(orient="records")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ["locked", "oot"]):
        g = dec[dec["Tập"] == split]
        x = g["Thập phân vị (1=uplift cao nhất)"]
        ax.bar(x - .2, g["Uplift quan sát"], .4, yerr=1.96 * g["SE"], color=PALETTE[0], capsize=3, label="Quan sát")
        ax.bar(x + .2, g["Uplift dự báo TB"], .4, color=PALETTE[1], label="Dự báo")
        ax.axhline(0, color="k", lw=1); ax.set_xticks(range(1, 11))
        ax.set_title(f"{best} - {'Locked Test' if split == 'locked' else 'Out-of-Time'}"); ax.legend()
        ax.set_xlabel("Thập phân vị uplift dự báo (1 = cao nhất)")
    save_fig(fig, cfg, "f23_uplift_deciles", "Uplift quan sát (KTC 95%) và uplift dự báo theo thập phân vị")

    # bootstrap Qini của learner tốt nhất trên locked
    d = (camp.split == "locked").values
    yl, tl, ul = y_all[d], t_all[d], preds.loc[d, best].values
    bq = []
    for _ in range(cfg["causal"]["n_bootstrap"]):
        idx = rng.integers(0, len(yl), len(yl))
        bq.append(qini_auc(yl[idx], tl[idx], ul[idx]))
    m["best_qini_locked_boot"] = {"mean": np.mean(bq), "lo": np.quantile(bq, .025), "hi": np.quantile(bq, .975)}

    # tương quan giữa các learner + độ bất đồng
    corr = preds[learners].corr(method="spearman")
    save_table(corr.reset_index().rename(columns={"index": "Learner"}), cfg, "t21_learner_corr",
               "Tương quan hạng Spearman giữa CATE của các learner")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(learners))); ax.set_xticklabels(learners, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(learners))); ax.set_yticklabels(learners, fontsize=8)
    for i in range(len(learners)):
        for j in range(len(learners)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("Tương quan Spearman giữa các ước lượng CATE"); ax.grid(False)
    save_fig(fig, cfg, "f24_learner_corr", "Ma trận tương quan hạng giữa ước lượng CATE của các learner")

    # phân phối CATE
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, ln in enumerate(learners):
        s = preds[ln].clip(preds[ln].quantile(.005), preds[ln].quantile(.995))
        ax.hist(s, bins=80, histtype="step", lw=1.5, color=PALETTE[i], label=ln, density=True)
    ax.axvline(0, color="k", lw=1); ax.legend(fontsize=8); ax.set_title("Phân phối CATE ước lượng"); ax.set_xlabel("τ̂(x)")
    save_fig(fig, cfg, "f25_cate_distribution", "Phân phối CATE ước lượng của các learner")

    # ---------------- không đồng nhất theo phân khúc (uplift quan sát)
    ev = camp[camp.split.isin(["locked", "oot"])].copy()
    ev["recency_bin"] = pd.cut(ev["recency"], [-1, 2, 4, 7, 10, 14, 1e9], labels=["≤2", "3-4", "5-7", "8-10", "11-14", ">14"]).astype(str)
    ev["freq_bin"] = pd.qcut(ev["n_tx_all"].rank(method="first"), 5, labels=["Q1 thấp", "Q2", "Q3", "Q4", "Q5 cao"]).astype(str)
    ev["gender"] = np.select([ev.gender_F == 1, ev.gender_M == 1], ["F", "M"], "U")
    ev["redeem"] = np.where(ev["has_redeemed"] == 1, "Đã đổi điểm", "Chưa đổi điểm")
    rows = []
    for col, nm in [("age_band", "Nhóm tuổi"), ("gender", "Giới tính"), ("recency_bin", "Recency (ngày)"),
                    ("freq_bin", "Tần suất mua"), ("redeem", "Đổi điểm")]:
        for k, g in ev.groupby(col):
            a, se = neyman(g["target"].values.astype(float), g["treatment_flg"].values.astype(float))
            rows.append({"Chiều phân khúc": nm, "Nhóm": k, "n": len(g), "Uplift quan sát": a, "SE": se,
                         "KTC95_dưới": a - 1.96 * se, "KTC95_trên": a + 1.96 * se,
                         "CATE dự báo TB": preds.loc[g.index, best].mean()})
    het = pd.DataFrame(rows)
    save_table(het, cfg, "t22_segment_heterogeneity", "Tác động không đồng nhất theo phân khúc (Locked + OOT)")
    m["segments"] = het.to_dict(orient="records")
    dims = het["Chiều phân khúc"].unique()
    fig, axes = plt.subplots(1, len(dims), figsize=(16, 4), sharey=True)
    for ax, dname in zip(axes, dims):
        g = het[het["Chiều phân khúc"] == dname]
        ax.errorbar(range(len(g)), g["Uplift quan sát"], yerr=1.96 * g["SE"], fmt="o", color=PALETTE[0], capsize=3, label="Quan sát")
        ax.plot(range(len(g)), g["CATE dự báo TB"], "s", color=PALETTE[1], label="Dự báo")
        ax.set_xticks(range(len(g))); ax.set_xticklabels(g["Nhóm"], rotation=45, fontsize=8)
        ax.axhline(0, color="k", lw=1); ax.set_title(dname, fontsize=9)
    axes[0].legend(fontsize=8)
    save_fig(fig, cfg, "f26_segment_heterogeneity", "Uplift quan sát và dự báo theo các chiều phân khúc")

    # phụ thuộc riêng phần của CATE
    cols = ["recency", "n_tx_all", "aov_all", "tenure_days", "age", "share_spent_tx"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, c in zip(axes.ravel(), cols):
        s = camp[c]
        q = pd.qcut(s.rank(method="first"), 20, labels=False)
        g = pd.DataFrame({"x": s, "u": preds[best], "q": q}).groupby("q").mean()
        ax.plot(g["x"], g["u"], "o-", color=PALETTE[0]); ax.axhline(0, color="k", lw=1); ax.set_title(c)
    fig.suptitle(f"CATE trung bình theo vigintile đặc trưng ({best})", fontweight="bold")
    save_fig(fig, cfg, "f27_cate_dependence", "Quan hệ giữa CATE dự báo và các đặc trưng chính")

    # tầm quan trọng (DR-learner final stage)
    imp = pd.Series(L.dr.booster_.feature_importance("gain"), index=feats)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    save_table(imp.rename("gain_share").reset_index().rename(columns={"index": "Đặc trưng"}), cfg,
               "t23_cate_importance", "Tỷ trọng gain của các đặc trưng trong mô hình CATE (DR-learner, giai đoạn cuối)")
    fig, ax = plt.subplots(figsize=(7, 7))
    imp.head(20).sort_values().plot.barh(ax=ax, color=PALETTE[2]); ax.set_title("Đặc trưng dẫn dắt tính không đồng nhất (DR-learner)")
    save_fig(fig, cfg, "f28_cate_importance", "Tầm quan trọng đặc trưng của mô hình CATE")

    # ---------------- lưu
    out = preds.copy()
    out["tau_hat"] = preds[best]
    out["tau_std"] = preds[learners].std(axis=1)
    out["p0_hat"] = preds["_mu0"]
    out["p1_hat"] = (preds["_mu0"] + preds[best]).clip(0, 1)
    out.to_parquet(cfg["paths"]["processed"] / "cate_campaign.parquet")
    pd.DataFrame({"psi": L.psi_train}, index=tr.index).to_parquet(cfg["paths"]["processed"] / "dr_pseudo_train.parquet")
    joblib.dump(L, cfg["paths"]["models"] / "uplift_learners.joblib")
    m["e_train"] = e
    save_metrics(cfg, "causal", m)
    return m


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
