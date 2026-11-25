"""Giai đoạn bổ trợ - Phân tích mở rộng và kiểm tra độ vững.

Không thay đổi bất kỳ lựa chọn nào của các giai đoạn 1-8. Các phân tích dùng để:
  (a) mô tả dữ liệu sâu hơn; (b) chẩn đoán mô hình dự báo; (c) phân tích từng meta-learner;
  (d) kiểm tra độ nhạy của SPO+, MILP và Action Router. Mọi phân tích độ nhạy của quy tắc
  được thực hiện trên tập VALIDATION; Locked Test chỉ dùng để xác nhận phương án đã chọn trước.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from s2_features import feature_cols, load_campaign, load_snapshots
from s3_forecast import ece_score
from s4_causal import neyman, qini_auc, qini_curve
from s5_spo import make_batches, regret
from s6_milp import dr_value, greedy, solve
from s7_router import ActionRouter
from utils import PALETTE, load_config, load_metrics, log, save_fig, save_metrics, save_table, set_seed, setup_mpl

LEARNERS = ["S-learner", "T-learner", "X-learner", "R-learner", "DR-learner", "Class Transformation"]
GROUPS = {
    "RFM & tần suất": ["recency", "n_tx_all", "n_tx_7d", "n_tx_14d", "n_tx_28d", "mean_gap", "first_tx_age", "has_tx"],
    "Giá trị & giỏ hàng": ["sum_all", "sum_7d", "sum_14d", "sum_28d", "aov_all", "aov_std", "aov_max", "items_per_tx", "qty_per_tx", "neg_tx_share"],
    "Xu hướng": ["trend_14_28", "sum_trend_14_28"],
    "Điểm thưởng": ["rp_recv_sum", "ep_recv_sum", "rp_spent_sum", "ep_spent_sum", "share_spent_tx"],
    "Thời gian & cửa hàng": ["share_weekend", "mean_hour", "n_stores"],
    "Thẻ & nhân khẩu học": ["tenure_days", "has_redeemed", "days_since_redeem", "issue_to_redeem_days", "age", "age_invalid", "gender_F", "gender_M", "gender_U"],
}


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    rng = np.random.default_rng(cfg["seed"] + 9)
    proc = cfg["paths"]["processed"]
    M = load_metrics(cfg)
    camp = load_campaign(cfg)
    fc = load_snapshots(cfg)
    feats = feature_cols(camp)
    ct = pd.read_parquet(proc / "cate_campaign.parquet")
    ds = pd.read_parquet(proc / "decision_scores.parquet")
    fcc = pd.read_parquet(proc / "forecast_campaign.parquet")
    fmod = joblib.load(cfg["paths"]["models"] / "forecast.joblib")
    bz, mc, rc = cfg["business"], cfg["milp"], cfg["router"]
    y_all, t_all = camp["target"].values.astype(float), camp["treatment_flg"].values.astype(float)
    m: dict = {}

    # =============================================================== (a) MÔ TẢ DỮ LIỆU
    # a1. phân phối đặc trưng: can thiệp vs đối chứng (lưới 9 hình)
    for gi, chunk in enumerate([feats[i:i + 9] for i in range(0, len(feats), 9)]):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        for ax, c in zip(axes.ravel(), chunk):
            s = camp[c].dropna()
            lo, hi = s.quantile(.005), s.quantile(.995)
            for tv, col, lab in [(1, PALETTE[0], "Can thiệp"), (0, PALETTE[1], "Đối chứng")]:
                ax.hist(s[camp.loc[s.index, "treatment_flg"] == tv].clip(lo, hi), bins=40, density=True,
                        alpha=.5, color=col, label=lab)
            ax.set_title(c, fontsize=9)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=8)
        save_fig(fig, cfg, f"fx01_feat_treat_control_{gi + 1}",
                 f"Phân phối đặc trưng giữa nhóm can thiệp và đối chứng (nhóm {gi + 1})")

    # a2. tương quan đặc trưng
    cor = camp[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cor.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=6.5)
    fig.colorbar(im, shrink=.7); ax.grid(False); ax.set_title("Ma trận tương quan Spearman giữa các đặc trưng (tập chiến dịch)")
    save_fig(fig, cfg, "fx02_feature_corr", "Ma trận tương quan hạng Spearman giữa các đặc trưng")
    pairs = cor.where(np.triu(np.ones(cor.shape), 1).astype(bool)).stack().rename("rho").reset_index()
    pairs["|rho|"] = pairs["rho"].abs()
    top = pairs.sort_values("|rho|", ascending=False).head(15)
    top.columns = ["Đặc trưng 1", "Đặc trưng 2", "Spearman ρ", "|ρ|"]
    save_table(top, cfg, "tx01_top_correlations", "Mười lăm cặp đặc trưng tương quan mạnh nhất")

    # a3. tỷ lệ mua, tỷ lệ can thiệp, chi tiêu theo phân khúc
    seg = camp.copy()
    seg["gender"] = np.select([seg.gender_F == 1, seg.gender_M == 1], ["F", "M"], "U")
    seg["freq_q"] = pd.qcut(seg["n_tx_all"].rank(method="first"), 5, labels=["Q1 thấp", "Q2", "Q3", "Q4", "Q5 cao"]).astype(str)
    seg["aov_q"] = pd.qcut(seg["aov_all"].rank(method="first"), 5, labels=["Q1 thấp", "Q2", "Q3", "Q4", "Q5 cao"]).astype(str)
    seg["tenure_q"] = pd.qcut(seg["tenure_days"].rank(method="first"), 5, labels=["Q1 mới", "Q2", "Q3", "Q4", "Q5 lâu"]).astype(str)
    rows = []
    for col, nm in [("age_band", "Nhóm tuổi"), ("gender", "Giới tính"), ("freq_q", "Tần suất mua"),
                    ("aov_q", "Giá trị giỏ"), ("tenure_q", "Thâm niên thẻ")]:
        for k, g in seg.groupby(col):
            rows.append({"Chiều": nm, "Nhóm": k, "Số khách": len(g), "Tỷ trọng (%)": 100 * len(g) / len(seg),
                         "Tỷ lệ can thiệp": g["treatment_flg"].mean(), "Tỷ lệ mua (đối chứng)": g.loc[g.treatment_flg == 0, "target"].mean(),
                         "Tỷ lệ mua (can thiệp)": g.loc[g.treatment_flg == 1, "target"].mean(),
                         "Chi tiêu TB cửa sổ nhìn lại (RUB)": g["sum_all"].mean(), "Số giao dịch TB": g["n_tx_all"].mean()})
    sp = pd.DataFrame(rows)
    save_table(sp, cfg, "tx02_segment_profile", "Hồ sơ phân khúc khách hàng: quy mô, tỷ lệ mua và chi tiêu")
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    for ax, (nm, g) in zip(axes, sp.groupby("Chiều", sort=False)):
        x = np.arange(len(g))
        ax.bar(x - .2, g["Tỷ lệ mua (đối chứng)"], .4, color=PALETTE[5], label="Đối chứng")
        ax.bar(x + .2, g["Tỷ lệ mua (can thiệp)"], .4, color=PALETTE[0], label="Can thiệp")
        ax.set_xticks(x); ax.set_xticklabels(g["Nhóm"], rotation=45, fontsize=8); ax.set_title(nm, fontsize=10); ax.set_ylim(0.3, 0.85)
    axes[0].legend(fontsize=8); axes[0].set_ylabel("Tỷ lệ mua")
    save_fig(fig, cfg, "fx03_segment_buy_rates", "Tỷ lệ mua của nhóm can thiệp và đối chứng theo phân khúc")

    # a4. hai chiều: tuổi x tần suất (uplift quan sát)
    ev = seg[seg.split.isin(["locked", "oot", "val"])]
    piv_u, piv_n = {}, {}
    for (a, f), g in ev.groupby(["age_band", "freq_q"]):
        if a == "Không rõ":
            continue
        u, _ = neyman(g["target"].values.astype(float), g["treatment_flg"].values.astype(float))
        piv_u[(a, f)] = u; piv_n[(a, f)] = len(g)
    pu = pd.Series(piv_u).unstack()
    pu = pu.reindex(["<25", "25-34", "35-44", "45-54", "55-64", "65+"])[["Q1 thấp", "Q2", "Q3", "Q4", "Q5 cao"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pu.values, cmap="RdYlGn", vmin=-0.08, vmax=0.12, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(pu.columns); ax.set_yticks(range(len(pu))); ax.set_yticklabels(pu.index)
    for i in range(pu.shape[0]):
        for j in range(pu.shape[1]):
            ax.text(j, i, f"{pu.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.grid(False); ax.set_xlabel("Tần suất mua"); ax.set_ylabel("Nhóm tuổi")
    ax.set_title("Uplift quan sát theo tuổi × tần suất mua (Validation + Locked + OOT)")
    save_fig(fig, cfg, "fx04_uplift_age_freq", "Bản đồ nhiệt uplift quan sát theo hai chiều tuổi và tần suất mua")
    tw = pu.reset_index().rename(columns={"index": "Nhóm tuổi"})
    save_table(tw, cfg, "tx03_uplift_age_freq", "Uplift quan sát theo hai chiều nhóm tuổi × tần suất mua")
    m["two_way_max"] = float(np.nanmax(pu.values)); m["two_way_min"] = float(np.nanmin(pu.values))

    # =============================================================== (b) CHẨN ĐOÁN DỰ BÁO
    gbm, cal = fmod["gbm"], fmod["calibrator"]
    lk = fc[(fc.snapshot == "S2") & (fc.split == "locked")]
    oot = fc[(fc.snapshot == "S3") & (fc.split.isin(["locked", "oot"]))]
    tr = fc[(fc.snapshot == "S1") & (fc.split == "train")]
    va = fc[(fc.snapshot == "S2") & (fc.split == "val")]
    p_lk = cal.predict(gbm.predict_proba(lk[feats])[:, 1]); p_oot = cal.predict(gbm.predict_proba(oot[feats])[:, 1])
    # b1. PR curve + gains
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (nm, d, p) in enumerate([("Locked Test (S2)", lk, p_lk), ("Out-of-Time (S3)", oot, p_oot)]):
        pr, rc_, _ = precision_recall_curve(d["y_buy"], p)
        ax[0].plot(rc_, pr, color=PALETTE[i], label=f"{nm} AP={average_precision_score(d['y_buy'], p):.3f}")
        o = np.argsort(-p); cum = np.cumsum(d["y_buy"].values[o]) / d["y_buy"].sum()
        ax[1].plot(np.arange(1, len(p) + 1) / len(p), cum, color=PALETTE[i], label=nm)
    ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision"); ax[0].legend(); ax[0].set_title("Đường Precision–Recall")
    ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].set_xlabel("Tỷ lệ khách (sắp theo xác suất)"); ax[1].set_ylabel("Tỷ lệ khách mua thu được")
    ax[1].legend(); ax[1].set_title("Đường gains tích lũy")
    save_fig(fig, cfg, "fx05_pr_gains", "Đường Precision–Recall và đường gains tích lũy của mô hình dự báo")
    # b2. phân rã Brier (Murphy) theo 15 ngăn phân vị
    rows = []
    for nm, d, p in [("Locked Test (S2)", lk, p_lk), ("Out-of-Time (S3)", oot, p_oot)]:
        y = d["y_buy"].values; ybar = y.mean()
        q = pd.qcut(p, 15, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(n=("y", "size"), pm=("p", "mean"), ym=("y", "mean"))
        rel = (g["n"] * (g["pm"] - g["ym"]) ** 2).sum() / len(y)
        res = (g["n"] * (g["ym"] - ybar) ** 2).sum() / len(y)
        unc = ybar * (1 - ybar)
        rows.append({"Tập": nm, "Brier": np.mean((p - y) ** 2), "Độ tin cậy (REL)": rel, "Độ phân giải (RES)": res,
                     "Độ bất định (UNC)": unc, "REL − RES + UNC": rel - res + unc})
    save_table(pd.DataFrame(rows), cfg, "tx04_brier_decomposition", "Phân rã Murphy của Brier score (mô hình đã hiệu chỉnh)")
    # b3. hiệu chỉnh theo nhóm tuổi
    cam_lk = lk.join(camp[["age_band"]], how="left")
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, band in zip(axes.ravel(), ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]):
        msk = (cam_lk["age_band"] == band).values
        y, p = lk["y_buy"].values[msk], p_lk[msk]
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.plot(g["p"], g["y"], "o-", color=PALETTE[0]); ax.set_title(f"Tuổi {band}", fontsize=10)
        rows.append({"Nhóm tuổi": band, "n": int(msk.sum()), "Tỷ lệ mua": y.mean(), "Xác suất TB": p.mean(),
                     "AUC": roc_auc_score(y, p), "ECE": ece_score(y, p)[0]})
    save_fig(fig, cfg, "fx06_calibration_by_age", "Biểu đồ độ tin cậy theo nhóm tuổi (Locked Test)")
    save_table(pd.DataFrame(rows), cfg, "tx05_calibration_by_age", "Hiệu năng và hiệu chỉnh của mô hình dự báo theo nhóm tuổi (Locked Test)")
    # b4. ablation nhóm đặc trưng
    log.info("Ablation nhóm đặc trưng ...")
    base_p = dict(cfg["lgbm"]); base_p["n_estimators"] = M["forecast"]["lgbm_best_iter"]
    rows = []
    full = lgb.LGBMClassifier(**base_p, random_state=cfg["seed"]).fit(tr[feats], tr["y_buy"])
    a_full = roc_auc_score(lk["y_buy"], full.predict_proba(lk[feats])[:, 1])
    rows.append({"Cấu hình": "Đầy đủ", "Số đặc trưng": len(feats), "AUC Locked": a_full, "Giảm AUC": 0.0})
    for gname, cols in GROUPS.items():
        keep = [c for c in feats if c not in cols]
        mdl = lgb.LGBMClassifier(**base_p, random_state=cfg["seed"]).fit(tr[keep], tr["y_buy"])
        a = roc_auc_score(lk["y_buy"], mdl.predict_proba(lk[keep])[:, 1])
        rows.append({"Cấu hình": f"Bỏ nhóm: {gname}", "Số đặc trưng": len(keep), "AUC Locked": a, "Giảm AUC": a_full - a})
    for gname, cols in GROUPS.items():
        keep = [c for c in feats if c in cols]
        mdl = lgb.LGBMClassifier(**base_p, random_state=cfg["seed"]).fit(tr[keep], tr["y_buy"])
        a = roc_auc_score(lk["y_buy"], mdl.predict_proba(lk[keep])[:, 1])
        rows.append({"Cấu hình": f"Chỉ nhóm: {gname}", "Số đặc trưng": len(keep), "AUC Locked": a, "Giảm AUC": a_full - a})
    ab = pd.DataFrame(rows)
    save_table(ab, cfg, "tx06_feature_group_ablation", "Phân tích loại bỏ nhóm đặc trưng (ablation) của mô hình dự báo")
    fig, ax = plt.subplots(figsize=(9, 5))
    ab2 = ab.iloc[1:].copy()
    ax.barh(ab2["Cấu hình"], ab2["AUC Locked"], color=[PALETTE[1]] * len(GROUPS) + [PALETTE[0]] * len(GROUPS))
    ax.axvline(a_full, color="k", ls="--", label=f"Đầy đủ = {a_full:.4f}"); ax.set_xlim(0.5, max(0.85, a_full + 0.02)); ax.legend()
    ax.set_xlabel("AUC trên Locked Test"); ax.set_title("Ablation theo nhóm đặc trưng")
    save_fig(fig, cfg, "fx07_ablation", "AUC khi bỏ hoặc chỉ giữ từng nhóm đặc trưng")
    # b5. learning curve
    rows = []
    for frac in [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        sub = tr.sample(frac=frac, random_state=cfg["seed"])
        mdl = lgb.LGBMClassifier(**base_p, random_state=cfg["seed"]).fit(sub[feats], sub["y_buy"])
        p = mdl.predict_proba(lk[feats])[:, 1]; pt = mdl.predict_proba(sub[feats])[:, 1]
        rows.append({"Tỷ lệ dữ liệu train": frac, "Số mẫu": len(sub), "AUC train": roc_auc_score(sub["y_buy"], pt),
                     "AUC Locked": roc_auc_score(lk["y_buy"], p)})
    lc = pd.DataFrame(rows)
    save_table(lc, cfg, "tx07_learning_curve", "Đường cong học: AUC theo quy mô dữ liệu huấn luyện")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lc["Số mẫu"], lc["AUC train"], "o-", color=PALETTE[1], label="Train"); ax.plot(lc["Số mẫu"], lc["AUC Locked"], "s-", color=PALETTE[0], label="Locked Test")
    ax.set_xscale("log"); ax.set_xlabel("Số mẫu huấn luyện"); ax.set_ylabel("AUC"); ax.legend(); ax.set_title("Đường cong học của mô hình dự báo")
    save_fig(fig, cfg, "fx08_learning_curve", "Đường cong học của mô hình dự báo")

    # =============================================================== (c) TỪNG META-LEARNER
    rows, pol_rows = [], []
    for split in ["val", "locked", "oot"]:
        d = (camp.split == split).values
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        for ax, ln in zip(axes.ravel(), LEARNERS):
            u = ct.loc[d, ln].values
            dec = pd.qcut(pd.Series(u).rank(method="first"), 10, labels=False).values
            obs, pred, se = [], [], []
            for k in range(9, -1, -1):
                mm = dec == k
                a, s_ = neyman(y_all[d][mm], t_all[d][mm]); obs.append(a); se.append(s_); pred.append(u[mm].mean())
                rows.append({"Tập": split, "Learner": ln, "Thập phân vị": 10 - k, "Uplift dự báo": u[mm].mean(), "Uplift quan sát": a, "SE": s_})
            x = np.arange(1, 11)
            ax.bar(x - .2, obs, .4, yerr=1.96 * np.array(se), color=PALETTE[0], capsize=2, label="Quan sát")
            ax.bar(x + .2, pred, .4, color=PALETTE[1], label="Dự báo"); ax.axhline(0, color="k", lw=1)
            ax.set_title(ln, fontsize=10); ax.set_xticks(x)
            # đường giá trị chính sách: lợi nhuận DR khi nhắm top q%
            for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                sel = np.zeros(d.sum()); sel[np.argsort(-u)[:int(q * d.sum())]] = 1
                pol_rows.append({"Tập": split, "Learner": ln, "Tỷ lệ nhắm": q,
                                 "Lượt mua tăng thêm (DR)": float((sel * ds.loc[d, "psi"].values).sum()),
                                 "Lợi nhuận DR (RUB)": float((sel * ds.loc[d, "value_proxy"].values).sum())})
        axes[0, 0].legend(fontsize=8)
        nm = {"val": "Validation", "locked": "Locked Test", "oot": "Out-of-Time"}[split]
        save_fig(fig, cfg, f"fx09_learner_deciles_{split}", f"Uplift quan sát và dự báo theo thập phân vị cho sáu learner – {nm}")
    dl = pd.DataFrame(rows)
    save_table(dl[dl["Tập"] == "locked"], cfg, "tx08_learner_deciles_locked", "Uplift theo thập phân vị của sáu learner trên Locked Test")
    # chỉ số hiệu chỉnh uplift: hệ số góc hồi quy uplift quan sát theo dự báo, tương quan
    rows = []
    for (s_, ln), g in dl.groupby(["Tập", "Learner"]):
        b = np.polyfit(g["Uplift dự báo"], g["Uplift quan sát"], 1)[0]
        rows.append({"Tập": s_, "Learner": ln, "Hệ số góc (quan sát ~ dự báo)": b,
                     "Tương quan thập phân vị": np.corrcoef(g["Uplift dự báo"], g["Uplift quan sát"])[0, 1],
                     "Số thập phân vị dự báo âm": int((g["Uplift dự báo"] < 0).sum()),
                     "Số thập phân vị quan sát âm có ý nghĩa": int((g["Uplift quan sát"] + 1.96 * g["SE"] < 0).sum())})
    uc = pd.DataFrame(rows)
    save_table(uc, cfg, "tx09_uplift_calibration", "Độ hiệu chỉnh của uplift dự báo theo learner (hồi quy trên 10 thập phân vị)")
    m["uplift_calibration"] = uc.to_dict(orient="records")
    pc = pd.DataFrame(pol_rows)
    save_table(pc[pc["Tập"] != "val"], cfg, "tx10_policy_curves", "Lợi nhuận DR khi nhắm top q% theo từng learner (Locked Test, OOT)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, split in zip(axes, ["val", "locked", "oot"]):
        for i, ln in enumerate(LEARNERS):
            g = pc[(pc["Tập"] == split) & (pc["Learner"] == ln)]
            ax.plot(g["Tỷ lệ nhắm"], g["Lợi nhuận DR (RUB)"], "o-", ms=3, color=PALETTE[i], label=ln)
        ax.axhline(0, color="k", lw=1); ax.set_xlabel("Tỷ lệ khách được nhắm"); ax.set_title({"val": "Validation", "locked": "Locked Test", "oot": "Out-of-Time"}[split])
    axes[0].set_ylabel("Lợi nhuận tăng thêm DR (RUB)"); axes[0].legend(fontsize=7)
    save_fig(fig, cfg, "fx10_policy_curves", "Đường lợi nhuận DR theo tỷ lệ khách được nhắm cho từng learner")
    # ensemble X/R/DR
    ens = ct[["X-learner", "R-learner", "DR-learner"]].mean(axis=1)
    rows = []
    for split in ["val", "locked", "oot"]:
        d = (camp.split == split).values
        for nm, u in [("Ensemble X+R+DR", ens.values[d])] + [(ln, ct.loc[d, ln].values) for ln in LEARNERS]:
            rows.append({"Tập": split, "Mô hình": nm, "Qini_norm": qini_auc(y_all[d], t_all[d], u)})
    en = pd.DataFrame(rows).pivot(index="Mô hình", columns="Tập", values="Qini_norm")[["val", "locked", "oot"]].reset_index()
    en["Hạng TB"] = en[["val", "locked", "oot"]].rank(ascending=False).mean(axis=1)
    save_table(en.sort_values("Hạng TB"), cfg, "tx11_ensemble_qini", "Hệ số Qini của tổ hợp X+R+DR-learner so với từng learner")
    m["ensemble_qini"] = en.to_dict(orient="records")

    # =============================================================== (d) ĐỘ NHẠY SPO+ / MILP / ROUTER
    # d1. SPO: regret theo kích thước lô và tỷ lệ chọn (đánh giá các điểm số đã lưu)
    rows = []
    chat_cols = [c for c in ds.columns if c.startswith("chat_")]
    C = -ds["value_proxy"].values
    idx_l = np.where(camp.split.values == "locked")[0]
    for n in [50, 100, 200, 500]:
        for kf in [0.1, 0.3, 0.5]:
            k = max(1, int(n * kf))
            B = idx_l[make_batches(len(idx_l), n, np.random.default_rng(5), reps=3)]
            for c in chat_cols:
                r, z, W = regret(ds[c].values[B], C[B], k)
                rows.append({"Kích thước lô n": n, "Tỷ lệ chọn k/n": kf, "Phương pháp": c.replace("chat_", ""),
                             "Regret chuẩn hóa": r.sum() / np.abs(z).sum(), "Giá trị DR TB / khách (RUB)": -(C[B] * W).sum() / B.size})
    sb = pd.DataFrame(rows)
    save_table(sb, cfg, "tx12_spo_batch_sensitivity", "Regret chuẩn hóa theo kích thước lô và tỷ lệ chọn (Locked Test)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, kf in zip(axes, [0.1, 0.3, 0.5]):
        for i, (nm, g) in enumerate(sb[sb["Tỷ lệ chọn k/n"] == kf].groupby("Phương pháp")):
            ax.plot(g["Kích thước lô n"], g["Regret chuẩn hóa"], "o-", color=PALETTE[i], label=nm)
        ax.set_xscale("log"); ax.set_title(f"k/n = {kf}"); ax.set_xlabel("Kích thước lô n")
    axes[0].set_ylabel("Regret chuẩn hóa"); axes[0].legend(fontsize=7)
    save_fig(fig, cfg, "fx11_spo_batch_sensitivity", "Regret chuẩn hóa theo kích thước lô và tỷ lệ chọn")

    # d2. MILP với các nguồn điểm quyết định khác nhau (đánh giá trên Validation và Locked)
    S, p1 = ds["spend_cap"].values, ds["p1_hat"].values
    cost_all = ds["cost_i"].values
    sources = {"SPO+ (đã chọn)": ds["value_hat"].values,
               "Plug-in CATE (learner đã chọn)": -ds["chat_PTO - Plug-in CATE (LightGBM)"].values,
               "Plug-in Ensemble X+R+DR": ens.values * bz["margin_rate"] * S - bz["contact_cost"] - (ct["p0_hat"].values + ens.values) * bz["incentive_cost"],
               "PTO - Ridge": -ds["chat_PTO - Ridge (MSE)"].values}
    rows = []
    for split in ["val", "locked", "oot"]:
        d = (camp.split == split).values
        N = d.sum(); K = int(mc["capacity_frac"] * N); Bd = mc["budget_frac_of_capacity_cost"] * K * cost_all[d].mean()
        vp = ds.loc[d, "value_proxy"].values
        bidx = rng.integers(0, N, (200, N))
        for nm, V in sources.items():
            xg = greedy(V[d], cost_all[d], camp.loc[d, "age_band"].values, K, Bd, mc["segment_max_share"], mc["segment_min_coverage"])
            x, info = solve(V[d], cost_all[d], camp.loc[d, "age_band"].values, K, Bd, mc["segment_max_share"], mc["segment_min_coverage"],
                            mc["solver"], time_limit=60, hint=xg)
            x = np.round(x)
            bt = np.array([dr_value(x[b], vp[b]) for b in bidx])
            rows.append({"Tập": split, "Nguồn điểm quyết định": nm, "Số gửi": int(x.sum()), "Giá trị mục tiêu V̂": info["objective"],
                         "Lợi nhuận DR (RUB)": dr_value(x, vp), "KTC95 dưới": np.quantile(bt, .025), "KTC95 trên": np.quantile(bt, .975)})
    sc = pd.DataFrame(rows)
    save_table(sc, cfg, "tx13_milp_score_sources", "MILP với các nguồn điểm quyết định khác nhau (Validation, Locked Test, OOT)")
    m["milp_score_sources"] = sc.to_dict(orient="records")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, split in zip(axes, ["val", "locked", "oot"]):
        g = sc[sc["Tập"] == split]
        ax.barh(g["Nguồn điểm quyết định"], g["Lợi nhuận DR (RUB)"],
                xerr=[g["Lợi nhuận DR (RUB)"] - g["KTC95 dưới"], g["KTC95 trên"] - g["Lợi nhuận DR (RUB)"]], color=PALETTE[:4], capsize=3)
        ax.axvline(0, color="k", lw=1); ax.set_title({"val": "Validation", "locked": "Locked Test", "oot": "Out-of-Time"}[split])
    save_fig(fig, cfg, "fx12_milp_score_sources", "Lợi nhuận DR của nghiệm MILP theo nguồn điểm quyết định")

    # d3. công bằng: kiểm tra áp lực sàn độ phủ (giá của công bằng khi ràng buộc chặt)
    d = (camp.split == "locked").values
    N = d.sum(); K = int(mc["capacity_frac"] * N); Bd = mc["budget_frac_of_capacity_cost"] * K * cost_all[d].mean()
    V = ds.loc[d, "value_hat"].values; segv = camp.loc[d, "age_band"].values; vp = ds.loc[d, "value_proxy"].values
    x0, i0 = solve(V, cost_all[d], segv, K, Bd, 1.0, 0.0, mc["solver"], time_limit=60)
    rows = []
    for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        for a in [1.0, 0.30, 0.25]:
            x, info = solve(V, cost_all[d], segv, K, Bd, a, b, mc["solver"], time_limit=60)
            x = np.round(x)
            rows.append({"Sàn độ phủ b": b, "Trần tập trung a": a, "Trạng thái": info["status"], "Giá trị mục tiêu V̂": info["objective"],
                         "Cái giá của công bằng (V̂)": i0["objective"] - info["objective"] if info["status"] in ("OPTIMAL", "FEASIBLE") else np.nan,
                         "Lợi nhuận DR (RUB)": dr_value(x, vp) if info["status"] in ("OPTIMAL", "FEASIBLE") else np.nan, "Số gửi": int(x.sum())})
    fs = pd.DataFrame(rows)
    save_table(fs, cfg, "tx14_fairness_stress", "Kiểm tra áp lực của ràng buộc công bằng: cái giá của công bằng theo sàn và trần (Locked Test)")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, a in enumerate([1.0, 0.30, 0.25]):
        g = fs[fs["Trần tập trung a"] == a]
        ax.plot(g["Sàn độ phủ b"], g["Cái giá của công bằng (V̂)"], "o-", color=PALETTE[i], label=f"Trần a = {a}")
    ax.set_xlabel("Sàn độ phủ b (tỷ lệ khách mỗi nhóm tuổi)"); ax.set_ylabel("Giảm giá trị mục tiêu V̂"); ax.legend()
    ax.set_title("Cái giá của công bằng khi ràng buộc trở nên chặt")
    save_fig(fig, cfg, "fx13_fairness_stress", "Cái giá của công bằng theo mức sàn độ phủ và trần tập trung")

    # d4. độ nhạy ngưỡng rào chắn (trên VALIDATION) + xác nhận trên Locked
    d = (camp.split == "val").values
    N = d.sum(); K = int(mc["capacity_frac"] * N); Bd = mc["budget_frac_of_capacity_cost"] * K * cost_all[d].mean()
    xg = greedy(ds.loc[d, "value_hat"].values, cost_all[d], camp.loc[d, "age_band"].values, K, Bd, mc["segment_max_share"], mc["segment_min_coverage"])
    xv, _ = solve(ds.loc[d, "value_hat"].values, cost_all[d], camp.loc[d, "age_band"].values, K, Bd, mc["segment_max_share"],
                  mc["segment_min_coverage"], mc["solver"], time_limit=60, hint=xg)
    base_df = camp.loc[d, ["recency", "sum_all", "has_tx"]].copy()
    base_df["tau_hat"] = ct.loc[d, "tau_hat"]; base_df["tau_std"] = ct.loc[d, "tau_std"]
    base_df["value_hat"] = ds.loc[d, "value_hat"]; base_df["x"] = np.round(xv).astype(int)
    vpv = ds.loc[d, "value_proxy"].values
    th = M["router"]["thresholds"]
    rows = []
    for sd in [-1.0, -0.20, -0.10, -0.05, -0.03, -0.01, 0.0]:
        cfg2 = {**cfg, "router": {**rc, "sleeping_dog_tau": sd}}
        r = ActionRouter(cfg2, th).route(base_df)
        xa = (r["action"] == "AUTO_SEND").values.astype(float)
        rows.append({"Ngưỡng sleeping dog τ": sd, "Số SUPPRESS": int((r["action"] == "SUPPRESS").sum()), "Số AUTO_SEND": int(xa.sum()),
                     "Lợi nhuận DR AUTO_SEND (RUB)": float(xa @ vpv)})
    sdt = pd.DataFrame(rows)
    save_table(sdt, cfg, "tx15_guardrail_threshold_val", "Độ nhạy của ngưỡng rào chắn 'sleeping dog' (đánh giá trên Validation)")
    # rào chắn đồng thuận: chặn khi ≥ 5/6 learner cùng dự báo uplift âm
    neg_votes = (ct[LEARNERS] < 0).sum(axis=1)
    rows = []
    for split in ["val", "locked", "oot"]:
        dd = (camp.split == split).values
        if split == "val":
            xs = np.round(xv)
        elif split == "locked":
            xs = pd.read_csv(proc / "milp_solution_locked.csv", dtype={"client_id": str}).set_index("client_id").reindex(camp.index[dd])["x"].values
        else:
            Nn = dd.sum(); Kk = int(mc["capacity_frac"] * Nn); Bb = mc["budget_frac_of_capacity_cost"] * Kk * cost_all[dd].mean()
            xg2 = greedy(ds.loc[dd, "value_hat"].values, cost_all[dd], camp.loc[dd, "age_band"].values, Kk, Bb, mc["segment_max_share"], mc["segment_min_coverage"])
            xs, _ = solve(ds.loc[dd, "value_hat"].values, cost_all[dd], camp.loc[dd, "age_band"].values, Kk, Bb, mc["segment_max_share"],
                          mc["segment_min_coverage"], mc["solver"], time_limit=60, hint=xg2)
            xs = np.round(xs)
        vps = ds.loc[dd, "value_proxy"].values
        for nm, blk in [("Không rào chắn", np.zeros(dd.sum(), bool)),
                        ("R1 gốc: τ̂ learner đã chọn < −0,01", (ct.loc[dd, "tau_hat"] < rc["sleeping_dog_tau"]).values),
                        ("R1 đồng thuận: ≥ 5/6 learner âm", (neg_votes[dd] >= 5).values),
                        ("R1 đồng thuận: 6/6 learner âm", (neg_votes[dd] >= 6).values)]:
            xx = xs * (~blk)
            rows.append({"Tập": split, "Phương án rào chắn": nm, "Số bị chặn trong tập MILP chọn": int((xs * blk).sum()),
                         "Số khách bị chặn (toàn tập)": int(blk.sum()), "Lợi nhuận DR (RUB)": float(xx @ vps)})
    cg = pd.DataFrame(rows)
    save_table(cg, cfg, "tx16_consensus_guardrail", "So sánh rào chắn 'sleeping dog' gốc và rào chắn đồng thuận giữa các learner")
    m["consensus_guardrail"] = cg.to_dict(orient="records")
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, split in zip(axes, ["val", "locked", "oot"]):
        g = cg[cg["Tập"] == split]
        ax.barh(g["Phương án rào chắn"], g["Lợi nhuận DR (RUB)"], color=PALETTE[:4]); ax.axvline(0, color="k", lw=1)
        ax.set_title({"val": "Validation", "locked": "Locked Test", "oot": "Out-of-Time"}[split])
    save_fig(fig, cfg, "fx14_consensus_guardrail", "Lợi nhuận DR theo phương án rào chắn 'sleeping dog'")

    # d5. độ trễ chấm điểm theo lô
    gb = fmod["gbm"].booster_
    X = camp[feats].astype("float32").values
    rows = []
    for bsz in [1, 10, 100, 1000, 10000]:
        ts = []
        for _ in range(30 if bsz <= 1000 else 5):
            ii = rng.integers(0, len(X), bsz)
            t0 = time.perf_counter(); gb.predict(X[ii]); ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000
        rows.append({"Kích thước lô": bsz, "Thời gian TB (ms)": ts.mean(), "P95 (ms)": np.quantile(ts, .95),
                     "Thời gian / khách (µs)": 1000 * ts.mean() / bsz, "Thông lượng (khách/giây)": bsz / (ts.mean() / 1000)})
    bl = pd.DataFrame(rows)
    save_table(bl, cfg, "tx17_batch_latency", "Độ trễ và thông lượng chấm điểm dự báo theo kích thước lô")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bl["Kích thước lô"], bl["Thời gian / khách (µs)"], "o-", color=PALETTE[0]); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Kích thước lô"); ax.set_ylabel("Thời gian mỗi khách (µs)"); ax.set_title("Hiệu quả chấm điểm theo lô")
    save_fig(fig, cfg, "fx15_batch_latency", "Thời gian chấm điểm mỗi khách hàng theo kích thước lô")

    # d6. trôi dạt theo tuần: trung bình đặc trưng hành vi giữa các snapshot
    rows = []
    for c in ["recency", "n_tx_all", "n_tx_14d", "aov_all", "sum_all", "share_spent_tx", "tenure_days", "trend_14_28"]:
        for sname in ["S1", "S2", "S3"]:
            s = fc.loc[fc.snapshot == sname, c]
            rows.append({"Đặc trưng": c, "Snapshot": sname, "Trung bình": s.mean(), "Trung vị": s.median(), "P90": s.quantile(.9)})
    sd_ = pd.DataFrame(rows)
    save_table(sd_, cfg, "tx18_snapshot_feature_stats", "Thống kê đặc trưng hành vi qua ba snapshot")
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, (c, g) in zip(axes.ravel(), sd_.groupby("Đặc trưng", sort=False)):
        ax.plot(g["Snapshot"], g["Trung bình"], "o-", color=PALETTE[0], label="TB"); ax.plot(g["Snapshot"], g["Trung vị"], "s--", color=PALETTE[1], label="Trung vị")
        ax.set_title(c, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    save_fig(fig, cfg, "fx16_snapshot_feature_trends", "Diễn biến của các đặc trưng hành vi qua ba snapshot")

    # =============================================================== (e) BỔ SUNG
    # e1. phân phối Train vs cohort OOT cho toàn bộ đặc trưng
    for gi, chunk in enumerate([feats[i:i + 9] for i in range(0, len(feats), 9)]):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        for ax, c in zip(axes.ravel(), chunk):
            s = camp[c].dropna(); lo, hi = s.quantile(.005), s.quantile(.995)
            for sp_, col in [("train", PALETTE[0]), ("oot", PALETTE[1])]:
                ax.hist(s[camp.loc[s.index, "split"] == sp_].clip(lo, hi), bins=40, density=True, alpha=.5, color=col, label=sp_)
            ax.set_title(c, fontsize=9)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=8)
        save_fig(fig, cfg, f"fx17_feat_train_oot_{gi + 1}", f"Phân phối đặc trưng giữa Train và cohort Out-of-Time (nhóm {gi + 1})")

    # e2. Qini trên Validation (tập dùng để chọn learner)
    d = (camp.split == "val").values
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, ln in enumerate(LEARNERS):
        x, q = qini_curve(y_all[d], t_all[d], ct.loc[d, ln].values)
        ax.plot(x / x[-1], q, color=PALETTE[i], label=f"{ln} (Qini={qini_auc(y_all[d], t_all[d], ct.loc[d, ln].values):.4f})")
    ax.plot([0, 1], [0, q[-1]], "k--", lw=1, label="Ngẫu nhiên"); ax.legend(fontsize=8)
    ax.set_xlabel("Tỷ lệ khách được nhắm"); ax.set_ylabel("Số lượt mua tăng thêm (Qini)"); ax.set_title("Đường Qini - Validation")
    save_fig(fig, cfg, "fx18_qini_val", "Đường Qini của các meta-learner trên tập Validation (tập dùng để lựa chọn)")

    # e3. phân phối bootstrap của Qini theo learner (Locked)
    d = (camp.split == "locked").values
    yl, tl = y_all[d], t_all[d]
    bidx = rng.integers(0, d.sum(), (100, d.sum()))
    rows = []
    boots = {ln: np.array([qini_auc(yl[b], tl[b], ct.loc[d, ln].values[b]) for b in bidx]) for ln in LEARNERS}
    for ln, b in boots.items():
        rows.append({"Learner": ln, "Qini TB bootstrap": b.mean(), "Độ lệch chuẩn": b.std(), "KTC95 dưới": np.quantile(b, .025),
                     "KTC95 trên": np.quantile(b, .975), "P(Qini > 0)": (b > 0).mean(),
                     "P(là learner tốt nhất)": np.mean([max(boots, key=lambda k: boots[k][j]) == ln for j in range(len(b))])})
    bq = pd.DataFrame(rows)
    save_table(bq, cfg, "tx19_qini_bootstrap", "Phân phối bootstrap của hệ số Qini theo learner trên Locked Test (100 lần)")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.boxplot([boots[ln] for ln in LEARNERS], tick_labels=LEARNERS, showfliers=False)
    ax.axhline(0, color="k", lw=1); ax.tick_params(axis="x", rotation=20); ax.set_ylabel("Qini chuẩn hóa")
    ax.set_title("Phân phối bootstrap của hệ số Qini (Locked Test)")
    save_fig(fig, cfg, "fx19_qini_bootstrap", "Phân phối bootstrap của hệ số Qini theo learner trên Locked Test")

    # e4. hồ sơ khách hàng được MILP chọn so với không được chọn (Locked)
    sol = pd.read_csv(proc / "milp_solution_locked.csv", dtype={"client_id": str}).set_index("client_id")
    lock = camp[camp.split == "locked"].copy()
    lock["x"] = sol.reindex(lock.index)["x"].values
    cols = ["age", "recency", "n_tx_all", "aov_all", "sum_all", "tenure_days", "has_redeemed", "share_spent_tx", "n_stores", "gender_F", "gender_M"]
    rows = []
    for c in cols:
        a, b = lock.loc[lock.x == 1, c], lock.loc[lock.x == 0, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"Đặc trưng": c, "TB nhóm được chọn": a.mean(), "TB nhóm không chọn": b.mean(), "SMD": (a.mean() - b.mean()) / sd if sd > 0 else 0})
    extra = pd.DataFrame({"x": lock["x"], "tau": ct.loc[lock.index, "tau_hat"], "ens": ens.loc[lock.index], "p1": ds.loc[lock.index, "p1_hat"],
                          "S": ds.loc[lock.index, "spend_cap"], "cost": ds.loc[lock.index, "cost_i"], "pb": fcc.loc[lock.index, "p_buy_cal"]})
    for c, nm in [("tau", "Uplift dự báo (learner đã chọn)"), ("ens", "Uplift dự báo (ensemble)"), ("p1", "Xác suất mua khi gửi p̂₁"),
                  ("S", "Chi tiêu kỳ vọng Ŝ (RUB)"), ("cost", "Chi phí kỳ vọng (RUB)"), ("pb", "Xác suất mua 14 ngày (dự báo)")]:
        a, b = extra.loc[extra.x == 1, c], extra.loc[extra.x == 0, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"Đặc trưng": nm, "TB nhóm được chọn": a.mean(), "TB nhóm không chọn": b.mean(), "SMD": (a.mean() - b.mean()) / sd if sd > 0 else 0})
    prof = pd.DataFrame(rows)
    save_table(prof, cfg, "tx20_selected_profile", "Hồ sơ khách hàng được MILP chọn so với không được chọn (Locked Test)")
    obs_rows = []
    for xv_ in [1, 0]:
        g = lock[lock.x == xv_]
        a, se = neyman(g["target"].values.astype(float), g["treatment_flg"].values.astype(float))
        obs_rows.append({"Nhóm": "Được MILP chọn" if xv_ == 1 else "Không được chọn", "n": len(g), "Uplift quan sát": a, "SE": se,
                         "KTC95 dưới": a - 1.96 * se, "KTC95 trên": a + 1.96 * se,
                         "Tỷ lệ mua đối chứng": g.loc[g.treatment_flg == 0, "target"].mean()})
    so = pd.DataFrame(obs_rows)
    save_table(so, cfg, "tx21_selected_uplift", "Uplift quan sát của nhóm được MILP chọn và không được chọn (Locked Test)")
    m["selected_uplift"] = so.to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(7, 5))
    p2 = prof.iloc[::-1]
    ax.barh(p2["Đặc trưng"], p2["SMD"], color=[PALETTE[0] if v > 0 else PALETTE[1] for v in p2["SMD"]])
    ax.axvline(0, color="k", lw=1); ax.set_xlabel("SMD (được chọn − không được chọn)"); ax.set_title("Ai được MILP chọn?")
    save_fig(fig, cfg, "fx20_selected_profile", "Khác biệt chuẩn hóa giữa khách hàng được và không được MILP chọn")

    # e5. tối ưu bảo thủ theo cận dưới CQR của chi tiêu
    rows = []
    for split in ["val", "locked", "oot"]:
        d = (camp.split == split).values
        N = d.sum(); K = int(mc["capacity_frac"] * N); Bd = mc["budget_frac_of_capacity_cost"] * K * cost_all[d].mean()
        vp = ds.loc[d, "value_proxy"].values; segv = camp.loc[d, "age_band"].values
        tau_e = ens.values[d]; p0 = ct["p0_hat"].values[d]
        S_pt = ds["spend_cap"].values[d]; S_lo = np.minimum(fcc["spend_lo"].values[d], S_pt)
        for nm, V in [("Ensemble, chi tiêu điểm", tau_e * bz["margin_rate"] * S_pt - bz["contact_cost"] - (p0 + tau_e) * bz["incentive_cost"]),
                      ("Ensemble, chi tiêu cận dưới CQR", tau_e * bz["margin_rate"] * S_lo - bz["contact_cost"] - (p0 + tau_e) * bz["incentive_cost"]),
                      ("Ensemble − 1 độ lệch chuẩn learner", (tau_e - ct["tau_std"].values[d]) * bz["margin_rate"] * S_pt - bz["contact_cost"] - (p0 + tau_e) * bz["incentive_cost"])]:
            xg = greedy(V, cost_all[d], segv, K, Bd, mc["segment_max_share"], mc["segment_min_coverage"])
            x, info = solve(V, cost_all[d], segv, K, Bd, mc["segment_max_share"], mc["segment_min_coverage"], mc["solver"], time_limit=60, hint=xg)
            x = np.round(x)
            rows.append({"Tập": split, "Hàm giá trị": nm, "Số gửi": int(x.sum()), "Số khách V̂>0": int((V > 0).sum()),
                         "Giá trị mục tiêu": info["objective"], "Lợi nhuận DR (RUB)": dr_value(x, vp),
                         "Tỷ lệ mục tiêu / DR": info["objective"] / dr_value(x, vp) if dr_value(x, vp) != 0 else np.nan})
    rb = pd.DataFrame(rows)
    save_table(rb, cfg, "tx22_conservative_value", "Tối ưu bảo thủ: hàm giá trị dùng cận dưới chi tiêu hoặc trừ bất định uplift")
    m["conservative"] = rb.to_dict(orient="records")

    # e6. độ phủ CQR theo nhóm tuổi
    lkb = lk[lk.y_buy == 1].join(camp[["age_band"]], how="left")
    alpha = cfg["conformal"]["alpha"]
    qm = fmod["q_models"]; qh = fmod["qhat"]
    ylog = np.log1p(lkb["y_spend"].clip(lower=0).values)
    lo = qm[alpha / 2].predict(lkb[feats]) - qh; hi = qm[1 - alpha / 2].predict(lkb[feats]) + qh
    rows = []
    for band in ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]:
        mm = (lkb["age_band"] == band).values
        rows.append({"Nhóm tuổi": band, "n (khách có mua)": int(mm.sum()), "Độ phủ CQR": np.mean((ylog[mm] >= lo[mm]) & (ylog[mm] <= hi[mm])),
                     "Độ rộng TB (RUB)": np.mean(np.expm1(hi[mm]) - np.expm1(lo[mm])), "Chi tiêu TB thực (RUB)": lkb["y_spend"].values[mm].mean()})
    save_table(pd.DataFrame(rows), cfg, "tx23_cqr_by_age", "Độ phủ và độ rộng khoảng CQR theo nhóm tuổi (Locked Test)")

    # e7. phụ thuộc riêng phần của mô hình dự báo
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    Xs = lk[feats].sample(3000, random_state=cfg["seed"])
    for ax, c in zip(axes.ravel(), ["n_tx_all", "mean_gap", "n_tx_28d", "recency", "sum_trend_14_28", "age"]):
        grid = np.unique(np.nanquantile(Xs[c], np.linspace(.02, .98, 20)))
        vals = []
        for g_ in grid:
            Xg = Xs.copy(); Xg[c] = g_
            vals.append(cal.predict(gbm.predict_proba(Xg)[:, 1]).mean())
        ax.plot(grid, vals, "o-", color=PALETTE[0], ms=3); ax.set_title(c, fontsize=10)
    fig.suptitle("Phụ thuộc riêng phần (PDP) của xác suất mua đã hiệu chỉnh", fontweight="bold")
    save_fig(fig, cfg, "fx21_forecast_pdp", "Đồ thị phụ thuộc riêng phần của mô hình dự báo theo sáu đặc trưng quan trọng")

    # e8. bảng tổng hợp kết quả theo giai đoạn
    pol_l = {r["Chính sách"]: r for r in M["milp"]["results"]["locked"]["policies"]}
    pol_o = {r["Chính sách"]: r for r in M["milp"]["results"]["oot"]["policies"]}
    cal_r = {(r["Hiệu chỉnh"], r["Tập"][:3]): r for r in M["forecast"]["calibration_results"]}
    sel = M["forecast"]["calibration_selected"]
    ut = {(r["Learner"], r["Tập"]): r for r in M["causal"]["uplift_table"]}
    best = M["causal"]["best_learner"]
    ate = [r for r in M["causal"]["ate_table"] if r["Tập"] == "Toàn bộ"][0]
    def vn(x, f=".4f"):
        return format(x, f).replace(",", "X").replace(".", ",").replace("X", ".")
    pm, po = pol_l["MILP (SCIP)"], pol_o["MILP (SCIP)"]
    e2e = [r for r in M["mlops"]["latency"] if r["Bước"] == "Đầu cuối (E2E)"][0]["P95 (ms)"]
    summ = pd.DataFrame([
        ["2. Dữ liệu", "Số khách / giao dịch", f"{vn(M['data_profile']['n_train_clients'], ',.0f')} / {vn(M['data_raw_rows']['transactions'], ',.0f')}", "Locked/OOT tách biệt"],
        ["3. Dự báo", "AUC (Locked / OOT)", f"{vn(cal_r[(sel, 'Loc')]['AUC'])} / {vn(cal_r[(sel, 'Out')]['AUC'])}", "Đạt cổng G1"],
        ["3. Hiệu chỉnh", "ECE (Locked / OOT)", f"{vn(cal_r[(sel, 'Loc')]['ECE'])} / {vn(cal_r[(sel, 'Out')]['ECE'])}", "Không đạt cổng G2 trên OOT"],
        ["3. Bất định", "Độ phủ CQR (Locked / OOT)", " / ".join(vn(r["Độ phủ"], ".3f") for r in M["forecast"]["cqr_results"] if r["Phương pháp"] == "CQR"), "Sát mục tiêu 0,90"],
        ["4. Nhân quả", "ATE toàn bộ [KTC95]", f"{vn(ate['Giá trị'])} [{vn(ate['KTC95_dưới'])}; {vn(ate['KTC95_trên'])}]", "Vững trước kiểm định phản bác"],
        ["4. Uplift", f"Qini {best} (Val / Locked / OOT)", f"{vn(ut[(best, 'val')]['Qini_norm'])} / {vn(ut[(best, 'locked')]['Qini_norm'])} / {vn(ut[(best, 'oot')]['Qini_norm'])}", "Thứ hạng không ổn định"],
        ["5. SPO+", "Regret chuẩn hóa Val (SPO+)", vn(M["spo"]["spo_best_val_regret"]), "Không khác biệt có ý nghĩa"],
        ["6. MILP", "Lợi nhuận DR Locked [KTC95] (RUB)", f"{vn(pm['Lợi nhuận tăng thêm DR (RUB)'], ',.0f')} [{vn(pm['KTC95 dưới'], ',.0f')}; {vn(pm['KTC95 trên'], ',.0f')}]", "Chứa 0 – không đạt G4"],
        ["6. MILP", "Lợi nhuận DR OOT [KTC95] (RUB)", f"{vn(po['Lợi nhuận tăng thêm DR (RUB)'], ',.0f')} [{vn(po['KTC95 dưới'], ',.0f')}; {vn(po['KTC95 trên'], ',.0f')}]", "Dương có ý nghĩa"],
        ["7. Router", "Kiểm thử đạt / gửi trùng", f"{M['router']['unit_tests_pass']}/{M['router']['unit_tests_total']} / 0", "Rào chắn R1 làm giảm giá trị"],
        ["8. MLOps", "P95 độ trễ E2E (ms)", vn(e2e, ".3f"), "Đạt cổng G5"],
        ["8. MLOps", "Cổng phát hành đạt", f"{M['mlops']['gates_pass']}/{M['mlops']['gates_total']}", "Khuyến nghị triển khai có đối chứng"],
    ], columns=["Giai đoạn", "Chỉ số chính", "Kết quả", "Đánh giá"])
    save_table(summ, cfg, "tx24_results_summary", "Tổng hợp kết quả chính theo chu trình tám giai đoạn")

    # =============================================================== (f) BỔ SUNG 2
    # f1. CATE dự báo theo phân khúc cho từng learner so với uplift quan sát (Locked + OOT)
    ev2 = seg[seg.split.isin(["locked", "oot"])]
    rows = []
    for col, nm in [("age_band", "Nhóm tuổi"), ("freq_q", "Tần suất mua")]:
        for k, g in ev2.groupby(col):
            if k == "Không rõ":
                continue
            u, se = neyman(g["target"].values.astype(float), g["treatment_flg"].values.astype(float))
            r = {"Chiều": nm, "Nhóm": k, "Uplift quan sát": u, "SE": se}
            for ln in LEARNERS:
                r[ln] = ct.loc[g.index, ln].mean()
            rows.append(r)
    cs = pd.DataFrame(rows)
    save_table(cs, cfg, "tx25_segment_cate_by_learner", "CATE dự báo trung bình theo phân khúc của từng learner so với uplift quan sát (Locked + OOT)")
    rows = []
    for ln in LEARNERS:
        rows.append({"Learner": ln, "Tương quan với uplift quan sát": np.corrcoef(cs[ln], cs["Uplift quan sát"])[0, 1],
                     "Sai số tuyệt đối TB": np.mean(np.abs(cs[ln] - cs["Uplift quan sát"]))})
    save_table(pd.DataFrame(rows), cfg, "tx26_segment_cate_agreement", "Mức khớp giữa CATE dự báo và uplift quan sát ở mức phân khúc")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (nm, g) in zip(axes, cs.groupby("Chiều", sort=False)):
        x = np.arange(len(g))
        ax.errorbar(x, g["Uplift quan sát"], yerr=1.96 * g["SE"], fmt="ko", capsize=3, label="Quan sát", zorder=5)
        for i, ln in enumerate(LEARNERS):
            ax.plot(x, g[ln], "-", marker=".", color=PALETTE[i], label=ln, alpha=.8)
        ax.set_xticks(x); ax.set_xticklabels(g["Nhóm"]); ax.axhline(0, color="k", lw=1); ax.set_title(nm)
    axes[0].legend(fontsize=7)
    save_fig(fig, cfg, "fx22_segment_cate_by_learner", "CATE dự báo theo phân khúc của sáu learner so với uplift quan sát")

    # f2. hồ sơ các luồng hành động và đồng xuất hiện quy tắc (Locked)
    logl = pd.read_csv(proc / "decision_log_locked.csv", index_col=0, dtype={0: str})
    logl.index = logl.index.astype(str)
    lk2 = camp.loc[logl.index]
    rows = []
    for a in ["AUTO_SEND", "HUMAN_REVIEW", "DEFER", "SUPPRESS", "NO_ACTION"]:
        ii = logl.index[logl["action"] == a]
        g = lk2.loc[ii]
        u, se = neyman(g["target"].values.astype(float), g["treatment_flg"].values.astype(float)) if len(g) > 50 else (np.nan, np.nan)
        rows.append({"Hành động": a, "Số khách": len(g), "Tuổi TB": g["age"].mean(), "Recency TB": g["recency"].mean(),
                     "Số giao dịch TB": g["n_tx_all"].mean(), "Giá trị giỏ TB (RUB)": g["aov_all"].mean(),
                     "Uplift quan sát": u, "SE": se})
    save_table(pd.DataFrame(rows), cfg, "tx27_action_profile", "Hồ sơ khách hàng và uplift quan sát theo luồng hành động (Locked Test)")
    rules = ["R1_SLEEPING_DOG", "R2_RECENT_PURCHASE", "R3_VIP", "R4_HIGH_UNCERTAINTY"]
    co = pd.DataFrame(0, index=rules, columns=rules)
    rs = logl["reasons"].fillna("")
    for r1 in rules:
        for r2 in rules:
            co.loc[r1, r2] = int((rs.str.contains(r1) & rs.str.contains(r2)).sum())
    save_table(co.reset_index().rename(columns={"index": "Quy tắc"}), cfg, "tx28_rule_cooccurrence", "Ma trận đồng kích hoạt giữa các quy tắc nghiệp vụ (Locked Test)")

    # f3. PSI giữa các snapshot liên tiếp + độ tin cậy theo snapshot
    from s8_mlops import psi as psi_fn
    rows = []
    for c in feats:
        rows.append({"Đặc trưng": c, "PSI S1→S2": psi_fn(fc.loc[fc.snapshot == "S1", c], fc.loc[fc.snapshot == "S2", c]),
                     "PSI S2→S3": psi_fn(fc.loc[fc.snapshot == "S2", c], fc.loc[fc.snapshot == "S3", c])})
    ps = pd.DataFrame(rows).sort_values("PSI S2→S3", ascending=False)
    save_table(ps.head(15), cfg, "tx29_psi_consecutive", "PSI giữa các snapshot liên tiếp (15 đặc trưng dịch chuyển mạnh nhất)")
    fig, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for i, (nm, d) in enumerate([("S1 (train, trong mẫu)", tr), ("S2 (Locked)", lk), ("S3 (OOT)", oot)]):
        p = cal.predict(gbm.predict_proba(d[feats])[:, 1]); y = d["y_buy"].values
        q = pd.qcut(p, 15, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
        ax.plot(g["p"], g["y"], "o-", color=PALETTE[i], label=f"{nm} ECE={ece_score(y, p)[0]:.4f}")
        rows.append({"Snapshot": nm, "Tỷ lệ mua": y.mean(), "Xác suất TB": p.mean(), "AUC": roc_auc_score(y, p), "ECE": ece_score(y, p)[0]})
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.legend(); ax.set_xlabel("Xác suất dự báo"); ax.set_ylabel("Tần suất quan sát")
    ax.set_title("Độ tin cậy của mô hình qua ba snapshot")
    save_fig(fig, cfg, "fx23_reliability_snapshots", "Biểu đồ độ tin cậy của mô hình đã hiệu chỉnh qua ba snapshot")
    save_table(pd.DataFrame(rows), cfg, "tx30_snapshot_performance", "Hiệu năng của mô hình đã hiệu chỉnh qua ba snapshot")

    # f4. truy vết một số khách hàng qua toàn bộ pipeline (Locked Test)
    pick = []
    for a in ["AUTO_SEND", "HUMAN_REVIEW", "DEFER", "SUPPRESS", "NO_ACTION"]:
        ii = logl.index[logl["action"] == a]
        if len(ii):
            pick.append(ii[0])
    rows = []
    for cid in pick:
        r = {"Mã khách hàng": cid, "Tuổi": camp.loc[cid, "age"], "Recency (ngày)": camp.loc[cid, "recency"],
             "Số giao dịch": camp.loc[cid, "n_tx_all"], "Giỏ TB (RUB)": camp.loc[cid, "aov_all"],
             "P(mua 14 ngày)": fcc.loc[cid, "p_buy_cal"], "Chi tiêu Ŝ (RUB)": ds.loc[cid, "spend_cap"],
             "τ̂ (learner chọn)": ct.loc[cid, "tau_hat"], "σ̂ giữa learner": ct.loc[cid, "tau_std"],
             "V̂ (RUB)": ds.loc[cid, "value_hat"], "MILP x": int(logl.loc[cid, "x"]), "Hành động": logl.loc[cid, "action"],
             "Mã lý do": logl.loc[cid, "reasons"], "Khóa lũy đẳng": logl.loc[cid, "idempotency_key"]}
        rows.append(r)
    tr_ = pd.DataFrame(rows).set_index("Mã khách hàng").T.reset_index().rename(columns={"index": "Thuộc tính"})
    tr_.columns = ["Thuộc tính"] + [f"KH {i + 1}" for i in range(len(pick))]
    tr_ = tr_.astype(str)
    for c in tr_.columns[1:]:
        tr_[c] = tr_[c].map(lambda v: (f"{float(v):.4f}".rstrip("0").rstrip(".").replace(".", ",") if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v))
    save_table(tr_, cfg, "tx31_customer_trace", "Truy vết năm khách hàng (mỗi luồng hành động một khách) qua toàn bộ pipeline (Locked Test)")

    save_metrics(cfg, "extended", m)
    return m


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
