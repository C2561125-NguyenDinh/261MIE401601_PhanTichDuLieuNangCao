"""Giai đoạn 8 - Triển khai, đo lường thực chứng và giám sát hệ thống (MLOps).

- Đóng băng mô hình: băm SHA-256 mô hình + dữ liệu, ma trận tương thích phiên bản.
- Đo độ trễ đầu cuối (P50/P95/P99) của luồng chấm điểm trực tuyến theo từng bước.
- Giám sát trôi dạt: PSI đặc trưng, Brier/ECE theo thời gian, Action TVD, độ trễ nhãn.
- Đối sánh Current vs Challenger sau tái huấn luyện; cổng phát hành (Release Gates).
"""
from __future__ import annotations

import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from s2_features import feature_cols, load_campaign, load_snapshots
from s3_forecast import cls_metrics, ece_score
from s4_causal import UpliftLearners, qini_auc
from utils import (PALETTE, load_config, load_metrics, log, save_fig, save_metrics, save_table, set_seed,
                   setup_mpl, sha256_file)


def psi(expected, actual, bins=10):
    e = pd.Series(expected).dropna().values
    a = pd.Series(actual).dropna().values
    if len(e) == 0 or len(a) == 0:
        return np.nan
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        qs = np.unique(np.r_[e.min() - 1e-9, np.median(e), e.max() + 1e-9])
    qs[0], qs[-1] = -np.inf, np.inf
    pe = np.histogram(e, qs)[0] / len(e)
    pa = np.histogram(a, qs)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    proc, models_dir = cfg["paths"]["processed"], cfg["paths"]["models"]
    mo = cfg["mlops"]
    M = load_metrics(cfg)
    camp = load_campaign(cfg)
    fc = load_snapshots(cfg)
    feats = feature_cols(camp)
    m: dict = {}

    # ---------------- 1. đóng băng & checksum
    rows = []
    for p in sorted(list(models_dir.glob("*.joblib")) + list(proc.glob("*.parquet")) + list(proc.glob("*.csv"))):
        rows.append({"Tệp": f"{p.parent.name}/{p.name}", "Kích thước (MB)": p.stat().st_size / 2**20, "SHA-256": sha256_file(p)})
    cks = pd.DataFrame(rows)
    save_table(cks, cfg, "t41_artifact_checksums", "Bảng mã băm SHA-256 của mô hình và dữ liệu đã xử lý (đóng băng phiên bản)")
    pkgs = ["numpy", "pandas", "scikit-learn", "lightgbm", "scipy", "ortools", "doubleml", "matplotlib", "pyarrow",
            "python-docx", "PyYAML", "joblib"]
    vrows = [{"Thành phần": "Python", "Phiên bản đã kiểm thử": platform.python_version(), "Ghi chú": platform.platform()}]
    for p in pkgs:
        try:
            v = metadata.version(p)
        except Exception:
            v = "không cài"
        vrows.append({"Thành phần": p, "Phiên bản đã kiểm thử": v, "Ghi chú": ""})
    save_table(pd.DataFrame(vrows), cfg, "t42_version_matrix", "Ma trận tương thích phiên bản môi trường thực thi")

    # ---------------- 2. độ trễ chấm điểm trực tuyến
    fmod = joblib.load(models_dir / "forecast.joblib")
    L = joblib.load(models_dir / "uplift_learners.joblib")
    best = M["causal"]["best_learner"]
    bz, rc = cfg["business"], cfg["router"]
    gbm_b = fmod["gbm"].booster_
    mean_b = fmod["mean_model"].booster_
    cal = fmod["calibrator"]
    ub = {"S-learner": [L.s.booster_], "T-learner": [L.m1.booster_, L.m0.booster_], "X-learner": [L.x0.booster_, L.x1.booster_],
          "R-learner": [L.r.booster_], "DR-learner": [L.dr.booster_], "Class Transformation": [L.ct.booster_]}[best]
    m0b = L.m0.booster_
    store = camp[feats].astype("float32")
    keys = list(store.index)
    arr = store.values
    pos = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(cfg["seed"])
    th_unc = M["router"]["thresholds"]["unc"]; th_vip = M["router"]["thresholds"]["vip"]
    e = L.e

    def tau_fn(x):
        if best == "T-learner":
            return ub[0].predict(x)[0] - ub[1].predict(x)[0]
        if best == "S-learner":
            return ub[0].predict(np.c_[x, [[1]]])[0] - ub[0].predict(np.c_[x, [[0]]])[0]
        if best == "X-learner":
            return e * ub[0].predict(x)[0] + (1 - e) * ub[1].predict(x)[0]
        if best == "Class Transformation":
            return 2 * ub[0].predict(x)[0] - 1
        return ub[0].predict(x)[0]

    lat = []
    for _ in range(mo["latency_requests"]):
        cid = keys[rng.integers(0, len(keys))]
        t0 = time.perf_counter()
        x = arr[pos[cid]].reshape(1, -1)                       # tra cứu feature store
        t1 = time.perf_counter()
        p = float(cal.predict(gbm_b.predict(x))[0])            # dự báo + hiệu chỉnh
        s_hat = float(np.expm1(mean_b.predict(x)[0]))           # chi tiêu kỳ vọng
        t2 = time.perf_counter()
        tau = float(tau_fn(x)); p0 = float(m0b.predict(x)[0])   # uplift
        t3 = time.perf_counter()
        v = tau * bz["margin_rate"] * s_hat - bz["contact_cost"] - (p0 + tau) * bz["incentive_cost"]
        action = "SUPPRESS" if tau < rc["sleeping_dog_tau"] else ("AUTO_SEND" if v > 0 else "NO_ACTION")
        t4 = time.perf_counter()
        lat.append({"Tra cứu đặc trưng": t1 - t0, "Dự báo + hiệu chỉnh": t2 - t1, "Uplift": t3 - t2,
                    "Giá trị + định tuyến": t4 - t3, "Đầu cuối (E2E)": t4 - t0})
    lat = pd.DataFrame(lat) * 1000
    lt = pd.DataFrame({"Bước": lat.columns, "P50 (ms)": lat.quantile(.5).values, "P95 (ms)": lat.quantile(.95).values,
                       "P99 (ms)": lat.quantile(.99).values, "Trung bình (ms)": lat.mean().values, "Max (ms)": lat.max().values})
    budget = {"Tra cứu đặc trưng": 5, "Dự báo + hiệu chỉnh": 15, "Uplift": 15, "Giá trị + định tuyến": 5, "Đầu cuối (E2E)": mo["gates"]["max_p95_latency_ms"]}
    lt["Ngân sách P95 (ms)"] = lt["Bước"].map(budget)
    lt["Trong ngân sách"] = lt["P95 (ms)"] <= lt["Ngân sách P95 (ms)"]
    save_table(lt, cfg, "t43_latency", f"Độ trễ chấm điểm trực tuyến theo bước ({mo['latency_requests']} yêu cầu) và ngân sách độ trễ")
    m["latency"] = lt.to_dict(orient="records")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(lat["Đầu cuối (E2E)"].clip(upper=lat["Đầu cuối (E2E)"].quantile(.995)), bins=60, color=PALETTE[0])
    for q, c in [(.5, "k"), (.95, PALETTE[1]), (.99, PALETTE[6])]:
        v = lat["Đầu cuối (E2E)"].quantile(q)
        ax[0].axvline(v, color=c, ls="--", label=f"P{int(q * 100)} = {v:.2f} ms")
    ax[0].legend(); ax[0].set_xlabel("ms"); ax[0].set_title("Phân phối độ trễ đầu cuối")
    lt2 = lt[lt["Bước"] != "Đầu cuối (E2E)"]
    ax[1].bar(lt2["Bước"], lt2["P95 (ms)"], color=PALETTE[2]); ax[1].set_title("P95 theo bước"); ax[1].tick_params(axis="x", rotation=20)
    save_fig(fig, cfg, "f40_latency", "Phân phối độ trễ đầu cuối và P95 theo từng bước của luồng chấm điểm")

    # ---------------- 3. trôi dạt dữ liệu (PSI)
    tr = camp[camp.split == "train"]
    rows = []
    s1, s3 = fc[fc.snapshot == "S1"], fc[fc.snapshot == "S3"]
    for c in feats:
        rows.append({"Đặc trưng": c, "PSI Train→Locked": psi(tr[c], camp.loc[camp.split == "locked", c]),
                     "PSI Train→OOT cohort": psi(tr[c], camp.loc[camp.split == "oot", c]),
                     "PSI S1→S3 (thời gian)": psi(s1[c], s3[c])})
    pt = pd.DataFrame(rows)

    def status(v):
        return "ALERT" if v >= mo["psi_alert"] else ("WARN" if v >= mo["psi_warn"] else "OK")
    for c in ["PSI Train→Locked", "PSI Train→OOT cohort", "PSI S1→S3 (thời gian)"]:
        pt["Trạng thái " + c.split(" ", 1)[1]] = pt[c].map(status)
    pt = pt.sort_values("PSI S1→S3 (thời gian)", ascending=False)
    save_table(pt, cfg, "t44_psi_drift", "Chỉ số ổn định quần thể (PSI) của các đặc trưng theo ba phép so sánh")
    m["psi"] = {"n_alert_oot": int((pt["PSI Train→OOT cohort"] >= mo["psi_alert"]).sum()),
                "n_warn_oot": int(((pt["PSI Train→OOT cohort"] >= mo["psi_warn"]) & (pt["PSI Train→OOT cohort"] < mo["psi_alert"])).sum()),
                "n_alert_time": int((pt["PSI S1→S3 (thời gian)"] >= mo["psi_alert"]).sum()),
                "n_warn_time": int(((pt["PSI S1→S3 (thời gian)"] >= mo["psi_warn"]) & (pt["PSI S1→S3 (thời gian)"] < mo["psi_alert"])).sum()),
                "max_locked": float(pt["PSI Train→Locked"].max()),
                "top_time": pt.head(5)[["Đặc trưng", "PSI S1→S3 (thời gian)"]].to_dict(orient="records"),
                "top_oot": pt.sort_values("PSI Train→OOT cohort", ascending=False).head(5)[["Đặc trưng", "PSI Train→OOT cohort"]].to_dict(orient="records")}
    fig, ax = plt.subplots(figsize=(10, 8))
    top = pt.head(25).iloc[::-1]
    yy = np.arange(len(top))
    ax.barh(yy - .27, top["PSI Train→Locked"], .27, color=PALETTE[2], label="Train→Locked")
    ax.barh(yy, top["PSI Train→OOT cohort"], .27, color=PALETTE[1], label="Train→OOT cohort")
    ax.barh(yy + .27, top["PSI S1→S3 (thời gian)"], .27, color=PALETTE[0], label="S1→S3 (thời gian)")
    ax.set_yticks(yy); ax.set_yticklabels(top["Đặc trưng"], fontsize=8)
    ax.axvline(mo["psi_warn"], color="orange", ls="--", label="Ngưỡng cảnh báo"); ax.axvline(mo["psi_alert"], color="r", ls="--", label="Ngưỡng báo động")
    ax.legend(fontsize=8); ax.set_title("PSI của 25 đặc trưng trôi dạt mạnh nhất")
    save_fig(fig, cfg, "f41_psi", "PSI của các đặc trưng trôi dạt mạnh nhất theo ba phép so sánh")

    # ---------------- 4. trôi dạt dự báo + Action TVD
    cr = pd.DataFrame(M["forecast"]["calibration_results"])
    sel = M["forecast"]["calibration_selected"]
    pred_drift = cr[cr["Hiệu chỉnh"].isin(["none", sel])][["Hiệu chỉnh", "Tập", "AUC", "Brier", "ECE", "mean_p", "base_rate"]]
    save_table(pred_drift, cfg, "t45_prediction_drift", "Trôi dạt hiệu năng dự báo giữa Locked Test (S2) và Out-of-Time (S3)")
    dl, do = M["router"]["locked"]["distribution"], M["router"]["oot"]["distribution"]
    nl, no = sum(dl.values()), sum(do.values())
    tvd = 0.5 * sum(abs(dl[a] / nl - do[a] / no) for a in dl)
    m["action_tvd"] = tvd
    qini_l = [r for r in M["causal"]["uplift_table"] if r["Learner"] == best and r["Tập"] == "locked"][0]["Qini_norm"]
    qini_o = [r for r in M["causal"]["uplift_table"] if r["Learner"] == best and r["Tập"] == "oot"][0]["Qini_norm"]
    m["qini_locked"], m["qini_oot"] = qini_l, qini_o

    # ---------------- 5. độ trễ nhãn
    s3b = s3[s3.y_buy == 1]
    H = cfg["temporal"]["horizon_days"]
    days = np.arange(1, H + 1)
    frac = [(s3b["days_to_first"] < d).mean() for d in days]
    rows = []
    gbm = fmod["gbm"]
    p3 = cal.predict(gbm.predict_proba(s3[feats])[:, 1])
    for d in sorted(set([1, 3, 5, 7, 10, H])):
        yd = ((s3["y_buy"] == 1) & (s3["days_to_first"] < d)).astype(int).values
        r = cls_metrics(yd, p3)
        rows.append({"Số ngày đã chờ nhãn": d, "Tỷ lệ nhãn dương đã quan sát": float((s3b["days_to_first"] < d).mean()),
                     "Tỷ lệ mua quan sát": yd.mean(), "AUC đo được": r["AUC"], "ECE đo được": r["ECE"], "Brier đo được": r["Brier"]})
    ld = pd.DataFrame(rows)
    save_table(ld, cfg, "t46_label_delay", "Ảnh hưởng của độ trễ nhãn lên các chỉ số giám sát (snapshot S3)")
    m["label_delay"] = ld.to_dict(orient="records")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(days, frac, "o-", color=PALETTE[0], ms=3); ax[0].set_xlabel("Số ngày sau mốc cắt"); ax[0].set_ylabel("Tỷ lệ nhãn dương đã đến")
    ax[0].set_title("Đường cong đến của nhãn")
    ax[1].plot(ld["Số ngày đã chờ nhãn"], ld["ECE đo được"], "o-", color=PALETTE[1], label="ECE")
    ax2 = ax[1].twinx(); ax2.plot(ld["Số ngày đã chờ nhãn"], ld["AUC đo được"], "s-", color=PALETTE[0], label="AUC"); ax2.grid(False)
    ax[1].set_xlabel("Số ngày đã chờ nhãn"); ax[1].set_ylabel("ECE"); ax2.set_ylabel("AUC"); ax[1].set_title("Chỉ số đo được khi nhãn đến trễ")
    save_fig(fig, cfg, "f42_label_delay", "Độ trễ nhãn và sai lệch của chỉ số giám sát")

    # ---------------- 6. Current vs Challenger
    log.info("Tái huấn luyện Challenger ...")
    va = fc[(fc.snapshot == "S2") & (fc.split == "val")]
    oot = fc[(fc.snapshot == "S3") & (fc.split.isin(["locked", "oot"]))]
    cur_p = cal.predict(gbm.predict_proba(oot[feats])[:, 1])
    trc = pd.concat([fc[(fc.snapshot == "S1") & (fc.split == "train")], fc[(fc.snapshot == "S2") & (fc.split == "train")]])
    va_c = fc[(fc.snapshot == "S2") & (fc.split == "val")]
    ch = lgb.LGBMClassifier(**{**cfg["lgbm"], "n_estimators": M["forecast"]["lgbm_best_iter"]}, random_state=cfg["seed"]).fit(trc[feats], trc["y_buy"])
    # Challenger chỉ thấy S2 của tập train nên hiệu chỉnh trên S2-val vẫn ngoài mẫu
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(ch.predict_proba(va_c[feats])[:, 1], va_c["y_buy"])
    ch_p = iso.predict(ch.predict_proba(oot[feats])[:, 1])
    rows = []
    for nm, p in [("Current (train S1)", cur_p), ("Challenger (train S1+S2)", ch_p)]:
        rows.append({"Thành phần": "Dự báo mua hàng", "Mô hình": nm, **{k: v for k, v in cls_metrics(oot["y_buy"].values, p).items() if k in ["AUC", "Brier", "ECE", "LogLoss"]}})
    tr_u = camp[camp.split == "train"]; trv_u = camp[camp.split.isin(["train", "val"])]
    oot_u = camp[camp.split == "oot"]
    L2 = UpliftLearners(cfg, feats, float(trv_u.treatment_flg.mean())).fit(trv_u, trv_u["target"].values.astype(float), trv_u["treatment_flg"].values.astype(float))
    u_cur = L.predict_all(oot_u)[best].values
    u_ch = L2.predict_all(oot_u)[best].values
    yo, to = oot_u["target"].values.astype(float), oot_u["treatment_flg"].values.astype(float)
    rows.append({"Thành phần": "Uplift", "Mô hình": f"Current ({best}, train)", "Qini_norm": qini_auc(yo, to, u_cur)})
    rows.append({"Thành phần": "Uplift", "Mô hình": f"Challenger ({best}, train+val)", "Qini_norm": qini_auc(yo, to, u_ch)})
    cc = pd.DataFrame(rows)
    save_table(cc, cfg, "t47_champion_challenger", "Đối sánh Current và Challenger sau tái huấn luyện trên dữ liệu Out-of-Time")
    m["champion_challenger"] = cc.to_dict(orient="records")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (nm, p) in enumerate([("Current", cur_p), ("Challenger", ch_p)]):
        q = pd.qcut(p, 15, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": oot["y_buy"].values, "q": q}).groupby("q", observed=True).mean()
        ax[0].plot(g["p"], g["y"], "o-", color=PALETTE[i], label=f"{nm} ECE={ece_score(oot['y_buy'].values, p)[0]:.4f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].legend(); ax[0].set_title("Hiệu chỉnh trên OOT (S3)")
    from s4_causal import qini_curve
    for i, (nm, u) in enumerate([("Current", u_cur), ("Challenger", u_ch)]):
        x, q = qini_curve(yo, to, u); ax[1].plot(x / x[-1], q, color=PALETTE[i], label=f"{nm} Qini={qini_auc(yo, to, u):.3f}")
    ax[1].plot([0, 1], [0, q[-1]], "k--", lw=1); ax[1].legend(); ax[1].set_title("Qini trên cohort OOT")
    save_fig(fig, cfg, "f43_champion_challenger", "Đối sánh Current và Challenger: hiệu chỉnh xác suất và đường Qini")

    # ---------------- 7. Release gates
    g = mo["gates"]
    cal_oot = [r for r in M["forecast"]["calibration_results"] if r["Hiệu chỉnh"] == sel and r["Tập"].startswith("Out")][0]
    pol = pd.DataFrame(M["milp"]["results"]["locked"]["policies"])
    milp_row = pol[pol["Chính sách"] == "MILP (SCIP)"].iloc[0]
    p95 = float(lt.loc[lt["Bước"] == "Đầu cuối (E2E)", "P95 (ms)"].iloc[0])
    gates = [
        ("G1 Phân biệt dự báo (AUC OOT)", f"≥ {g['min_auc']}", cal_oot["AUC"], cal_oot["AUC"] >= g["min_auc"]),
        ("G2 Hiệu chỉnh (ECE OOT)", f"≤ {g['max_ece']}", cal_oot["ECE"], cal_oot["ECE"] <= g["max_ece"]),
        ("G3 Uplift ngoài mẫu (Qini OOT)", f"> {g['min_qini_pos']}", qini_o, qini_o > g["min_qini_pos"]),
        ("G4 Giá trị chính sách MILP (KTC95 dưới, Locked)", f"> {g['min_policy_lift']} RUB", milp_row["KTC95 dưới"], milp_row["KTC95 dưới"] > g["min_policy_lift"]),
        ("G5 Độ trễ E2E P95", f"≤ {g['max_p95_latency_ms']} ms", p95, p95 <= g["max_p95_latency_ms"]),
        ("G6 Số đặc trưng PSI báo động (OOT cohort)", f"≤ {g['max_psi_alert_features']}", m["psi"]["n_alert_oot"], m["psi"]["n_alert_oot"] <= g["max_psi_alert_features"]),
        ("G7 Kiểm thử Action Router", "100% PASS", f"{M['router']['unit_tests_pass']}/{M['router']['unit_tests_total']}", M["router"]["unit_tests_pass"] == M["router"]["unit_tests_total"]),
        ("G8 Idempotency (không gửi trùng)", "0 bản gửi trùng", M["router"]["idempotency_test"]["sends"] - M["router"]["idempotency_test"]["requests"] // 2,
         M["router"]["idempotency_test"]["sends"] == M["router"]["idempotency_test"]["requests"] // 2),
        ("G9 Nghiệm MILP khả thi & tối ưu", "OPTIMAL/FEASIBLE", M["milp"]["results"]["locked"]["ip"]["status"], M["milp"]["results"]["locked"]["ip"]["status"] in ("OPTIMAL", "FEASIBLE")),
    ]
    gt = pd.DataFrame(gates, columns=["Cổng phát hành", "Tiêu chí", "Giá trị đo được", "Đạt"])
    gt["Kết quả"] = np.where(gt["Đạt"], "PASS", "FAIL")
    gt["Giá trị đo được"] = gt["Giá trị đo được"].astype(str)
    save_table(gt.drop(columns="Đạt"), cfg, "t48_release_gates", "Kiểm định các cổng phát hành (Release Gates)")
    m["gates"] = gt.drop(columns="Đạt").to_dict(orient="records")
    m["gates_pass"] = int(gt["Đạt"].sum()); m["gates_total"] = len(gt)

    retrain = pd.DataFrame([
        ["Định kỳ", "Mỗi 28 ngày (bằng kỳ dự báo H)", "Huấn luyện Challenger trên cửa sổ mới, so với Current trên OOT"],
        ["Trôi dạt đặc trưng", f"≥ {g['max_psi_alert_features']} đặc trưng PSI ≥ {mo['psi_alert']}", "Tái huấn luyện sớm + chuyển AUTO_SEND sang duyệt"],
        ["Suy giảm hiệu chỉnh", f"ECE > {g['max_ece']} khi đã đủ nhãn", "Hiệu chỉnh lại trên cửa sổ validation mới"],
        ["Suy giảm uplift", "Qini OOT ≤ 0 hoặc KTC giá trị chính sách chứa 0", "Dừng nhắm mục tiêu, quay về thí nghiệm ngẫu nhiên có nhóm đối chứng"],
        ["Trôi dạt chính sách", "Action TVD > 0,10 giữa hai kỳ", "Rà soát quy tắc nghiệp vụ và phân phối V̂"],
        ["Thăng hạng Challenger", "Thắng Current trên ≥ 2/3 chỉ số chính, không vi phạm cổng nào", "Triển khai shadow → canary 10% → 100%"],
    ], columns=["Loại kích hoạt", "Điều kiện", "Hành động"])
    save_table(retrain, cfg, "t49_retraining_policy", "Chính sách tái huấn luyện và thăng hạng mô hình")
    save_metrics(cfg, "mlops", m)
    return m


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
