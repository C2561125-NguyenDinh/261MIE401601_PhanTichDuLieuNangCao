"""Giai đoạn bổ trợ 11 - Đối chứng bằng thư viện chuẩn, giải thích nhân quả và học chính sách.

  (A) EconML: CausalForestDML (Athey, Tibshirani & Wager, 2019; Chernozhukov et al., 2018) huấn luyện trên Train,
      đánh giá Qini/AUUC trên Validation, Locked Test, OOT; so sánh với sáu meta-learner tự cài đặt; khoảng tin cậy CATE.
  (B) SHAP (Lundberg & Lee, 2017; Lundberg et al., 2020): TreeSHAP cho mô hình CATE (DR-learner, giai đoạn cuối)
      để giải thích đặc trưng nào dẫn dắt tác động biên; so sánh với tầm quan trọng gain.
  (C) Học chính sách (Athey & Wager, 2021; Zhou, Athey & Wager, 2023): cây chính sách trên điểm thưởng kép vững
      (lợi nhuận tăng thêm DR), chọn độ sâu trên Validation; đánh giá giá trị DR trên Locked Test và OOT,
      bootstrap ghép cặp với MILP.
Mô-đun chỉ đọc tạo tác của giai đoạn 2-6; mọi lựa chọn được thực hiện trên Train/Validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s2_features import feature_cols, load_campaign  # noqa: E402
from s4_causal import eval_uplift, qini_curve  # noqa: E402
from s6_milp import feasible  # noqa: E402
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl  # noqa: E402

FEAT_VI = {
    "recency": "Số ngày từ lần mua gần nhất", "n_tx_all": "Số giao dịch (cửa sổ)", "aov_all": "Giá trị giỏ TB",
    "tenure_days": "Thâm niên thẻ (ngày)", "age": "Tuổi", "share_spent_tx": "Tỷ lệ giao dịch có tiêu điểm",
    "sum_all": "Tổng chi tiêu (cửa sổ)", "has_redeemed": "Đã từng đổi điểm",
}


def _load_names(cfg):
    p = cfg["paths"]["tables"] / "t08_feature_dictionary.csv"
    if p.exists():
        d = pd.read_csv(p, encoding="utf-8-sig")
        FEAT_VI.update(dict(zip(d["Đặc trưng"], d["Mô tả"])))


def _vn(name):
    return FEAT_VI.get(name, name)


# ------------------------------------------------------------------ (A) EconML
def econml_benchmark(cfg, plt, camp, feats, ct):
    from econml.dml import CausalForestDML
    from lightgbm import LGBMClassifier
    from sklearn.dummy import DummyClassifier

    tr = camp.split == "train"
    X, y, t = camp.loc[tr, feats], camp.loc[tr, "target"].astype(float).values, camp.loc[tr, "treatment_flg"].astype(int).values
    log.info("EconML CausalForestDML ...")
    est = CausalForestDML(
        model_y=LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                               random_state=cfg["seed"], verbose=-1),
        model_t=DummyClassifier(strategy="prior"),   # RCT: xác suất can thiệp là hằng số đã biết
        discrete_outcome=True, discrete_treatment=True, n_estimators=400, min_samples_leaf=200,
        max_samples=0.45, honest=True, cv=3, random_state=cfg["seed"], n_jobs=2)
    est.fit(y, t, X=X)
    rows, preds = [], {}
    tab = pd.read_csv(cfg["paths"]["tables"] / "t19_uplift_learners.csv", encoding="utf-8-sig")
    sp_map = {"val": "val", "locked": "locked", "oot": "oot"}
    for s in ["val", "locked", "oot"]:
        d = camp.split == s
        u = np.ravel(est.effect(camp.loc[d, feats]))
        preds[s] = u
        r = eval_uplift(camp.loc[d, "target"].values.astype(float), camp.loc[d, "treatment_flg"].values.astype(float), u)
        lo, hi = (np.ravel(z) for z in est.effect_interval(camp.loc[d, feats], alpha=0.05))
        rows.append({"Tập": s, "Qini chuẩn hóa (CausalForestDML)": r["Qini_norm"], "AUUC": r["AUUC"],
                     "Uplift@10%": r["Uplift@10%"],
                     "Tương quan Spearman với learner đã chọn": pd.Series(u).corr(pd.Series(ct.loc[d, "tau_hat"].values), method="spearman"),
                     "Tỷ lệ khách có KTC95 của CATE không chứa 0": float(((lo > 0) | (hi < 0)).mean()),
                     "CATE TB": float(u.mean())})
    res = pd.DataFrame(rows)
    # so sánh với các learner tự cài đặt (Qini chuẩn hóa)
    comp = []
    for s in ["val", "locked", "oot"]:
        t19 = tab[tab["Tập"] == sp_map[s]] if "Tập" in tab.columns else None
        row = {"Tập": s, "CausalForestDML (EconML)": res.loc[res["Tập"] == s, "Qini chuẩn hóa (CausalForestDML)"].iloc[0]}
        if t19 is not None:
            for _, r in t19.iterrows():
                row[r["Learner"]] = r["Qini_norm"]
        comp.append(row)
    comp = pd.DataFrame(comp)
    save_table(res, cfg, "tx35_econml_causal_forest",
               "Kết quả CausalForestDML (EconML) trên ba tập: Qini, AUUC, tương quan với learner đã chọn và khoảng tin cậy CATE")
    save_table(comp, cfg, "tx36_econml_vs_learners",
               "Hệ số Qini chuẩn hóa: CausalForestDML (EconML) so với sáu meta-learner tự cài đặt")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, s in zip(axs, ["val", "locked", "oot"]):
        d = camp.split == s
        yy, tt = camp.loc[d, "target"].values.astype(float), camp.loc[d, "treatment_flg"].values.astype(float)
        for lab, u, c in (("CausalForestDML", preds[s], PALETTE[1]), ("Learner đã chọn", ct.loc[d, "tau_hat"].values, PALETTE[0])):
            x, q = qini_curve(yy, tt, u)
            ax.plot(x / x[-1], q, color=c, label=lab)
        ax.plot([0, 1], [0, q[-1]], "k--", lw=1, label="Ngẫu nhiên")
        ax.set_title({"val": "Validation", "locked": "Locked Test", "oot": "OOT"}[s]); ax.set_xlabel("Tỷ lệ khách được nhắm")
    axs[0].set_ylabel("Lượt mua tăng thêm (Qini)"); axs[0].legend(fontsize=8)
    save_fig(fig, cfg, "fx27_econml_qini", "Đường Qini của CausalForestDML (EconML) và learner đã chọn trên ba tập")
    joblib.dump(est, cfg["paths"]["models"] / "econml_causal_forest.joblib")
    return res, comp


# ------------------------------------------------------------------ (B) SHAP
def shap_cate(cfg, plt, camp, feats, L):
    import shap
    log.info("SHAP cho mô hình CATE ...")
    d = camp.split == "locked"
    Xl = camp.loc[d, feats]
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(Xl), size=min(5000, len(Xl)), replace=False)
    Xs = Xl.iloc[idx]
    expl = shap.TreeExplainer(L.dr)
    sv = expl.shap_values(Xs)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=feats)
    share = (mean_abs / mean_abs.sum()).sort_values(ascending=False)
    gain = pd.read_csv(cfg["paths"]["tables"] / "t23_cate_importance.csv", encoding="utf-8-sig").set_index("Đặc trưng")["gain_share"]
    # hướng tác động: tương quan giữa giá trị đặc trưng và SHAP
    direc = {f: pd.Series(Xs[f].values).corr(pd.Series(sv[:, i]), method="spearman") for i, f in enumerate(feats)}
    top = share.head(15)
    tab = pd.DataFrame({"Đặc trưng": top.index, "Diễn giải": [_vn(f) for f in top.index],
                        "Tỷ trọng SHAP tuyệt đối TB": top.values,
                        "Hạng theo SHAP": range(1, len(top) + 1),
                        "Hạng theo gain": [int(gain.rank(ascending=False).get(f, np.nan)) if f in gain.index else None for f in top.index],
                        "Tương quan giá trị – SHAP": [direc[f] for f in top.index]})
    save_table(tab, cfg, "tx37_shap_cate",
               "Giải thích mô hình CATE (DR-learner) bằng TreeSHAP trên 5.000 khách Locked Test: 15 đặc trưng quan trọng nhất")
    # beeswarm
    plt.figure()
    shap.summary_plot(sv, Xs, feature_names=[_vn(f) for f in feats], max_display=12, show=False, plot_size=(9, 5.5))
    fig = plt.gcf(); fig.axes[0].set_xlabel("Giá trị SHAP (đóng góp vào CATE ước lượng)")
    save_fig(fig, cfg, "fx28_shap_cate_beeswarm",
             "Biểu đồ SHAP của mô hình CATE: mỗi điểm là một khách hàng, màu thể hiện giá trị đặc trưng")
    # dependence cho 3 đặc trưng hàng đầu
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, f in zip(axs, share.index[:3]):
        i = feats.index(f)
        ax.scatter(Xs[f], sv[:, i], s=4, alpha=0.35, color=PALETTE[0])
        ax.axhline(0, color="k", lw=0.8); ax.set_xlabel(_vn(f)); ax.set_title(_vn(f), fontsize=10)
    axs[0].set_ylabel("Giá trị SHAP")
    save_fig(fig, cfg, "fx29_shap_dependence", "Quan hệ giữa giá trị đặc trưng và đóng góp SHAP vào CATE (ba đặc trưng hàng đầu)")
    rho = pd.Series(share).rank(ascending=False).corr(gain.reindex(share.index).rank(ascending=False), method="spearman")
    return tab, {"rank_corr_shap_gain": float(rho), "base_value": float(np.ravel(expl.expected_value)[0])}


# ------------------------------------------------------------------ (C) policy learning
def policy_learning(cfg, plt, camp, feats):
    from econml.policy import PolicyTree
    proc = cfg["paths"]["processed"]
    mc = cfg["milp"]
    ds = pd.read_parquet(proc / "decision_scores.parquet")
    rng = np.random.default_rng(cfg["seed"])
    # điểm thưởng DR: lợi nhuận tăng thêm (proxy) nếu gửi, 0 nếu không gửi
    R = ds["value_proxy"]
    tr, va = camp.split == "train", camp.split == "val"
    Xtr = camp.loc[tr, feats]
    grid = []
    log.info("Học cây chính sách ...")
    models = {}
    for depth in [1, 2, 3]:
        for leaf in [500, 2000]:
            pt = PolicyTree(max_depth=depth, min_samples_leaf=leaf, random_state=cfg["seed"])
            pt.fit(Xtr, np.c_[np.zeros(tr.sum()), R[tr].values])
            a = pt.predict(camp.loc[va, feats])
            val_dr = float((a * R[va].values).sum())
            grid.append({"Độ sâu": depth, "Số khách tối thiểu / lá": leaf, "Tỷ lệ được gửi (Validation)": float(a.mean()),
                         "Lợi nhuận DR trên Validation (RUB)": val_dr})
            models[(depth, leaf)] = pt
    grid = pd.DataFrame(grid)
    best = grid.sort_values("Lợi nhuận DR trên Validation (RUB)", ascending=False).iloc[0]
    bd, bl = int(best["Độ sâu"]), int(best["Số khách tối thiểu / lá"])
    pt = models[(bd, bl)]
    grid["Được chọn"] = (grid["Độ sâu"] == bd) & (grid["Số khách tối thiểu / lá"] == bl)
    save_table(grid, cfg, "tx38_policy_tree_selection",
               "Chọn cấu hình cây chính sách trên Validation theo lợi nhuận tăng thêm ước lượng kép vững")
    # quy tắc của cây
    rules = _tree_rules(pt, feats)
    save_table(rules, cfg, "tx39_policy_tree_rules", "Các quy tắc của cây chính sách được chọn (mỗi dòng là một lá)")

    rows = []
    for split in ["locked", "oot"]:
        d = camp.split == split
        vp = R[d].values
        cost = ds.loc[d, "cost_i"].values
        seg = camp.loc[d, "age_band"].values
        N = len(vp); K = int(mc["capacity_frac"] * N); B = mc["budget_frac_of_capacity_cost"] * K * cost.mean()
        a_tree = pt.predict(camp.loc[d, feats]).astype(float)
        # cây + cắt theo ngân sách: ưu tiên khách theo giá trị V̂ trong các lá được gửi cho tới khi hết ngân sách/năng lực
        V = ds.loc[d, "value_hat"].values
        order = np.argsort(-V)
        a_bud = np.zeros(N); used_b, used_k = 0.0, 0
        for i in order:
            if a_tree[i] == 1 and used_k + 1 <= K and used_b + cost[i] <= B:
                a_bud[i] = 1; used_b += cost[i]; used_k += 1
        sol_path = proc / f"milp_solution_{split}.csv"
        if sol_path.exists():
            sol = pd.read_csv(sol_path, dtype={"client_id": str}).set_index("client_id")
            x_milp = sol.reindex(camp.index[d].astype(str))["x"].fillna(0).values.astype(float)
        else:
            from s6_milp import greedy, solve
            xg = greedy(V, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"])
            x_milp, _ = solve(V, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"], mc["solver"],
                              time_limit=mc["time_limit_s"], hint=xg)
            x_milp = np.round(x_milp)
        pols = {"MILP (SCIP)": x_milp, "Cây chính sách (không giới hạn ngân sách)": a_tree,
                "Cây chính sách + giới hạn ngân sách": a_bud}
        bidx = rng.integers(0, N, (mc["n_bootstrap"], N))
        bm = np.array([(x_milp[ix] * vp[ix]).sum() for ix in bidx])
        for name, x in pols.items():
            bs = np.array([(x[ix] * vp[ix]).sum() for ix in bidx])
            spent = float((cost * x).sum())
            rows.append({"Tập": {"locked": "Locked Test", "oot": "OOT"}[split], "Chính sách": name,
                         "Số khách gửi": int(x.sum()), "Chi phí (RUB)": spent,
                         "Lợi nhuận tăng thêm DR (RUB)": float((x * vp).sum()),
                         "KTC95 dưới": np.quantile(bs, .025), "KTC95 trên": np.quantile(bs, .975),
                         "ROI tăng thêm": float((x * vp).sum()) / spent if spent > 0 else np.nan,
                         "Chính sách − MILP: KTC95 dưới": np.quantile(bs - bm, .025),
                         "Chính sách − MILP: KTC95 trên": np.quantile(bs - bm, .975),
                         "Trong ngân sách": bool(spent <= B + 1e-6),
                         "Khả thi (mọi ràng buộc)": feasible(x, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"])})
    ev = pd.DataFrame(rows)
    save_table(ev, cfg, "tx40_policy_tree_eval",
               "Giá trị chính sách của cây chính sách so với MILP trên Locked Test và OOT (ước lượng DR, bootstrap ghép cặp)")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, s in zip(axs, ["Locked Test", "OOT"]):
        g = ev[ev["Tập"] == s].reset_index(drop=True)
        yv = g["Lợi nhuận tăng thêm DR (RUB)"].values
        ax.bar(range(len(g)), yv, color=[PALETTE[0], PALETTE[1], PALETTE[2]])
        ax.errorbar(range(len(g)), yv, yerr=[yv - g["KTC95 dưới"].values, g["KTC95 trên"].values - yv], fmt="none", ecolor="k", capsize=4)
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(g)))
        ax.set_xticklabels(["MILP", "Cây\n(không giới hạn)", "Cây +\nngân sách"]); ax.set_title(s)
    axs[0].set_ylabel("Lợi nhuận tăng thêm DR (RUB)")
    save_fig(fig, cfg, "fx30_policy_tree", "Lợi nhuận tăng thêm của cây chính sách và MILP (thanh lỗi: KTC95 bootstrap)")
    joblib.dump(pt, cfg["paths"]["models"] / "policy_tree.joblib")
    return grid, rules, ev, {"depth": bd, "min_leaf": bl}


def _tree_rules(pt, feats):
    tree = pt.tree_
    rows = []

    def rec(node, conds):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            val = tree.value[node].ravel()
            rows.append({"Quy tắc": " và ".join(conds) if conds else "(toàn bộ)",
                         "Hành động": "Gửi SMS" if np.argmax(val) == 1 else "Không gửi",
                         "Số khách Train": int(tree.n_node_samples[node]),
                         "Lợi nhuận DR TB / khách nếu gửi (RUB)": float(val[1] - val[0]) if len(val) > 1 else np.nan})
            return
        f, th = feats[tree.feature[node]], tree.threshold[node]
        rec(left, conds + [f"{_vn(f)} ≤ {th:.2f}".replace(".", ",")])
        rec(right, conds + [f"{_vn(f)} > {th:.2f}".replace(".", ",")])

    rec(0, [])
    return pd.DataFrame(rows)


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    _load_names(cfg)
    camp = load_campaign(cfg)
    L = joblib.load(cfg["paths"]["models"] / "uplift_learners.joblib")
    feats = L.feats
    camp_raw = camp.copy()
    # EconML và cây chính sách của scikit-learn không nhận giá trị thiếu: điền trung vị của Train
    med = camp.loc[camp.split == "train", feats].median()
    camp = camp.copy(); camp[feats] = camp[feats].fillna(med)
    ct = pd.read_parquet(cfg["paths"]["processed"] / "cate_campaign.parquet")
    m = {}
    res, comp = econml_benchmark(cfg, plt, camp, feats, ct)
    m["econml"] = res.to_dict(orient="records")
    tab, sm = shap_cate(cfg, plt, camp_raw, feats, L)
    m["shap"] = {**sm, "top": tab.to_dict(orient="records")}
    grid, rules, ev, best = policy_learning(cfg, plt, camp, feats)
    m["policy_tree"] = {"best": best, "eval": ev.to_dict(orient="records"), "n_leaves": len(rules)}
    save_metrics(cfg, "xai_policy", m)
    return m


if __name__ == "__main__":
    run(load_config())
