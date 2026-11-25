"""Bằng chứng bổ sung cho giai đoạn 5-6 và cho kết luận thống kê trên Locked Test.

Ba phân tích, không dùng lại dữ liệu Locked Test/OOT để lựa chọn bất cứ điều gì:
  (A) Phân tích lực thống kê: từ khoảng tin cậy bootstrap đã có trong t28 (Locked Test, OOT), suy ra sai số
      chuẩn, mức chênh lệch nhỏ nhất phát hiện được (MDE, lực 80%) và cỡ mẫu cần để phát hiện ước lượng điểm.
  (B) Mô phỏng tổng hợp SPO+ so với PTO: dữ liệu có hàm giá trị biết trước, thay đổi mức sai đặc tả
      (bậc đa thức) và mức nhiễu của nhãn chi phí, đo regret thật trên tập kiểm tra.
  (C) Mô phỏng tổng hợp MILP so với Greedy: thay đổi số ràng buộc chặt (ngân sách, năng lực, trần, sàn).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_spo import Adam, oracle, regret, spo_plus_loss_grad  # noqa: E402
from s6_milp import feasible, greedy, solve  # noqa: E402
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl  # noqa: E402

Z_A, Z_B = norm.ppf(0.975), norm.ppf(0.80)


# ------------------------------------------------------------------ (A) lực thống kê
def power_analysis(cfg, plt):
    tabs = cfg["paths"]["tables"]
    split = pd.read_csv(tabs / "t06_split_summary.csv", encoding="utf-8-sig").set_index("Tập")["Số khách"]
    rows = []
    for key, lab in (("locked", "Locked Test"), ("oot", "OOT")):
        t = pd.read_csv(tabs / f"t28_policy_comparison_{key}.csv", encoding="utf-8-sig").set_index("Chính sách")
        n_set = int(split[key])
        milp = t.loc["MILP (SCIP)"]
        comps = [("MILP so với không gửi", milp["Lợi nhuận tăng thêm DR (RUB)"], milp["KTC95 dưới"], milp["KTC95 trên"])]
        for pol in ["Ngẫu nhiên K khách", "Top-K xác suất mua (p_buy)"]:
            r = t.loc[pol]
            eff = milp["Lợi nhuận tăng thêm DR (RUB)"] - r["Lợi nhuận tăng thêm DR (RUB)"]
            comps.append((f"MILP so với {pol.split(' (')[0].lower()}", eff,
                          r["MILP − chính sách: KTC95 dưới"], r["MILP − chính sách: KTC95 trên"]))
        for name, eff, lo, hi in comps:
            se = (hi - lo) / (2 * Z_A)
            mde = (Z_A + Z_B) * se
            power = norm.cdf(abs(eff) / se - Z_A) + norm.cdf(-abs(eff) / se - Z_A)
            n_req = n_set * ((Z_A + Z_B) * se / abs(eff)) ** 2 if eff != 0 else np.nan
            rows.append({"Tập": lab, "So sánh": name, "Cỡ tập": n_set, "Ước lượng điểm (RUB)": eff,
                         "KTC95 dưới": lo, "KTC95 trên": hi, "Sai số chuẩn (RUB)": se,
                         "MDE lực 80% (RUB)": mde, "Lực tại ước lượng điểm": power,
                         "Cỡ mẫu cần cho lực 80%": n_req, "Bội số so với cỡ tập": n_req / n_set})
    pa = pd.DataFrame(rows)
    save_table(pa, cfg, "tx32_power_analysis",
               "Phân tích lực thống kê của các so sánh giá trị chính sách (suy ra từ khoảng tin cậy bootstrap)")

    # đường lực theo cỡ mẫu cho MILP so với không gửi
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, (lab, c) in enumerate((("Locked Test", PALETTE[0]), ("OOT", PALETTE[1]))):
        r = pa[(pa["Tập"] == lab) & (pa["So sánh"] == "MILP so với không gửi")].iloc[0]
        mult = np.linspace(0.25, 6, 200)
        se_m = r["Sai số chuẩn (RUB)"] / np.sqrt(mult)
        pw = norm.cdf(abs(r["Ước lượng điểm (RUB)"]) / se_m - Z_A)
        ax.plot(mult * r["Cỡ tập"] / 1000, pw, color=c, lw=2, label=f"{lab} (ước lượng điểm {r['Ước lượng điểm (RUB)']:,.0f} RUB)".replace(",", "."))
        ax.axvline(r["Cỡ tập"] / 1000, color=c, ls=":", lw=1)
    ax.axhline(0.8, color="black", ls="--", lw=1)
    ax.set_xlabel("Cỡ tập kiểm định chính sách (nghìn khách hàng)"); ax.set_ylabel("Lực thống kê")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right")
    ax.set_title("Lực phát hiện lợi nhuận tăng thêm của MILP theo cỡ tập kiểm định")
    save_fig(fig, cfg, "fx24_power_curve",
             "Lực thống kê phát hiện lợi nhuận tăng thêm của MILP theo cỡ tập kiểm định (đường chấm: cỡ tập hiện tại)")
    return pa


# ------------------------------------------------------------------ (B) SPO+ so với PTO
NOISES = [("Không nhiễu", "none", 0.0), ("Nhiễu nhân ±50%", "mult", 0.5), ("Nhiễu cộng lớn (3 × ĐLC)", "add", 3.0)]


def _costs(X, Bm, deg):
    """Thiết kế của Elmachtoub và Grigas (2022): mỗi bài toán có vector đặc trưng x, chi phí của d phương án
    c_j = -[((B x)_j / sqrt(p) + 3)^deg + 1] (dấu âm vì bài toán chọn k phương án có giá trị cao nhất)."""
    p = X.shape[1]
    return -(((X @ Bm.T) / np.sqrt(p) + 3.0) ** deg + 1.0) / (3.5 ** deg)


def spo_simulation(cfg, plt, rng):
    """Tái lập thí nghiệm tổng hợp của Elmachtoub và Grigas (2022) cho bài toán chọn k trong d phương án:
    mô hình giả thuyết tuyến tính, bậc đa thức điều khiển mức sai đặc tả; thêm một kịch bản nhiễu cộng lớn
    để mô phỏng nhãn chi phí thay thế (DR pseudo-outcome) như trên dữ liệu X5."""
    p, d, k = 5, 20, 5
    n_train, n_test = 1000, 2000
    degrees, reps = [1, 2, 4, 6, 8], 5
    rows = []
    for rep in range(reps):
        Bm = rng.binomial(1, 0.5, size=(d, p)).astype(float)
        for deg in degrees:
            Xtr = rng.normal(size=(n_train, p)); Xte = rng.normal(size=(n_test, p))
            Xtr1, Xte1 = np.c_[Xtr, np.ones(n_train)], np.c_[Xte, np.ones(n_test)]
            Ctr_true, Cte = _costs(Xtr, Bm, deg), _costs(Xte, Bm, deg)
            sd = Ctr_true.std()
            for lab, kind, lv in NOISES:
                if kind == "none":
                    Ctr = Ctr_true.copy()
                elif kind == "mult":
                    Ctr = Ctr_true * rng.uniform(1 - lv, 1 + lv, size=Ctr_true.shape)
                else:
                    Ctr = Ctr_true + rng.normal(scale=lv * sd, size=Ctr_true.shape)
                ridge = Ridge(alpha=1.0, fit_intercept=False).fit(Xtr1, Ctr)
                Th = ridge.coef_.copy()            # (d, p+1)
                opt = Adam(Th.shape, 0.01)
                for ep in range(60):
                    perm = rng.permutation(n_train)
                    for i in range(0, n_train, 32):
                        idx = perm[i:i + 32]
                        Xb, Cb = Xtr1[idx], Ctr[idx]
                        _, g = spo_plus_loss_grad(Xb @ Th.T, Cb, k)   # g: (b, d)
                        grad = g.T @ Xb / len(idx) + 1e-4 * Th
                        Th -= opt.step(grad)
                for name, coef in (("PTO - Ridge (MSE)", ridge.coef_), ("SPO+ (tuyến tính)", Th)):
                    r, z, _ = regret(Xte1 @ coef.T, Cte, k)
                    rows.append({"Lần lặp": rep, "Bậc đa thức": deg, "Loại nhiễu nhãn": lab,
                                 "Phương pháp": name, "Regret chuẩn hóa": r.sum() / np.abs(z).sum()})
    raw = pd.DataFrame(rows)
    out = []
    for (lab, dg), g in raw.groupby(["Loại nhiễu nhãn", "Bậc đa thức"], sort=False):
        a = g[g["Phương pháp"] == "PTO - Ridge (MSE)"].sort_values("Lần lặp")["Regret chuẩn hóa"].values
        b = g[g["Phương pháp"] == "SPO+ (tuyến tính)"].sort_values("Lần lặp")["Regret chuẩn hóa"].values
        out.append({"Loại nhiễu nhãn": lab, "Bậc đa thức": dg, "Regret PTO": a.mean(), "ĐLC PTO": a.std(ddof=1),
                    "Regret SPO+": b.mean(), "ĐLC SPO+": b.std(ddof=1),
                    "Mức giảm regret của SPO+ (%)": 100 * (a.mean() - b.mean()) / a.mean() if a.mean() > 1e-4 else np.nan,
                    "Số lần SPO+ tốt hơn (/5)": int((b < a).sum())})
    wide = pd.DataFrame(out)
    save_table(wide, cfg, "tx33_spo_simulation",
               "Mô phỏng tổng hợp theo thiết kế của Elmachtoub và Grigas (2022): regret chuẩn hóa của PTO và SPO+ (5 lần lặp)")

    fig, axs = plt.subplots(1, len(NOISES), figsize=(13, 3.9), sharey=True)
    for ax, (lab, _, _) in zip(axs, NOISES):
        dd = wide[wide["Loại nhiễu nhãn"] == lab]
        ax.errorbar(dd["Bậc đa thức"], dd["Regret PTO"], yerr=dd["ĐLC PTO"], marker="o", color=PALETTE[0], label="PTO - Ridge (MSE)", capsize=3)
        ax.errorbar(dd["Bậc đa thức"], dd["Regret SPO+"], yerr=dd["ĐLC SPO+"], marker="s", color=PALETTE[1], label="SPO+ (tuyến tính)", capsize=3)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Bậc đa thức (mức sai đặc tả)"); ax.set_xticks(degrees)
    axs[0].set_ylabel("Regret chuẩn hóa (thấp hơn = tốt hơn)"); axs[0].legend()
    save_fig(fig, cfg, "fx25_spo_simulation",
             "Mô phỏng tổng hợp: regret chuẩn hóa của PTO và SPO+ theo mức sai đặc tả và loại nhiễu nhãn")
    return wide


# ------------------------------------------------------------------ (C) MILP so với Greedy
def milp_simulation(cfg, plt, rng):
    N, reps = 3000, 5
    segs = np.array(["A", "B", "C", "D", "E"])
    scen = [
        ("S1. Chỉ ngân sách chặt", dict(cap=0.60, bud=0.15, a=1.00, b=0.00)),
        ("S2. Ngân sách và năng lực cùng chặt", dict(cap=0.20, bud=0.22, a=1.00, b=0.00)),
        ("S3. Thêm trần tập trung theo nhóm", dict(cap=0.20, bud=0.22, a=0.30, b=0.00)),
        ("S4. Thêm sàn độ phủ theo nhóm", dict(cap=0.20, bud=0.22, a=0.30, b=0.10)),
    ]
    rows = []
    for rep in range(reps):
        seg = rng.choice(segs, size=N, p=[0.35, 0.25, 0.2, 0.12, 0.08])
        base = pd.Series(seg).map({"A": 1.2, "B": 0.9, "C": 0.6, "D": 0.2, "E": -0.2}).values
        V = rng.normal(base, 1.0) * 10
        cost = rng.choice([1.0, 2.0, 4.0], size=N, p=[0.5, 0.3, 0.2]) + rng.uniform(0, 1, N)
        V = V + 2.0 * (cost - cost.mean())  # khách đắt hơn có giá trị cao hơn, tỷ số V/c khác thứ tự V
        for name, s in scen:
            K = int(s["cap"] * N); B = s["bud"] * cost.sum()
            t0 = time.perf_counter(); xm, info = solve(V, cost, seg, K, B, s["a"], s["b"], time_limit=60); tm = time.perf_counter() - t0
            xg = greedy(V, cost, seg, K, B, s["a"], s["b"], repair=True)
            xg0 = greedy(V, cost, seg, K, B, s["a"], s["b"], repair=False)
            _, lp = solve(V, cost, seg, K, B, s["a"], s["b"], relax=True)
            binding = sum(abs(dv) > 1e-6 for dv in lp["duals"].values())
            vm, vg = float(V @ xm), float(V @ xg)
            rows.append({"Lần lặp": rep, "Kịch bản": name, "Số ràng buộc chặt (LP)": binding,
                         "Giá trị MILP": vm, "Giá trị Greedy + sửa chữa": vg,
                         "Khoảng cách Greedy (%)": 100 * (vm - vg) / abs(vm),
                         "Greedy không sửa khả thi": feasible(xg0, cost, seg, K, B, s["a"], s["b"]),
                         "Greedy + sửa chữa khả thi": feasible(xg, cost, seg, K, B, s["a"], s["b"]),
                         "Thời gian MILP (giây)": tm})
    raw = pd.DataFrame(rows)
    agg = raw.groupby("Kịch bản", sort=False).agg(**{
        "Số ràng buộc chặt (LP)": ("Số ràng buộc chặt (LP)", "median"),
        "Giá trị MILP": ("Giá trị MILP", "mean"),
        "Giá trị Greedy + sửa chữa": ("Giá trị Greedy + sửa chữa", "mean"),
        "Khoảng cách Greedy TB (%)": ("Khoảng cách Greedy (%)", "mean"),
        "Khoảng cách Greedy lớn nhất (%)": ("Khoảng cách Greedy (%)", "max"),
        "Greedy không sửa khả thi (/5)": ("Greedy không sửa khả thi", "sum"),
        "Greedy + sửa chữa khả thi (/5)": ("Greedy + sửa chữa khả thi", "sum"),
        "Thời gian MILP TB (giây)": ("Thời gian MILP (giây)", "mean")}).reset_index()
    save_table(agg, cfg, "tx34_milp_simulation",
               "Mô phỏng tổng hợp: khoảng cách giữa Greedy và MILP theo số ràng buộc chặt (3.000 khách, 5 lần lặp)")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(range(len(agg)), agg["Khoảng cách Greedy TB (%)"], color=PALETTE[0])
    ax.errorbar(range(len(agg)), agg["Khoảng cách Greedy TB (%)"],
                yerr=[agg["Khoảng cách Greedy TB (%)"] * 0, agg["Khoảng cách Greedy lớn nhất (%)"] - agg["Khoảng cách Greedy TB (%)"]],
                fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(range(len(agg))); ax.set_xticklabels([s.split(". ")[0] for s in agg["Kịch bản"]])
    ax.set_ylabel("Giá trị MILP vượt Greedy (%)")
    ax.set_title("Lợi thế của MILP so với Greedy tăng theo số ràng buộc chặt")
    save_fig(fig, cfg, "fx26_milp_simulation",
             "Mô phỏng tổng hợp: phần giá trị MILP vượt Greedy + sửa chữa theo kịch bản ràng buộc (thanh lỗi: lần lặp xấu nhất của Greedy)")
    return agg


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    rng = np.random.default_rng(cfg["seed"])
    log.info("Phân tích lực thống kê ...")
    pa = power_analysis(cfg, plt)
    log.info("Mô phỏng SPO+ so với PTO ...")
    sp = spo_simulation(cfg, plt, rng)
    log.info("Mô phỏng MILP so với Greedy ...")
    ml = milp_simulation(cfg, plt, rng)
    m = {"power": pa.to_dict(orient="records"), "spo_sim": sp.to_dict(orient="records"), "milp_sim": ml.to_dict(orient="records")}
    save_metrics(cfg, "evidence", m)
    return m


if __name__ == "__main__":
    run(load_config())
