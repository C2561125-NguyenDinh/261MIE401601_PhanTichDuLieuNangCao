"""Giai đoạn 6 - Tối ưu hóa phân bổ nguồn lực dưới các ràng buộc bằng MILP (bài toán cái túi đa chiều).

    max  sum_i V_i x_i
    s.t. sum_i x_i                 <= K           (năng lực gửi SMS)
         sum_i cost_i x_i          <= B           (ngân sách tiền)
         sum_{i in s} x_i          <= a * K       (không tập trung quá mức vào một nhóm tuổi s)
         sum_{i in s} x_i          >= b * n_s     (độ phủ tối thiểu cho mỗi nhóm tuổi - công bằng)
         x_i in {0, 1}
So sánh: tham lam (greedy), nới lỏng LP, các chính sách tham chiếu; đánh giá giá trị bằng ước lượng DR.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from s2_features import load_campaign
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl


def solve(V, cost, seg, K, B, a, b, solver_name="SCIP", relax=False, time_limit=120, hint=None):
    solver = pywraplp.Solver.CreateSolver("GLOP" if relax else solver_name)
    n = len(V)
    x = [solver.NumVar(0, 1, f"x{i}") if relax else solver.BoolVar(f"x{i}") for i in range(n)]
    cons = {}
    cons["capacity"] = solver.Add(solver.Sum(x) <= K)
    cons["budget"] = solver.Add(solver.Sum([cost[i] * x[i] for i in range(n)]) <= B)
    for s in np.unique(seg):
        ids = np.where(seg == s)[0]
        cons[f"max_share[{s}]"] = solver.Add(solver.Sum([x[i] for i in ids]) <= a * K)
        cons[f"min_cov[{s}]"] = solver.Add(solver.Sum([x[i] for i in ids]) >= np.floor(b * len(ids)))
    solver.Maximize(solver.Sum([V[i] * x[i] for i in range(n)]))
    if not relax:
        solver.SetTimeLimit(int(time_limit * 1000))
        if hint is not None:
            solver.SetHint(x, list(map(float, hint)))
    t0 = time.perf_counter()
    params = pywraplp.MPSolverParameters()
    if not relax:
        params.SetDoubleParam(params.RELATIVE_MIP_GAP, 1e-7)  # dung sai tối ưu 0,00001%
    status = solver.Solve(params)
    dt = time.perf_counter() - t0
    sol = np.array([v.solution_value() for v in x])
    info = {"status": {0: "OPTIMAL", 1: "FEASIBLE", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "ABNORMAL", 6: "NOT_SOLVED"}.get(status, str(status)),
            "objective": solver.Objective().Value(), "time_s": dt}
    if not relax:
        info["best_bound"] = solver.Objective().BestBound()
    else:
        info["duals"] = {k: c.dual_value() for k, c in cons.items()}
    return sol, info


def greedy(V, cost, seg, K, B, a, b, repair=True):
    n = len(V)
    x = np.zeros(n)
    order = np.argsort(-(V / cost))
    used_k, used_b = 0, 0.0
    segs = np.unique(seg)
    seg_cnt = {s: 0 for s in segs}
    seg_n = {s: (seg == s).sum() for s in segs}
    for i in order:
        if V[i] <= 0:
            break
        if used_k + 1 > K or used_b + cost[i] > B or seg_cnt[seg[i]] + 1 > a * K:
            continue
        x[i] = 1; used_k += 1; used_b += cost[i]; seg_cnt[seg[i]] += 1
    if repair:  # sửa chữa ràng buộc độ phủ tối thiểu
        for s in segs:
            need = int(np.floor(b * seg_n[s])) - seg_cnt[s]
            if need <= 0:
                continue
            cand = [i for i in order if seg[i] == s and x[i] == 0]
            for i in cand[:need]:
                # nếu vượt năng lực/ngân sách -> bỏ khách có tỷ số thấp nhất thuộc nhóm còn dư
                while used_k + 1 > K or used_b + cost[i] > B:
                    sel = np.where((x == 1) & (seg != s))[0]
                    sel = [j for j in sel if seg_cnt[seg[j]] - 1 >= np.floor(b * seg_n[seg[j]])]
                    if not sel:
                        break
                    j = min(sel, key=lambda j: V[j] / cost[j])
                    x[j] = 0; used_k -= 1; used_b -= cost[j]; seg_cnt[seg[j]] -= 1
                x[i] = 1; used_k += 1; used_b += cost[i]; seg_cnt[s] += 1
    return x


def feasible(x, cost, seg, K, B, a, b):
    ok = x.sum() <= K + 1e-6 and (cost * x).sum() <= B + 1e-6
    for s in np.unique(seg):
        m = seg == s
        ok &= x[m].sum() <= a * K + 1e-6 and x[m].sum() >= np.floor(b * m.sum()) - 1e-6
    return bool(ok)


def dr_value(x, vproxy):
    return float((x * vproxy).sum())


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    rng = np.random.default_rng(cfg["seed"])
    proc = cfg["paths"]["processed"]
    mc = cfg["milp"]; bz = cfg["business"]
    camp = load_campaign(cfg)
    ds = pd.read_parquet(proc / "decision_scores.parquet")
    ct = pd.read_parquet(proc / "cate_campaign.parquet")
    fc = pd.read_parquet(proc / "forecast_campaign.parquet")
    m: dict = {}
    results = {}

    def instance(split, cap_frac=None):
        d = camp.split == split
        V = ds.loc[d, "value_hat"].values.astype(float)
        cost = ds.loc[d, "cost_i"].values.astype(float)
        seg = camp.loc[d, "age_band"].values
        N = len(V)
        K = int((cap_frac or mc["capacity_frac"]) * N)
        B = mc["budget_frac_of_capacity_cost"] * K * cost.mean()
        return d, V, cost, seg, N, K, B

    for split in ["locked", "oot"]:
        d, V, cost, seg, N, K, B = instance(split)
        vp = ds.loc[d, "value_proxy"].values
        psi = ds.loc[d, "psi"].values
        a, b = mc["segment_max_share"], mc["segment_min_coverage"]
        log.info(f"MILP {split}: N={N}, K={K}, B={B:.0f}")
        x_lp, info_lp = solve(V, cost, seg, K, B, a, b, relax=True)
        t0 = time.perf_counter(); x_g = greedy(V, cost, seg, K, B, a, b, repair=False); tg = time.perf_counter() - t0
        t0 = time.perf_counter(); x_gr = greedy(V, cost, seg, K, B, a, b, repair=True); tgr = time.perf_counter() - t0
        x_ip, info_ip = solve(V, cost, seg, K, B, a, b, mc["solver"], time_limit=mc["time_limit_s"], hint=x_gr)
        x_ip = np.round(x_ip)
        # cùng bài toán nhưng bỏ ràng buộc trần/sàn nhóm tuổi -> đo "cái giá của công bằng"
        x_nf, info_nf = solve(V, cost, seg, K, B, 1.0, 0.0, mc["solver"], time_limit=mc["time_limit_s"])
        x_nf = np.round(x_nf)
        policies = {
            "Không gửi (baseline)": np.zeros(N),
            "Ngẫu nhiên K khách": np.isin(np.arange(N), rng.choice(N, K, replace=False)).astype(float),
            "Top-K xác suất mua (p_buy)": np.isin(np.arange(N), np.argsort(-fc.loc[d, "p_buy_cal"].values)[:K]).astype(float),
            "Top-K uplift (bỏ qua chi phí)": np.isin(np.arange(N), np.argsort(-ct.loc[d, "tau_hat"].values)[:K]).astype(float),
            "Greedy (không sửa ràng buộc)": x_g,
            "Greedy + sửa chữa": x_gr,
            "MILP không ràng buộc nhóm tuổi": x_nf,
            "MILP (SCIP)": x_ip,
        }
        rows = []
        boot_idx = rng.integers(0, N, (mc["n_bootstrap"], N))  # bootstrap ghép cặp: cùng mẫu cho mọi chính sách
        boot_milp = np.array([dr_value(x_ip[ix], vp[ix]) for ix in boot_idx])
        for name, x in policies.items():
            boots = np.array([dr_value(x[ix], vp[ix]) for ix in boot_idx])
            diff = boot_milp - boots
            spent = float((cost * x).sum())
            val = dr_value(x, vp)
            rows.append({"Chính sách": name, "Số khách gửi": int(x.sum()), "Chi phí (RUB)": spent,
                         "Giá trị mục tiêu V̂ (RUB)": float((V * x).sum()),
                         "Lợi nhuận tăng thêm DR (RUB)": val, "KTC95 dưới": np.quantile(boots, .025),
                         "KTC95 trên": np.quantile(boots, .975), "Lượt mua tăng thêm (DR)": float((psi * x).sum()),
                         "ROI tăng thêm": val / spent if spent > 0 else np.nan,
                         "MILP − chính sách: KTC95 dưới": np.quantile(diff, .025),
                         "MILP − chính sách: KTC95 trên": np.quantile(diff, .975),
                         "Khả thi": feasible(x, cost, seg, K, B, a, b)})
        pol = pd.DataFrame(rows)
        save_table(pol, cfg, f"t28_policy_comparison_{split}",
                   f"So sánh chính sách phân bổ trên tập {'Locked Test (policy validation)' if split == 'locked' else 'Out-of-Time'}")
        results[split] = {"N": N, "K": K, "B": B, "lp": {k: v for k, v in info_lp.items() if k != "duals"},
                          "ip": info_ip, "greedy_time": tg, "greedy_repair_time": tgr,
                          "lp_fractional": int(((x_lp > 1e-6) & (x_lp < 1 - 1e-6)).sum()),
                          "gap_ip_lp": (info_lp["objective"] - info_ip["objective"]) / abs(info_lp["objective"]),
                          "gap_greedy_ip": (info_ip["objective"] - (V * x_gr).sum()) / abs(info_ip["objective"]),
                          "price_of_fairness_obj": info_nf["objective"] - info_ip["objective"],
                          "policies": pol.to_dict(orient="records")}
        if split == "locked":
            duals = pd.DataFrame([{"Ràng buộc": k, "Giá đối ngẫu (shadow price)": v} for k, v in info_lp["duals"].items()])
            save_table(duals, cfg, "t29_shadow_prices", "Giá đối ngẫu của các ràng buộc từ nới lỏng LP (Locked Test)")
            results[split]["duals"] = info_lp["duals"]
            seg_t = pd.DataFrame({"Nhóm tuổi": seg, "MILP": x_ip, "Greedy": x_g, "Greedy+sửa": x_gr, "V": V})
            st = seg_t.groupby("Nhóm tuổi").agg(n=("V", "size"), milp=("MILP", "sum"), greedy=("Greedy", "sum"),
                                                gr=("Greedy+sửa", "sum"), v_pos=("V", lambda s: (s > 0).mean()))
            st["Sàn tối thiểu"] = np.floor(b * st["n"]); st["Trần"] = a * K
            st = st.reset_index().rename(columns={"n": "Số khách", "milp": "MILP chọn", "greedy": "Greedy chọn",
                                                  "gr": "Greedy+sửa chọn", "v_pos": "Tỷ lệ V̂>0"})
            save_table(st, cfg, "t30_segment_allocation", "Phân bổ lượt gửi theo nhóm tuổi và các ràng buộc công bằng")
            x_locked = x_ip
            fig, ax = plt.subplots(figsize=(9, 4))
            xs = np.arange(len(st))
            ax.bar(xs - .27, st["MILP chọn"], .27, color=PALETTE[0], label="MILP")
            ax.bar(xs, st["Greedy chọn"], .27, color=PALETTE[1], label="Greedy")
            ax.bar(xs + .27, st["Greedy+sửa chọn"], .27, color=PALETTE[2], label="Greedy + sửa")
            ax.plot(xs, st["Sàn tối thiểu"], "k_", ms=25, mew=2, label="Sàn tối thiểu")
            ax.axhline(a * K, color="r", ls="--", label="Trần 40% K")
            ax.set_xticks(xs); ax.set_xticklabels(st["Nhóm tuổi"]); ax.legend(fontsize=8)
            ax.set_title("Phân bổ lượt gửi theo nhóm tuổi")
            save_fig(fig, cfg, "f33_segment_allocation", "Phân bổ lượt gửi theo nhóm tuổi so với sàn và trần ràng buộc")

        fig, ax = plt.subplots(figsize=(9, 4.2))
        p2 = pol.iloc[1:]
        ax.barh(p2["Chính sách"], p2["Lợi nhuận tăng thêm DR (RUB)"],
                xerr=[p2["Lợi nhuận tăng thêm DR (RUB)"] - p2["KTC95 dưới"], p2["KTC95 trên"] - p2["Lợi nhuận tăng thêm DR (RUB)"]],
                color=[PALETTE[5]] * 3 + [PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[0]], capsize=4)
        ax.axvline(0, color="k", lw=1); ax.set_xlabel("Lợi nhuận tăng thêm ước lượng DR (RUB, KTC 95% bootstrap)")
        ax.set_title(f"Giá trị chính sách - {'Locked Test' if split == 'locked' else 'Out-of-Time'}")
        save_fig(fig, cfg, f"f34_policy_value_{split}", f"Lợi nhuận tăng thêm của các chính sách trên tập {'Locked Test' if split == 'locked' else 'Out-of-Time'}")

    # ---------- độ nhạy theo năng lực / ngân sách
    rows = []
    for cf in mc["budget_grid"]:
        d, V, cost, seg, N, K, B = instance("locked", cf)
        vp = ds.loc[d, "value_proxy"].values
        a, b = mc["segment_max_share"], mc["segment_min_coverage"]
        b_eff = min(b, cf * 0.5)  # giữ bài toán khả thi khi năng lực nhỏ
        _, ilp = solve(V, cost, seg, K, B, a, b_eff, relax=True)
        xg = greedy(V, cost, seg, K, B, a, b_eff, repair=True)
        xi, iip = solve(V, cost, seg, K, B, a, b_eff, mc["solver"], time_limit=mc["time_limit_s"], hint=xg)
        xi = np.round(xi)
        rows.append({"Tỷ lệ năng lực K/N": cf, "K": K, "Ngân sách B (RUB)": B, "Cận LP": ilp["objective"],
                     "MILP mục tiêu": iip["objective"], "Greedy+sửa mục tiêu": float((V * xg).sum()),
                     "MILP lợi nhuận DR": dr_value(xi, vp), "Greedy lợi nhuận DR": dr_value(xg, vp),
                     "MILP số gửi": int(xi.sum()), "MILP thời gian (s)": iip["time_s"],
                     "Chi phí MILP (RUB)": float((cost * xi).sum())})
    sens = pd.DataFrame(rows)
    sens["ROI tăng thêm MILP"] = sens["MILP lợi nhuận DR"] / sens["Chi phí MILP (RUB)"].replace(0, np.nan)
    sens["Lợi nhuận biên / lượt gửi thêm"] = sens["MILP lợi nhuận DR"].diff() / sens["MILP số gửi"].diff()
    save_table(sens, cfg, "t31_budget_sensitivity", "Độ nhạy theo năng lực gửi và ngân sách (Locked Test)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(sens["Tỷ lệ năng lực K/N"], sens["Cận LP"], "k--", label="Cận trên LP")
    ax[0].plot(sens["Tỷ lệ năng lực K/N"], sens["MILP mục tiêu"], "o-", color=PALETTE[0], label="MILP")
    ax[0].plot(sens["Tỷ lệ năng lực K/N"], sens["Greedy+sửa mục tiêu"], "s-", color=PALETTE[1], label="Greedy + sửa")
    ax[0].set_title("Giá trị mục tiêu V̂"); ax[0].set_xlabel("K / N"); ax[0].legend()
    ax[1].plot(sens["Tỷ lệ năng lực K/N"], sens["MILP lợi nhuận DR"], "o-", color=PALETTE[0], label="MILP")
    ax[1].plot(sens["Tỷ lệ năng lực K/N"], sens["Greedy lợi nhuận DR"], "s-", color=PALETTE[1], label="Greedy + sửa")
    ax[1].axhline(0, color="k", lw=1); ax[1].set_title("Lợi nhuận tăng thêm (DR)"); ax[1].set_xlabel("K / N"); ax[1].legend()
    save_fig(fig, cfg, "f35_budget_sensitivity", "Độ nhạy của giá trị mục tiêu và lợi nhuận tăng thêm theo năng lực gửi")

    # ---------- độ nhạy theo giả định kinh tế
    d, _, _, seg, N, K, _ = instance("locked")
    S = ds.loc[d, "spend_cap"].values; p1 = ds.loc[d, "p1_hat"].values; tau = ct.loc[d, "tau_hat"].values
    psi = ds.loc[d, "psi"].values
    rows = []
    for mr in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for ci in [0.0, 1.0, 2.0, 4.0, 8.0]:
            cost = bz["contact_cost"] + p1 * ci
            V = tau * mr * S - cost
            vp = psi * mr * S - cost
            B = mc["budget_frac_of_capacity_cost"] * K * cost.mean()
            xg = greedy(V, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"], repair=True)
            xi, iip = solve(V, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"], mc["solver"],
                            time_limit=60, hint=xg)
            xi = np.round(xi)
            rows.append({"Biên lợi nhuận m": mr, "Chi phí ưu đãi (RUB)": ci, "Số khách V̂>0": int((V > 0).sum()),
                         "MILP số gửi": int(xi.sum()), "MILP lợi nhuận DR": dr_value(xi, vp),
                         "Gửi tất cả K theo uplift - lợi nhuận DR": dr_value(np.isin(np.arange(N), np.argsort(-tau)[:K]).astype(float), vp)})
    es = pd.DataFrame(rows)
    save_table(es, cfg, "t32_economic_sensitivity", "Độ nhạy của quyết định tối ưu theo giả định biên lợi nhuận và chi phí ưu đãi")
    piv = es.pivot(index="Biên lợi nhuận m", columns="Chi phí ưu đãi (RUB)", values="MILP lợi nhuận DR")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:,.0f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Chi phí ưu đãi (RUB)"); ax.set_ylabel("Biên lợi nhuận m"); fig.colorbar(im); ax.grid(False)
    ax.set_title("Lợi nhuận tăng thêm DR của MILP (RUB)")
    save_fig(fig, cfg, "f36_economic_sensitivity", "Bản đồ nhiệt độ nhạy lợi nhuận theo giả định kinh tế")

    # ---------- độ ổn định nghiệm dưới bất định của uplift
    d, V, cost, seg, N, K, B = instance("locked")
    vp = ds.loc[d, "value_proxy"].values
    tau_std = ct.loc[d, "tau_std"].values
    S = ds.loc[d, "spend_cap"].values
    rows = []
    for r in range(min(30, mc["n_bootstrap"])):
        Vp = V + rng.normal(size=N) * tau_std * bz["margin_rate"] * S
        xg = greedy(Vp, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"])
        xr, _ = solve(Vp, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"], mc["solver"], time_limit=60, hint=xg)
        xr = np.round(xr)
        inter = (xr * x_locked).sum(); union = ((xr + x_locked) > 0).sum()
        rows.append({"Lần": r + 1, "Jaccard với nghiệm gốc": inter / union if union else 1.0,
                     "Lợi nhuận DR": dr_value(xr, vp), "Số gửi": int(xr.sum())})
    stab = pd.DataFrame(rows)
    save_table(stab, cfg, "t33_solution_stability", "Độ ổn định nghiệm MILP khi nhiễu hóa uplift theo độ bất đồng giữa các learner")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(stab["Jaccard với nghiệm gốc"], bins=15, color=PALETTE[3]); ax[0].set_title("Jaccard so với nghiệm gốc")
    ax[1].hist(stab["Lợi nhuận DR"], bins=15, color=PALETTE[0]); ax[1].axvline(dr_value(x_locked, vp), color="k", ls="--", label="Nghiệm gốc")
    ax[1].set_title("Lợi nhuận DR dưới nhiễu uplift"); ax[1].legend()
    save_fig(fig, cfg, "f37_solution_stability", "Độ ổn định của nghiệm MILP dưới bất định của uplift")
    m["stability"] = {"jaccard_mean": stab["Jaccard với nghiệm gốc"].mean(), "jaccard_min": stab["Jaccard với nghiệm gốc"].min(),
                      "dr_mean": stab["Lợi nhuận DR"].mean(), "dr_std": stab["Lợi nhuận DR"].std()}

    # ---------- checkpoint nghiệm
    sol = pd.DataFrame({"client_id": camp.index[camp.split == "locked"], "x": x_locked.astype(int)})
    sol.to_csv(proc / "milp_solution_locked.csv", index=False)
    m["solution_sha256"] = hashlib.sha256(sol.to_csv(index=False).encode()).hexdigest()
    m["results"] = results
    m["budget_sensitivity"] = sens.to_dict(orient="records")
    m["economic_sensitivity"] = es.to_dict(orient="records")
    save_metrics(cfg, "milp", m)
    return m


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
