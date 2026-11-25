"""Giai đoạn 7 - Định tuyến hành động (Action Router) tích hợp quy tắc nghiệp vụ và Human-in-the-loop.

Thứ tự ưu tiên quy tắc (rule nào đứng trước thắng khi xung đột):
  R1 SUPPRESS      : uplift dự báo < ngưỡng "sleeping dog" -> không bao giờ liên hệ
  R2 DEFER         : khách vừa mua trong <= d ngày (tránh làm phiền) và được MILP chọn
  R3 HUMAN_REVIEW  : khách VIP (chi tiêu top 1%) được MILP chọn
  R4 HUMAN_REVIEW  : độ bất định uplift cao (top 5% bất đồng giữa learner) và được MILP chọn
  R5 HUMAN_REVIEW  : khách không có lịch sử giao dịch (cold-start) và được MILP chọn
  R6 AUTO_SEND     : được MILP chọn, không vướng rào chắn nào
  R7 NO_ACTION     : còn lại
Hàng chờ duyệt vượt năng lực -> phần dư chuyển DEFER (mặc định an toàn).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from s2_features import load_campaign
from s6_milp import greedy, solve
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl

ACTIONS = ["AUTO_SEND", "HUMAN_REVIEW", "DEFER", "SUPPRESS", "NO_ACTION"]


class ActionRouter:
    def __init__(self, cfg: dict, thresholds: dict):
        self.cfg, self.rc, self.th = cfg, cfg["router"], thresholds

    def route_one(self, r: dict) -> tuple[str, list[str]]:
        reasons = []
        if r["tau_hat"] < self.rc["sleeping_dog_tau"]:
            reasons.append("R1_SLEEPING_DOG")
        if r["x"] == 1 and r["recency"] <= self.rc["recent_purchase_days"]:
            reasons.append("R2_RECENT_PURCHASE")
        if r["x"] == 1 and r["sum_all"] >= self.th["vip"]:
            reasons.append("R3_VIP")
        if r["x"] == 1 and r["tau_std"] >= self.th["unc"]:
            reasons.append("R4_HIGH_UNCERTAINTY")
        if r["x"] == 1 and r["has_tx"] == 0:
            reasons.append("R5_COLD_START")
        if "R1_SLEEPING_DOG" in reasons:
            return "SUPPRESS", reasons
        if "R2_RECENT_PURCHASE" in reasons:
            return "DEFER", reasons
        if any(x in reasons for x in ["R3_VIP", "R4_HIGH_UNCERTAINTY", "R5_COLD_START"]):
            return "HUMAN_REVIEW", reasons
        if r["x"] == 1:
            return "AUTO_SEND", reasons + ["R6_MILP_SELECTED"]
        return "NO_ACTION", reasons + ["R7_NOT_SELECTED"]

    def route(self, df: pd.DataFrame) -> pd.DataFrame:
        recs = df[["tau_hat", "x", "recency", "sum_all", "tau_std", "has_tx"]].to_dict(orient="records")
        out = [self.route_one(r) for r in recs]
        res = df.copy()
        res["action"] = [a for a, _ in out]
        res["reasons"] = ["|".join(rs) for _, rs in out]
        res["n_rules"] = [len([x for x in rs if x[:2] in ("R1", "R2", "R3", "R4", "R5")]) for _, rs in out]
        # năng lực duyệt
        rev = res.index[res["action"] == "HUMAN_REVIEW"]
        cap = self.rc["review_capacity"]
        if len(rev) > cap:
            over = res.loc[rev].sort_values("value_hat", ascending=False).index[cap:]
            res.loc[over, "action"] = "DEFER"
            res.loc[over, "reasons"] = res.loc[over, "reasons"] + "|Q_REVIEW_OVERFLOW"
        return res


def idem_key(campaign, client, version, date):
    return hashlib.sha256(f"{campaign}|{client}|{version}|{date}".encode()).hexdigest()[:20]


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    proc = cfg["paths"]["processed"]
    mc, rc = cfg["milp"], cfg["router"]
    camp = load_campaign(cfg)
    ds = pd.read_parquet(proc / "decision_scores.parquet")
    ct = pd.read_parquet(proc / "cate_campaign.parquet")
    from utils import load_metrics
    decision_date = load_metrics(cfg)["features"]["cutoffs"]["DECISION"]
    dev = camp.split.isin(["train", "val"])
    th = {"vip": camp.loc[dev, "sum_all"].quantile(rc["vip_quantile"]),
          "unc": ct.loc[dev, "tau_std"].quantile(rc["uncertainty_quantile"])}
    router = ActionRouter(cfg, th)
    m = {"thresholds": th}

    # ---------- kiểm thử đơn vị cho các tình huống biên
    base = {"tau_hat": 0.05, "x": 1, "recency": 10, "sum_all": 1000.0, "tau_std": 0.0, "has_tx": 1, "value_hat": 5.0}
    cases = [
        ("Khách bình thường được chọn", {}, "AUTO_SEND"),
        ("Không được MILP chọn", {"x": 0}, "NO_ACTION"),
        ("Sleeping dog dù được chọn", {"tau_hat": -0.05}, "SUPPRESS"),
        ("Sleeping dog không được chọn", {"tau_hat": -0.05, "x": 0}, "SUPPRESS"),
        ("Vừa mua hôm qua", {"recency": 1}, "DEFER"),
        ("VIP", {"sum_all": th["vip"] + 1}, "HUMAN_REVIEW"),
        ("Bất định cao", {"tau_std": th["unc"] + 1e-3}, "HUMAN_REVIEW"),
        ("Cold-start", {"has_tx": 0}, "HUMAN_REVIEW"),
        ("Xung đột: VIP + vừa mua", {"sum_all": th["vip"] + 1, "recency": 0.5}, "DEFER"),
        ("Xung đột: sleeping dog + VIP", {"sum_all": th["vip"] + 1, "tau_hat": -0.2}, "SUPPRESS"),
        ("Biên: uplift đúng bằng ngưỡng", {"tau_hat": rc["sleeping_dog_tau"]}, "AUTO_SEND"),
        ("Biên: recency đúng bằng ngưỡng", {"recency": rc["recent_purchase_days"]}, "DEFER"),
        ("VIP nhưng không được chọn", {"sum_all": th["vip"] + 1, "x": 0}, "NO_ACTION"),
    ]
    rows = []
    for name, upd, exp in cases:
        a, rs = router.route_one({**base, **upd})
        rows.append({"Tình huống kiểm thử": name, "Kỳ vọng": exp, "Thực tế": a, "Lý do": "|".join(rs),
                     "Kết quả": "PASS" if a == exp else "FAIL"})
    ut = pd.DataFrame(rows)
    save_table(ut, cfg, "t34_router_unit_tests", "Kiểm thử tình huống biên của Action Router")
    m["unit_tests_pass"] = int((ut["Kết quả"] == "PASS").sum()); m["unit_tests_total"] = len(ut)

    summaries = []
    for split in ["locked", "oot"]:
        d = camp.split == split
        df = camp.loc[d, ["recency", "sum_all", "has_tx", "age_band"]].copy()
        df["tau_hat"] = ct.loc[d, "tau_hat"]; df["tau_std"] = ct.loc[d, "tau_std"]
        df["value_hat"] = ds.loc[d, "value_hat"]; df["value_proxy"] = ds.loc[d, "value_proxy"]
        df["cost_i"] = ds.loc[d, "cost_i"]
        if split == "locked":
            sol = pd.read_csv(proc / "milp_solution_locked.csv", dtype={"client_id": str}).set_index("client_id")
            df["x"] = sol.reindex(df.index.astype(str))["x"].values
        else:
            V, cost, seg = df["value_hat"].values, df["cost_i"].values, df["age_band"].values
            K = int(mc["capacity_frac"] * len(df)); B = mc["budget_frac_of_capacity_cost"] * K * cost.mean()
            xg = greedy(V, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"])
            x, _ = solve(V, cost, seg, K, B, mc["segment_max_share"], mc["segment_min_coverage"], mc["solver"],
                         time_limit=mc["time_limit_s"], hint=xg)
            df["x"] = np.round(x).astype(int)
        res = router.route(df)
        res["idempotency_key"] = [idem_key(cfg["campaign_id"], c, cfg["model_version"], decision_date) for c in res.index]
        res["model_version"] = cfg["model_version"]; res["decision_date"] = decision_date
        log_cols = ["action", "reasons", "x", "value_hat", "tau_hat", "tau_std", "idempotency_key", "model_version", "decision_date"]
        res[log_cols].to_csv(proc / f"decision_log_{split}.csv")
        dist = res["action"].value_counts().reindex(ACTIONS).fillna(0).astype(int)
        val = {}
        x_auto = (res["action"] == "AUTO_SEND").astype(float).values
        x_rev = (res["action"] == "HUMAN_REVIEW").astype(float).values
        vp = res["value_proxy"].values
        val["MILP thuần (không rào chắn)"] = float((res["x"].values * vp).sum())
        val["Router - duyệt viên từ chối tất cả"] = float((x_auto * vp).sum())
        val["Router - duyệt viên chấp thuận tất cả"] = float(((x_auto + x_rev) * vp).sum())
        val["Router - duyệt viên chấp thuận nếu V̂>0"] = float(((x_auto + x_rev * (res["value_hat"].values > 0)) * vp).sum())
        for a in ACTIONS:
            summaries.append({"Tập": split, "Hành động": a, "Số khách": int(dist[a]), "Tỷ trọng (%)": 100 * dist[a] / len(res),
                              "V̂ TB": res.loc[res.action == a, "value_hat"].mean(),
                              "Uplift dự báo TB": res.loc[res.action == a, "tau_hat"].mean(),
                              "Giá trị DR nếu gửi cả nhóm (RUB)": float((res["action"].values == a).astype(float) @ vp)})
        rule_hits = {r: int(res["reasons"].str.contains(r).sum()) for r in
                     ["R1_SLEEPING_DOG", "R2_RECENT_PURCHASE", "R3_VIP", "R4_HIGH_UNCERTAINTY", "R5_COLD_START", "Q_REVIEW_OVERFLOW"]}
        m[split] = {"distribution": dist.to_dict(), "value_scenarios": val, "rule_hits": rule_hits,
                    "conflicts": int((res["n_rules"] >= 2).sum()),
                    "milp_selected": int(res["x"].sum()),
                    "milp_selected_suppressed": int(((res["x"] == 1) & (res["action"] == "SUPPRESS")).sum()),
                    "unique_keys": int(res["idempotency_key"].nunique()), "n": len(res)}
        if split == "locked":
            vs = pd.DataFrame([{"Kịch bản": k, "Lợi nhuận tăng thêm DR (RUB)": v} for k, v in val.items()])
            save_table(vs, cfg, "t36_router_value_scenarios", "Tác động của rào chắn nghiệp vụ và Human-in-the-loop lên giá trị (Locked Test)")
            rh = pd.DataFrame([{"Quy tắc": k, "Số khách kích hoạt": v} for k, v in rule_hits.items()])
            save_table(rh, cfg, "t37_rule_hits", "Số lần kích hoạt từng quy tắc nghiệp vụ (Locked Test)")
            cross = pd.crosstab(res["x"].map({1: "MILP chọn", 0: "MILP không chọn"}), res["action"]).reindex(columns=ACTIONS, fill_value=0)
            save_table(cross.reset_index().rename(columns={"x": "Quyết định MILP"}), cfg, "t38_milp_vs_action",
                       "Ma trận chuyển từ quyết định MILP sang hành động cuối cùng")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            left = np.zeros(2)
            for i, a in enumerate(ACTIONS):
                ax.barh(cross.index, cross[a], left=left, color=PALETTE[i], label=a)
                left += cross[a].values
            ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.25)); ax.set_xlabel("Số khách")
            ax.set_title("Từ điểm quyết định đến hành động (Locked Test)", pad=30)
            save_fig(fig, cfg, "f39_milp_to_action", "Luồng chuyển từ quyết định MILP sang hành động sau định tuyến")
            res.head(30)[["action", "reasons", "x", "value_hat", "tau_hat", "idempotency_key"]].reset_index().to_csv(
                cfg["paths"]["tables"] / "t39_decision_log_sample.csv", index=False, encoding="utf-8-sig")
            from utils import _register
            _register(cfg, "tables", "t39_decision_log_sample", {"file": "t39_decision_log_sample.csv",
                                                                 "caption": "Trích nhật ký quyết định (30 bản ghi đầu, Locked Test)"})
            res_locked = res

    sm = pd.DataFrame(summaries)
    save_table(sm, cfg, "t35_action_distribution", "Phân bổ khách hàng theo luồng hành động sau định tuyến")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    piv = sm.pivot(index="Hành động", columns="Tập", values="Tỷ trọng (%)").reindex(ACTIONS)
    piv.plot.bar(ax=ax, color=[PALETTE[0], PALETTE[1]]); ax.set_ylabel("% khách"); ax.set_title("Phân bổ hành động: Locked Test vs Out-of-Time")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, cfg, "f38_action_distribution", "Tỷ trọng các luồng hành động trên Locked Test và Out-of-Time")

    # ---------- kiểm thử idempotency: phát lại yêu cầu
    sample = res_locked[res_locked.action == "AUTO_SEND"].head(1000)
    sent = set(); dup_blocked = 0; sends = 0
    for _ in range(2):  # phát lại toàn bộ hai lần (mô phỏng retry)
        for k in sample["idempotency_key"]:
            if k in sent:
                dup_blocked += 1
            else:
                sent.add(k); sends += 1
    m["idempotency_test"] = {"requests": 2 * len(sample), "sends": sends, "duplicates_blocked": dup_blocked}

    playbook = pd.DataFrame([
        ["PSI > 0,25 trên ≥ 3 đặc trưng", "Trôi dạt dữ liệu nghiêm trọng", "Chuyển toàn bộ AUTO_SEND sang HUMAN_REVIEW, kích hoạt tái huấn luyện", "Data Scientist trực", "4 giờ"],
        ["Tỷ lệ SUPPRESS tăng > 2 lần so với kỳ trước", "Mô hình uplift dịch chuyển", "Tạm dừng chiến dịch, so sánh Current vs Challenger", "Trưởng nhóm phân tích", "1 ngày"],
        ["Bộ giải MILP quá thời gian / không khả thi", "Ràng buộc mâu thuẫn hoặc dữ liệu lỗi", "Dùng nghiệm Greedy + sửa chữa đã kiểm tra khả thi", "Kỹ sư vận hành", "30 phút"],
        ["Hàng chờ duyệt vượt năng lực", "Quá nhiều ca VIP/bất định", "Phần dư → DEFER, ưu tiên theo V̂", "Trưởng nhóm CRM", "Trong ngày"],
        ["P95 độ trễ > ngưỡng", "Sự cố hạ tầng", "Chuyển sang chấm điểm theo lô (batch)", "Kỹ sư MLOps", "2 giờ"],
        ["Khiếu nại khách hàng về tần suất SMS", "Rào chắn R2 chưa đủ", "Tăng ngưỡng recency, rà soát nhật ký theo idempotency key", "CSKH + CRM", "2 ngày"],
        ["Checksum dữ liệu đầu vào không khớp", "Dữ liệu bị thay đổi/hỏng", "Dừng pipeline, khôi phục snapshot gần nhất", "Kỹ sư dữ liệu", "Ngay lập tức"],
    ], columns=["Tín hiệu kích hoạt", "Chẩn đoán", "Hành động ứng phó", "Người chịu trách nhiệm", "SLA"])
    save_table(playbook, cfg, "t40_exception_playbook", "Playbook ngoại lệ và ứng phó sự cố (thiết kế vận hành)")
    save_metrics(cfg, "router", m)
    return m


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
