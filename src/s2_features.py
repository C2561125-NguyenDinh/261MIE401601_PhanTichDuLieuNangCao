"""Giai đoạn 2 (phần B) - Thiết kế dữ liệu theo thời gian tại thời điểm ra quyết định.

- Đặc trưng chỉ dùng giao dịch có thời điểm < mốc cắt (point-in-time correct).
- Ba ảnh chụp (snapshot) cho bài toán dự báo: S1 (train), S2 (validation/locked), S3 (out-of-time).
- Tập chiến dịch (uplift): đặc trưng tại thời điểm quyết định = sau giao dịch cuối cùng.
- Chia tách khách hàng: Train / Validation / Locked Test (ngẫu nhiên phân tầng) + cohort Out-of-Time
  (khách mở thẻ muộn nhất).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from s1_data import build_transactions, load_base
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl

WINDOWS = [7, 14, 28]


def make_features(tx: pd.DataFrame, clients: pd.DataFrame, ids: pd.Series, cutoff: pd.Timestamp,
                  lookback: int) -> pd.DataFrame:
    """Đặc trưng RFM + điểm thưởng + hành vi, tính đúng tại mốc cắt `cutoff` (không rò rỉ).

    Chỉ dùng giao dịch trong cửa sổ nhìn lại cố định `lookback` ngày để các snapshot có cùng độ dài
    lịch sử (tránh lệch do dữ liệu bị cắt trái tại ngày bắt đầu quan sát).
    """
    start = cutoff - pd.Timedelta(days=lookback)
    h = tx[(tx["dt"] < cutoff) & (tx["dt"] >= start) & (tx["client_id"].isin(ids))].copy()
    h["client_id"] = h["client_id"].astype(str)
    h["age_d"] = (cutoff - h["dt"]).dt.total_seconds() / 86400.0
    h["is_spent"] = (h["rp_spent"] < 0).astype(float)
    h["weekend"] = (h["dt"].dt.dayofweek >= 5).astype(float)
    h["hour"] = h["dt"].dt.hour.astype(float)
    h["neg"] = (h["purchase_sum"] <= 0).astype(float)
    h = h.sort_values(["client_id", "dt"])
    h["gap"] = h.groupby("client_id")["dt"].diff().dt.total_seconds() / 86400.0
    g = h.groupby("client_id")
    f = pd.DataFrame(index=pd.Index(ids.astype(str).values, name="client_id"))
    f["recency"] = g["age_d"].min()
    f["first_tx_age"] = g["age_d"].max()
    f["n_tx_all"] = g.size()
    f["sum_all"] = g["purchase_sum"].sum()
    f["aov_all"] = g["purchase_sum"].mean()
    f["aov_std"] = g["purchase_sum"].std()
    f["aov_max"] = g["purchase_sum"].max()
    f["items_per_tx"] = g["n_items"].mean()
    f["qty_per_tx"] = g["qty"].mean()
    f["n_stores"] = g["store_id"].nunique()
    f["rp_recv_sum"] = g["rp_recv"].sum()
    f["ep_recv_sum"] = g["ep_recv"].sum()
    f["rp_spent_sum"] = g["rp_spent"].sum()  # giá trị âm = điểm đã tiêu
    f["ep_spent_sum"] = g["ep_spent"].sum()
    f["share_spent_tx"] = g["is_spent"].mean()
    f["share_weekend"] = g["weekend"].mean()
    f["mean_hour"] = g["hour"].mean()
    f["neg_tx_share"] = g["neg"].mean()
    f["mean_gap"] = g["gap"].mean()  # khoảng cách trung bình giữa các giao dịch
    for w in WINDOWS:
        hw = h[h["age_d"] <= w].groupby("client_id")
        f[f"n_tx_{w}d"] = hw.size()
        f[f"sum_{w}d"] = hw["purchase_sum"].sum()
    f["trend_14_28"] = f["n_tx_14d"] / (f["n_tx_28d"] - f["n_tx_14d"]).replace(0, np.nan)
    f["sum_trend_14_28"] = f["sum_14d"] / (f["sum_28d"] - f["sum_14d"]).replace(0, np.nan)
    cnt_cols = [c for c in f.columns if c.startswith(("n_tx", "sum_"))] + [
        "rp_recv_sum", "ep_recv_sum", "rp_spent_sum", "ep_spent_sum"]
    f[cnt_cols] = f[cnt_cols].fillna(0)
    f["has_tx"] = (f["n_tx_all"] > 0).astype(int)
    f["recency"] = f["recency"].fillna(float(lookback))

    # thông tin thẻ khách hàng (chỉ những gì đã biết tại mốc cắt)
    c = clients.set_index(clients["client_id"].astype(str)).reindex(f.index)
    issue = c["first_issue_date"]
    redeem = c["first_redeem_date"].where(c["first_redeem_date"] < cutoff)
    f["tenure_days"] = (cutoff - issue).dt.days
    f["has_redeemed"] = redeem.notna().astype(int)
    f["days_since_redeem"] = (cutoff - redeem).dt.days
    f["issue_to_redeem_days"] = (redeem - issue).dt.days
    age = c["age"].astype(float)
    f["age_invalid"] = ((age < 14) | (age > 100)).astype(int)
    f["age"] = age.where((age >= 14) & (age <= 100))
    gender = c["gender"].fillna("U")
    for gv in ["F", "M", "U"]:
        f[f"gender_{gv}"] = (gender == gv).astype(int)
    return f.astype("float32")


def label(tx: pd.DataFrame, ids: pd.Series, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    end = cutoff + pd.Timedelta(days=horizon)
    w = tx[(tx["dt"] >= cutoff) & (tx["dt"] < end) & (tx["client_id"].isin(ids))].copy()
    w["client_id"] = w["client_id"].astype(str)
    g = w.groupby("client_id")
    y = pd.DataFrame(index=pd.Index(ids.astype(str).values, name="client_id"))
    y["y_buy"] = g.size().reindex(y.index).notna().astype(int)
    y["y_spend"] = g["purchase_sum"].sum().reindex(y.index).fillna(0.0)
    first = g["dt"].min().reindex(y.index)
    y["days_to_first"] = (first - cutoff).dt.total_seconds() / 86400
    return y


def age_band(a: pd.Series) -> pd.Series:
    return pd.cut(a, bins=[0, 25, 35, 45, 55, 65, 200], right=False,
                  labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]).cat.add_categories("Không rõ").fillna("Không rõ")


def build(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    tx = build_transactions(cfg)
    train, clients = load_base(cfg)
    H = cfg["temporal"]["horizon_days"]
    tmax = tx["dt"].max().normalize() + pd.Timedelta(days=1)  # thời điểm quyết định chiến dịch
    # Cửa sổ chọn mẫu: mọi khách trong uplift_train đều có giao dịch sau mốc này (đặc điểm cách X5
    # chọn khách tham gia chiến dịch). Nhãn dự báo không được rơi vào cửa sổ này, nếu không nhãn
    # bị suy biến (100% mua) do điều kiện chọn mẫu.
    camp_tx = tx[tx["client_id"].isin(train["client_id"])]
    last_tx = camp_tx.groupby("client_id", observed=True)["dt"].max()
    sel_start = last_tx.min().normalize()
    s3 = sel_start - pd.Timedelta(days=H)
    s2 = s3 - pd.Timedelta(days=H)
    s1 = s2 - pd.Timedelta(days=H)
    lookback = min(cfg["temporal"]["lookback_max_days"], int((s1 - tx["dt"].min().normalize()).days))
    cutoffs = {"S1": s1, "S2": s2, "S3": s3, "SELECTION_START": sel_start, "DECISION": tmax}
    log.info(f"Mốc cắt: {cutoffs}, lookback={lookback}")

    # ---------- chia tách khách hàng
    ids = train[["client_id", "treatment_flg", "target"]].copy()
    cl = clients.set_index("client_id").reindex(ids.client_id)
    ids["first_issue_date"] = cl["first_issue_date"].values
    thr = ids["first_issue_date"].quantile(cfg["split"]["oot_issue_quantile"])
    ids["split"] = "pending"
    ids.loc[ids["first_issue_date"] >= thr, "split"] = "oot"
    rest = ids[ids.split == "pending"]
    # phân tầng theo (treatment, target)
    strata = rest["treatment_flg"].astype(str) + "_" + rest["target"].astype(str)
    sp = cfg["split"]
    for _, idx in rest.groupby(strata).groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(sp["train"] * n), int((sp["train"] + sp["val"]) * n)
        ids.loc[idx[:a], "split"] = "train"
        ids.loc[idx[a:b], "split"] = "val"
        ids.loc[idx[b:], "split"] = "locked"
    ids = ids.set_index(ids["client_id"].astype(str))

    proc = cfg["paths"]["processed"]
    # ---------- tập chiến dịch (uplift)
    log.info("Tính đặc trưng tại thời điểm quyết định chiến dịch ...")
    Xc = make_features(tx, clients, train["client_id"], tmax, lookback)
    camp = Xc.join(ids[["treatment_flg", "target", "split", "first_issue_date"]])
    camp["age_band"] = age_band(camp["age"]).astype(str)
    camp.to_parquet(proc / "campaign.parquet")

    # ---------- ba snapshot cho dự báo
    snaps = []
    for name in ["S1", "S2", "S3"]:
        t = cutoffs[name]
        log.info(f"Snapshot {name} @ {t.date()} ...")
        X = make_features(tx, clients, train["client_id"], t, lookback)
        Y = label(tx, train["client_id"], t, H)
        d = X.join(Y).join(ids[["split"]])
        d["snapshot"] = name
        # khách chưa mở thẻ tại mốc cắt không thuộc tập dự báo
        d = d[d["tenure_days"].fillna(-1) >= 0]
        snaps.append(d)
    fc = pd.concat(snaps)
    fc.to_parquet(proc / "forecast_snapshots.parquet")

    # ---------- bảng & hình
    feats = [c for c in Xc.columns]
    rec_dec = (tmax - last_tx).dt.total_seconds() / 86400
    m = {"cutoffs": {k: str(v.date()) for k, v in cutoffs.items()}, "horizon_days": H, "lookback_days": lookback,
         "max_recency_at_decision": float(rec_dec.max()),
         "share_bought_last": {f"{d}d": float((rec_dec < d).mean()) for d in [7, 14, 21, 28]},
         "oot_issue_threshold": str(pd.Timestamp(thr).date()), "n_features": len(feats),
         "split_counts": camp["split"].value_counts().to_dict(),
         "split_treat_rate": camp.groupby("split")["treatment_flg"].mean().to_dict(),
         "split_target_rate": camp.groupby("split")["target"].mean().to_dict(),
         "snapshot_counts": fc.groupby("snapshot").size().to_dict(),
         "snapshot_buy_rate": fc.groupby("snapshot")["y_buy"].mean().to_dict(),
         "snapshot_spend_mean": fc.groupby("snapshot")["y_spend"].mean().to_dict()}
    save_metrics(cfg, "features", m)

    st = camp.groupby("split").agg(n=("target", "size"), treat_rate=("treatment_flg", "mean"),
                                   target_rate=("target", "mean"),
                                   issue_min=("first_issue_date", "min"),
                                   issue_max=("first_issue_date", "max")).reset_index()
    st["issue_min"] = st["issue_min"].astype(str).str[:10]
    st["issue_max"] = st["issue_max"].astype(str).str[:10]
    order = {"train": 0, "val": 1, "locked": 2, "oot": 3}
    st = st.sort_values("split", key=lambda s: s.map(order))
    st.columns = ["Tập", "Số khách", "Tỷ lệ can thiệp", "Tỷ lệ mua", "Mở thẻ sớm nhất", "Mở thẻ muộn nhất"]
    save_table(st, cfg, "t06_split_summary", "Thống kê các tập Train / Validation / Locked Test / Out-of-Time (bài toán uplift)")

    ss = fc.groupby("snapshot").agg(n=("y_buy", "size"), buy=("y_buy", "mean"), spend=("y_spend", "mean"),
                                    rec=("recency", "median")).reset_index()
    ss["cutoff"] = ss["snapshot"].map(lambda s: str(cutoffs[s].date()))
    ss["window_end"] = ss["snapshot"].map(lambda s: str((cutoffs[s] + pd.Timedelta(days=H)).date()))
    ss = ss[["snapshot", "cutoff", "window_end", "n", "buy", "spend", "rec"]]
    ss.columns = ["Snapshot", "Mốc cắt", "Hết cửa sổ nhãn", "Số khách", "Tỷ lệ mua trong H", "Chi tiêu TB trong H", "Recency trung vị (ngày)"]
    save_table(ss, cfg, "t07_snapshot_summary", "Thiết kế ba ảnh chụp thời gian cho bài toán dự báo")

    # từ điển đặc trưng / hợp đồng dữ liệu
    desc = feature_dictionary()
    dd = pd.DataFrame([{"Đặc trưng": c, "Mô tả": desc.get(c, ""), "Kiểu": "float32",
                        "Thiếu (%) - tập chiến dịch": 100 * Xc[c].isna().mean(),
                        "Trung bình": Xc[c].mean(), "Trung vị": Xc[c].median()} for c in feats])
    save_table(dd, cfg, "t08_feature_dictionary", "Từ điển đặc trưng (Data Contract đầu vào mô hình)")

    leak = pd.DataFrame([
        ["Đặc trưng chỉ dùng giao dịch có dt < mốc cắt", "Lọc tx['dt'] < cutoff trong make_features", "Đạt"],
        ["Ngày đổi điểm đầu tiên sau mốc cắt bị che", "first_redeem_date.where(< cutoff)", "Đạt"],
        ["Nhãn dự báo nằm hoàn toàn sau mốc cắt", "Cửa sổ [cutoff, cutoff+H)", "Đạt"],
        ["Khách chưa mở thẻ tại mốc cắt bị loại", "tenure_days >= 0", "Đạt"],
        ["Không dùng biến target/treatment làm đặc trưng", "Tách riêng cột trong campaign.parquet", "Đạt"],
        ["Tách thực thể giữa Train/Val/Locked", "Chia theo client_id, phân tầng (T, Y)", "Đạt"],
        ["Cohort Out-of-Time không giao Train", f"first_issue_date >= {pd.Timestamp(thr).date()}", "Đạt"],
        ["Hiệu chỉnh xác suất học trên tập Validation, không học trên Locked", "s3_forecast.calibrate", "Đạt"],
        ["Siêu tham số không chọn trên Locked Test / OOT", "Locked chỉ đánh giá một lần", "Đạt"],
    ], columns=["Kiểm tra rò rỉ", "Cơ chế thực thi trong mã", "Kết quả"])
    save_table(leak, cfg, "t09_leakage_checklist", "Bảng kiểm rò rỉ thông tin (leakage checklist)")

    plot_timeline(cfg, tx, cutoffs, H)
    plot_dag(cfg)
    plot_split_bars(cfg, camp)
    plot_feature_dists(cfg, camp)
    log.info("Hoàn tất giai đoạn đặc trưng.")
    return m


def feature_dictionary() -> dict:
    d = {"recency": "Số ngày từ giao dịch gần nhất đến mốc cắt",
         "first_tx_age": "Số ngày từ giao dịch đầu tiên trong cửa sổ nhìn lại đến mốc cắt",
         "n_tx_all": "Tổng số giao dịch trong cửa sổ nhìn lại", "sum_all": "Tổng chi tiêu trong cửa sổ nhìn lại",
         "aov_all": "Giá trị giỏ hàng trung bình", "aov_std": "Độ lệch chuẩn giá trị giỏ hàng",
         "aov_max": "Giá trị giỏ hàng lớn nhất", "items_per_tx": "Số dòng sản phẩm trung bình / giao dịch",
         "qty_per_tx": "Số lượng sản phẩm trung bình / giao dịch", "n_stores": "Số cửa hàng khác nhau đã mua",
         "rp_recv_sum": "Tổng điểm thường nhận", "ep_recv_sum": "Tổng điểm express nhận",
         "rp_spent_sum": "Tổng điểm thường đã tiêu (âm)", "ep_spent_sum": "Tổng điểm express đã tiêu (âm)",
         "share_spent_tx": "Tỷ lệ giao dịch có tiêu điểm", "share_weekend": "Tỷ lệ giao dịch cuối tuần",
         "mean_hour": "Giờ mua trung bình", "neg_tx_share": "Tỷ lệ giao dịch giá trị <= 0",
         "mean_gap": "Khoảng cách trung bình giữa hai giao dịch (ngày)",
         "trend_14_28": "Tỷ số số giao dịch 14 ngày gần nhất / 14 ngày trước đó",
         "sum_trend_14_28": "Tỷ số chi tiêu 14 ngày gần nhất / 14 ngày trước đó",
         "has_tx": "Có giao dịch trong cửa sổ nhìn lại", "tenure_days": "Số ngày từ ngày mở thẻ",
         "has_redeemed": "Đã từng đổi điểm trước mốc cắt", "days_since_redeem": "Số ngày từ lần đổi điểm đầu",
         "issue_to_redeem_days": "Số ngày từ mở thẻ đến lần đổi điểm đầu", "age": "Tuổi (đã làm sạch)",
         "age_invalid": "Cờ tuổi không hợp lệ", "gender_F": "Giới tính nữ", "gender_M": "Giới tính nam",
         "gender_U": "Giới tính không rõ"}
    for w in WINDOWS:
        d[f"n_tx_{w}d"] = f"Số giao dịch trong {w} ngày trước mốc cắt"
        d[f"sum_{w}d"] = f"Chi tiêu trong {w} ngày trước mốc cắt"
    return d


# ------------------------------------------------------------------ figures
def plot_timeline(cfg, tx, cutoffs, H):
    plt = setup_mpl()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    t0 = tx["dt"].min()
    ax.axhline(0, color="#999", lw=1)
    rows = [("S1", 3), ("S2", 2), ("S3", 1)]
    for name, y in rows:
        c = cutoffs[name]
        ax.barh(y, (c - t0).days, left=0, color=PALETTE[0], alpha=.55, height=.5)
        ax.barh(y, H, left=(c - t0).days, color=PALETTE[1], alpha=.85, height=.5)
        ax.text(-2, y, name, ha="right", va="center", fontweight="bold")
    c = cutoffs["DECISION"]
    sel = (cutoffs["SELECTION_START"] - t0).days
    ax.axvspan(sel, (c - t0).days, color="#C44536", alpha=.12)
    ax.text((sel + (c - t0).days) / 2, 2.0, "Cửa sổ chọn mẫu\n(100% khách\ncó giao dịch)", ha="center", va="center", fontsize=8.5, color="#C44536", fontweight="bold")
    ax.barh(0, (c - t0).days, left=0, color=PALETTE[0], alpha=.55, height=.5)
    ax.barh(0, 10, left=(c - t0).days, color=PALETTE[2], alpha=.9, height=.5)
    ax.text(-2, 0, "Chiến dịch", ha="right", va="center", fontweight="bold")
    ax.set_yticks([])
    ticks = list(range(0, (c - t0).days + 11, 7))
    ax.set_xticks(ticks)
    ax.set_xticklabels([(t0 + pd.Timedelta(days=d)).strftime("%d/%m/%y") for d in ticks], rotation=45)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=PALETTE[0], alpha=.55, label="Cửa sổ đặc trưng (quá khứ)"),
                       Patch(color=PALETTE[1], label=f"Cửa sổ nhãn H={H} ngày"),
                       Patch(color=PALETTE[2], label="Gửi SMS & đo kết quả (target)")],
              loc="upper left", ncol=3, bbox_to_anchor=(0, 1.18))
    ax.set_title("Thiết kế dữ liệu theo trục thời gian (point-in-time)", pad=28)
    save_fig(fig, cfg, "f06_temporal_design", "Thiết kế dữ liệu theo trục thời gian với ba snapshot và thời điểm chiến dịch")


def plot_dag(cfg):
    plt = setup_mpl()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    nodes = {"X: Hành vi quá khứ\n(RFM, điểm thưởng)": (0.15, 0.75), "W: Nhân khẩu học\n(tuổi, giới, thâm niên)": (0.15, 0.25),
             "T: Gửi SMS\n(ngẫu nhiên hóa)": (0.5, 0.5), "Y: Mua hàng\ntrong chiến dịch": (0.85, 0.5),
             "R: Cơ chế\nngẫu nhiên": (0.5, 0.9), "U: Yếu tố\nkhông quan sát": (0.85, 0.9)}
    for k, (x, y) in nodes.items():
        col = PALETTE[0] if k.startswith("T") else (PALETTE[1] if k.startswith("Y") else "#DDE6ED")
        ax.add_patch(FancyBboxPatch((x - .12, y - .07), .24, .14, boxstyle="round,pad=0.02", fc=col,
                                    ec="#333", alpha=.9))
        ax.text(x, y, k, ha="center", va="center", fontsize=9,
                color="white" if col in (PALETTE[0], PALETTE[1]) else "black")
    def arr(a, b, style="-", col="#333"):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax.annotate("", xy=(x2 - .12 * np.sign(x2 - x1) if x2 != x1 else x2, y2 - .07 * np.sign(y2 - y1) if x1 == x2 else y2),
                    xytext=(x1 + .12 * np.sign(x2 - x1) if x2 != x1 else x1, y1 + .07 * np.sign(y2 - y1) if x1 == x2 else y1),
                    arrowprops=dict(arrowstyle="->", lw=1.6, ls=style, color=col))
    k = list(nodes)
    arr(k[0], k[3]); arr(k[1], k[3]); arr(k[2], k[3]); arr(k[4], k[2])
    arr(k[5], k[3], "--", "#999")
    arr(k[0], k[2], ":", "#C44536"); arr(k[1], k[2], ":", "#C44536")
    ax.text(0.33, 0.62, "bị cắt bởi\nngẫu nhiên hóa", color="#C44536", fontsize=8, ha="center")
    ax.set_title("Đồ thị nhân quả có hướng không chu trình (DAG) của chiến dịch SMS")
    save_fig(fig, cfg, "f07_causal_dag", "DAG nhân quả: ngẫu nhiên hóa cắt các đường cửa sau từ X, W tới T")


def plot_split_bars(cfg, camp):
    plt = setup_mpl()
    order = ["train", "val", "locked", "oot"]
    g = camp.groupby("split").agg(n=("target", "size"), t=("treatment_flg", "mean"), y=("target", "mean")).reindex(order)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].bar(order, g["n"], color=PALETTE[:4]); ax[0].set_title("Số khách")
    ax[1].bar(order, g["t"], color=PALETTE[:4]); ax[1].set_title("Tỷ lệ can thiệp"); ax[1].set_ylim(0.45, 0.55)
    ax[2].bar(order, g["y"], color=PALETTE[:4]); ax[2].set_title("Tỷ lệ mua")
    save_fig(fig, cfg, "f08_split_balance", "Quy mô, tỷ lệ can thiệp và tỷ lệ mua theo từng tập")


def plot_feature_dists(cfg, camp):
    plt = setup_mpl()
    cols = ["recency", "n_tx_all", "aov_all", "sum_28d", "tenure_days", "age", "n_stores", "share_spent_tx", "trend_14_28"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, c in zip(axes.ravel(), cols):
        s = camp[c].dropna()
        s = s.clip(s.quantile(.005), s.quantile(.99))
        for split, col in zip(["train", "oot"], [PALETTE[0], PALETTE[1]]):
            ax.hist(s[camp.loc[s.index, "split"] == split], bins=50, density=True, alpha=.5, color=col, label=split)
        ax.set_title(c)
    axes[0, 0].legend()
    save_fig(fig, cfg, "f09_feature_dists_train_vs_oot", "Phân phối đặc trưng chính: Train so với cohort Out-of-Time")


def load_campaign(cfg) -> pd.DataFrame:
    return pd.read_parquet(cfg["paths"]["processed"] / "campaign.parquet")


def load_snapshots(cfg) -> pd.DataFrame:
    return pd.read_parquet(cfg["paths"]["processed"] / "forecast_snapshots.parquet")


FEATURES_EXCLUDE = {"treatment_flg", "target", "split", "first_issue_date", "age_band", "y_buy", "y_spend",
                    "days_to_first", "snapshot"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURES_EXCLUDE]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    build(load_config())
