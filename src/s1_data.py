"""Giai đoạn 2 (phần A) - Thu thập dữ liệu, kiểm chứng SHA-256, chuẩn hóa về mức giao dịch, hồ sơ dữ liệu.

Nguồn: X5 RetailHero Uplift Modeling (bản phân phối công khai của scikit-uplift).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed,
                   setup_mpl, sha256_file, timer)

PURCHASE_COLS = ["client_id", "transaction_id", "transaction_datetime", "regular_points_received",
                 "express_points_received", "regular_points_spent", "express_points_spent",
                 "purchase_sum", "store_id", "product_quantity"]


# ------------------------------------------------------------------ download
def download(cfg: dict) -> pd.DataFrame:
    raw = cfg["paths"]["raw"]
    rows = []
    for key, fname in cfg["data"]["files"].items():
        dest = raw / fname
        if not dest.exists():
            url = f"{cfg['data']['base_url']}/{fname}"
            log.info(f"Tải {url}")
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
        rows.append({"file": fname, "role": key, "size_mb": dest.stat().st_size / 2**20,
                     "sha256": sha256_file(dest)})
    df = pd.DataFrame(rows)
    save_table(df, cfg, "t01_raw_checksums", "Bảng băm SHA-256 của các tệp dữ liệu gốc")
    return df


# ------------------------------------------------------------------ transactions
def build_transactions(cfg: dict) -> pd.DataFrame:
    """Gộp dữ liệu mức sản phẩm (purchases) về mức giao dịch, đọc theo khối để tiết kiệm RAM."""
    out = cfg["paths"]["processed"] / "transactions.parquet"
    if out.exists():
        return pd.read_parquet(out)
    path = cfg["paths"]["raw"] / cfg["data"]["files"]["purchases"]
    parts, n_rows = [], 0
    reader = pd.read_csv(path, usecols=PURCHASE_COLS, chunksize=cfg["data"]["chunksize"],
                         dtype={"client_id": str, "transaction_id": str, "store_id": str})
    for i, ch in enumerate(reader):
        n_rows += len(ch)
        g = ch.groupby(["client_id", "transaction_id"], sort=False).agg(
            dt=("transaction_datetime", "first"), purchase_sum=("purchase_sum", "first"),
            store_id=("store_id", "first"), rp_recv=("regular_points_received", "first"),
            ep_recv=("express_points_received", "first"), rp_spent=("regular_points_spent", "first"),
            ep_spent=("express_points_spent", "first"), n_items=("product_quantity", "size"),
            qty=("product_quantity", "sum"))
        parts.append(g)
        log.info(f"  khối {i}: {n_rows:,} dòng sản phẩm")
    tx = pd.concat(parts)
    # giao dịch bị cắt ngang giữa hai khối -> gộp lại
    tx = tx.groupby(level=[0, 1], sort=False).agg(
        dt=("dt", "first"), purchase_sum=("purchase_sum", "first"), store_id=("store_id", "first"),
        rp_recv=("rp_recv", "first"), ep_recv=("ep_recv", "first"), rp_spent=("rp_spent", "first"),
        ep_spent=("ep_spent", "first"), n_items=("n_items", "sum"), qty=("qty", "sum")).reset_index()
    tx["dt"] = pd.to_datetime(tx["dt"])
    for c in ["client_id", "store_id"]:
        tx[c] = tx[c].astype("category")
    tx = tx.drop(columns=["transaction_id"])
    tx.to_parquet(out, index=False)
    save_metrics(cfg, "data_raw_rows", {"purchase_rows": n_rows, "transactions": len(tx)})
    return tx


def load_base(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = cfg["paths"]["raw"]
    train = pd.read_csv(raw / cfg["data"]["files"]["train"], dtype={"client_id": str})
    clients = pd.read_csv(raw / cfg["data"]["files"]["clients"],
                          parse_dates=["first_issue_date", "first_redeem_date"], dtype={"client_id": str})
    return train, clients


# ------------------------------------------------------------------ profiling
def profile(cfg: dict, train: pd.DataFrame, clients: pd.DataFrame, tx: pd.DataFrame) -> dict:
    plt = setup_mpl()
    m: dict = {}
    m["n_train_clients"] = len(train)
    m["n_clients_table"] = len(clients)
    m["treatment_rate"] = train["treatment_flg"].mean()
    m["target_rate"] = train["target"].mean()
    m["target_rate_treated"] = train.loc[train.treatment_flg == 1, "target"].mean()
    m["target_rate_control"] = train.loc[train.treatment_flg == 0, "target"].mean()
    m["naive_ate"] = m["target_rate_treated"] - m["target_rate_control"]
    m["tx_min_date"] = str(tx["dt"].min())
    m["tx_max_date"] = str(tx["dt"].max())
    m["n_transactions"] = len(tx)
    m["n_tx_clients"] = tx["client_id"].nunique()
    m["n_stores"] = tx["store_id"].nunique()
    m["mean_purchase_sum"] = tx["purchase_sum"].mean()
    m["median_purchase_sum"] = tx["purchase_sum"].median()
    m["mean_items_per_tx"] = tx["n_items"].mean()

    # Bảng 2x2 treatment x target
    ct = pd.crosstab(train["treatment_flg"], train["target"], margins=True)
    ct.index = ["Đối chứng (T=0)", "Can thiệp (T=1)", "Tổng"]
    ct.columns = ["Không mua (Y=0)", "Mua (Y=1)", "Tổng"]
    save_table(ct.reset_index().rename(columns={"index": "Nhóm"}), cfg, "t02_treatment_target",
               "Phân bố khách hàng theo nhóm can thiệp và biến kết quả")

    # Hồ sơ biến của bảng clients
    prof = []
    for c in clients.columns:
        s = clients[c]
        prof.append({"Biến": c, "Kiểu": str(s.dtype), "Thiếu (%)": 100 * s.isna().mean(),
                     "Số giá trị khác nhau": s.nunique(),
                     "Min": str(s.min()) if s.dtype != object else "",
                     "Max": str(s.max()) if s.dtype != object else ""})
    save_table(pd.DataFrame(prof), cfg, "t03_clients_profile", "Hồ sơ dữ liệu bảng khách hàng (clients)")

    txp = []
    for c in ["purchase_sum", "n_items", "qty", "rp_recv", "ep_recv", "rp_spent", "ep_spent"]:
        s = tx[c]
        txp.append({"Biến": c, "Trung bình": s.mean(), "Độ lệch chuẩn": s.std(), "Min": s.min(),
                    "P25": s.quantile(.25), "Trung vị": s.median(), "P75": s.quantile(.75),
                    "P99": s.quantile(.99), "Max": s.max(), "Thiếu (%)": 100 * s.isna().mean()})
    save_table(pd.DataFrame(txp), cfg, "t04_transactions_profile", "Thống kê mô tả dữ liệu giao dịch")

    # chất lượng dữ liệu
    ages = clients["age"]
    m["age_invalid_share"] = float(((ages < 14) | (ages > 100)).mean())
    m["redeem_missing_share"] = float(clients["first_redeem_date"].isna().mean())
    m["neg_purchase_share"] = float((tx["purchase_sum"] <= 0).mean())
    dq = pd.DataFrame([
        {"Kiểm tra": "Tuổi ngoài khoảng [14, 100]", "Tỷ lệ (%)": 100 * m["age_invalid_share"],
         "Xử lý": "Gán NaN, thêm cờ age_invalid"},
        {"Kiểm tra": "Chưa từng đổi điểm (first_redeem_date thiếu)", "Tỷ lệ (%)": 100 * m["redeem_missing_share"],
         "Xử lý": "Cờ has_redeemed; số ngày = NaN"},
        {"Kiểm tra": "Giao dịch có purchase_sum <= 0", "Tỷ lệ (%)": 100 * m["neg_purchase_share"],
         "Xử lý": "Giữ lại (hoàn trả/điều chỉnh), tách đặc trưng"},
        {"Kiểm tra": "Khách trong uplift_train không có giao dịch",
         "Tỷ lệ (%)": 100 * (1 - train.client_id.isin(tx.client_id.cat.categories).mean()),
         "Xử lý": "Đặc trưng giao dịch = 0, recency = trần"},
        {"Kiểm tra": "Trùng client_id trong uplift_train", "Tỷ lệ (%)": 100 * train.client_id.duplicated().mean(),
         "Xử lý": "Không phát sinh"},
    ])
    save_table(dq, cfg, "t05_data_quality", "Bảng kiểm chất lượng dữ liệu")

    # ---- hình: giao dịch theo ngày
    daily = tx.set_index("dt").resample("D").agg({"purchase_sum": ["size", "sum"]})
    daily.columns = ["n_tx", "revenue"]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax[0].plot(daily.index, daily["n_tx"], color=PALETTE[0])
    ax[0].set_ylabel("Số giao dịch / ngày")
    ax[1].plot(daily.index, daily["revenue"] / 1e6, color=PALETTE[1])
    ax[1].set_ylabel("Doanh thu (triệu RUB)")
    ax[0].set_title("Hoạt động giao dịch theo ngày trong toàn kỳ quan sát")
    save_fig(fig, cfg, "f01_daily_transactions", "Số giao dịch và doanh thu theo ngày")

    # ---- hình: phân phối tuổi, giới tính, ngày mở thẻ
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ages.where((ages >= 14) & (ages <= 100)).plot.hist(bins=60, ax=ax[0], color=PALETTE[0])
    ax[0].set_title("Tuổi (sau làm sạch)")
    clients["gender"].value_counts().plot.bar(ax=ax[1], color=PALETTE[2])
    ax[1].set_title("Giới tính")
    clients.set_index("first_issue_date").resample("ME").size().plot(ax=ax[2], color=PALETTE[3])
    ax[2].set_title("Số thẻ phát hành / tháng")
    save_fig(fig, cfg, "f02_client_demographics", "Đặc điểm nhân khẩu học và thời điểm mở thẻ")

    # ---- hình: phân phối giá trị giao dịch & giờ
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    tx["purchase_sum"].clip(0, tx["purchase_sum"].quantile(.99)).plot.hist(bins=80, ax=ax[0], color=PALETTE[1])
    ax[0].set_title("Giá trị giao dịch (cắt P99)")
    tx["dt"].dt.hour.value_counts().sort_index().plot.bar(ax=ax[1], color=PALETTE[0])
    ax[1].set_title("Giao dịch theo giờ")
    dow = tx["dt"].dt.dayofweek.value_counts().sort_index()
    dow.index = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    dow.plot.bar(ax=ax[2], color=PALETTE[2])
    ax[2].set_title("Giao dịch theo ngày trong tuần")
    save_fig(fig, cfg, "f03_tx_distributions", "Phân phối giá trị giao dịch, giờ và ngày trong tuần")

    # ---- hình: tỷ lệ mua theo nhóm
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    r = train.groupby("treatment_flg")["target"].agg(["mean", "count"])
    se = np.sqrt(r["mean"] * (1 - r["mean"]) / r["count"])
    ax.bar(["Đối chứng", "Can thiệp"], r["mean"], yerr=1.96 * se, color=[PALETTE[5], PALETTE[0]], capsize=6)
    for i, v in enumerate(r["mean"]):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
    ax.set_ylabel("Tỷ lệ mua hàng (target)")
    ax.set_title("Tỷ lệ mua theo nhóm thí nghiệm (KTC 95%)")
    save_fig(fig, cfg, "f04_target_by_group", "Tỷ lệ mua hàng theo nhóm can thiệp và đối chứng")

    # ---- số giao dịch mỗi khách
    per = tx.groupby("client_id", observed=True).size()
    m["tx_per_client_mean"] = per.mean()
    m["tx_per_client_median"] = per.median()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    per.clip(upper=per.quantile(.99)).plot.hist(bins=60, ax=ax, color=PALETTE[3])
    ax.set_title("Số giao dịch mỗi khách hàng (cắt P99)")
    ax.set_xlabel("Số giao dịch")
    save_fig(fig, cfg, "f05_tx_per_client", "Phân phối số giao dịch trên mỗi khách hàng")

    save_metrics(cfg, "data_profile", m)
    return m


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    with timer("download"):
        download(cfg)
    with timer("transactions"):
        tx = build_transactions(cfg)
    train, clients = load_base(cfg)
    profile(cfg, train, clients, tx)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
