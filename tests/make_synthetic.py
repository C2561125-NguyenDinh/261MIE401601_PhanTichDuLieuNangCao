"""Sinh dữ liệu TỔNG HỢP cùng lược đồ X5 để KIỂM THỬ KHÓI (smoke test) pipeline.

KHÔNG dùng cho báo cáo. Chạy:  X5_SMOKE=1 python tests/make_synthetic.py && X5_SMOKE=1 python run_all.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data_smoke" / "raw"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ids = [f"{i:010x}" for i in rng.choice(16**8, N, replace=False)]
issue = pd.Timestamp("2017-01-01") + pd.to_timedelta(rng.integers(0, 680, N), unit="D")
redeem = issue + pd.to_timedelta(rng.integers(5, 400, N), unit="D")
redeem = pd.Series(redeem).where(rng.random(N) < 0.6)
age = rng.normal(45, 13, N).round().astype(int); age[rng.random(N) < 0.01] = 1900
clients = pd.DataFrame({"client_id": ids, "first_issue_date": issue, "first_redeem_date": redeem, "age": age,
                        "gender": rng.choice(["F", "M", "U"], N, p=[.45, .3, .25])})
act = rng.gamma(2, 1.2, N)  # mức độ hoạt động
start, end = pd.Timestamp("2018-11-21"), pd.Timestamp("2019-03-18 23:00")
rows = []
for i, cid in enumerate(ids):
    n = rng.poisson(act[i] * 8)
    t = start + pd.to_timedelta(rng.random(n) * (end - start).total_seconds(), unit="s")
    t = t[t >= issue[i]]
    for j, tt in enumerate(t):
        k = rng.integers(1, 6)
        ps = round(float(rng.gamma(2, 150)), 2)
        for _ in range(k):
            rows.append((cid, f"tx{i}_{j}", tt, rng.integers(0, 10), 0, -rng.integers(0, 2) * 5, 0, ps,
                         f"s{rng.integers(0, 50)}", f"p{rng.integers(0, 500)}", 1, ps, 0))
purch = pd.DataFrame(rows, columns=["client_id", "transaction_id", "transaction_datetime", "regular_points_received",
                                    "express_points_received", "regular_points_spent", "express_points_spent",
                                    "purchase_sum", "store_id", "product_id", "product_quantity", "trn_sum_from_iss", "trn_sum_from_red"])
T = rng.integers(0, 2, N)
p0 = 1 / (1 + np.exp(-(-0.5 + 0.6 * np.log1p(act))))
tau = 0.08 * (act < 1.5) - 0.03 * (act > 4)
Y = (rng.random(N) < np.clip(p0 + T * tau, 0, 1)).astype(int)
pd.DataFrame({"client_id": ids, "treatment_flg": T, "target": Y}).to_csv(out / "uplift_train.csv.gz", index=False)
clients.to_csv(out / "clients.csv.gz", index=False)
purch.to_csv(out / "purchases.csv.gz", index=False)
print("synthetic:", len(clients), "clients,", len(purch), "purchase rows")
