"""Giai đoạn 5 - Cấu trúc hóa bài toán quyết định: Predict-Then-Optimize và SPO+.

Hàm giá trị (lợi nhuận tăng thêm kỳ vọng khi gửi SMS cho khách i):
    V_i = tau_i * m * S_i  -  c_contact  -  p1_i * c_incentive
với tau_i: uplift, S_i: chi tiêu kỳ vọng khi mua (dự báo CQR, giai đoạn 3), p1_i: xác suất mua khi được gửi.
Vector chi phí của bài toán tối ưu: c_i = -V_i.
Bài toán con (oracle) trên mỗi lô n khách: min c^T w  s.t.  sum w <= k, 0 <= w <= 1  (nghiệm nguyên = top-k âm).
Nhãn chi phí "thật" không quan sát được ở mức cá nhân -> dùng proxy DR pseudo-outcome (không chệch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from s2_features import feature_cols, load_campaign
from utils import PALETTE, load_config, log, save_fig, save_metrics, save_table, set_seed, setup_mpl


# ------------------------------------------------------------------ oracle
def oracle(C: np.ndarray, k: int) -> np.ndarray:
    """C: (B, n) chi phí. Trả về W: (B, n) nhị phân, chọn tối đa k phần tử chi phí âm nhỏ nhất mỗi lô."""
    C = np.atleast_2d(C)
    W = np.zeros_like(C)
    idx = np.argpartition(C, k - 1, axis=1)[:, :k]
    rows = np.arange(C.shape[0])[:, None]
    W[rows, idx] = (C[rows, idx] < 0).astype(float)
    return W


def spo_plus_loss_grad(Chat, C, k):
    W_true = oracle(C, k)
    W_spo = oracle(2 * Chat - C, k)
    loss = -( (2 * Chat - C) * W_spo).sum(1) + 2 * (Chat * W_true).sum(1) - (C * W_true).sum(1)
    grad = 2 * (W_true - W_spo)
    return loss, grad


def regret(Chat, C, k):
    W_hat, W_true = oracle(Chat, k), oracle(C, k)
    z_hat, z_true = (C * W_hat).sum(1), (C * W_true).sum(1)
    return z_hat - z_true, z_true, W_hat


def make_batches(n_rows, n, rng, reps=1):
    out = []
    for _ in range(reps):
        p = rng.permutation(n_rows)
        nb = n_rows // n
        out.append(p[: nb * n].reshape(nb, n))
    return np.vstack(out)


class Adam:
    def __init__(self, shape, lr):
        self.m, self.v, self.t, self.lr = np.zeros(shape), np.zeros(shape), 0, lr

    def step(self, g):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g ** 2
        mh, vh = self.m / (1 - 0.9 ** self.t), self.v / (1 - 0.999 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + 1e-8)


def value_terms(cfg, df, fc, ct):
    b = cfg["business"]
    S = fc.loc[df.index, "spend_hat"].clip(upper=fc["spend_hat"].quantile(b["aov_cap_quantile"])).values
    p1 = ct.loc[df.index, "p1_hat"].values
    return S, p1


def proxy_cost(cfg, df, fc, ct, psi):
    b = cfg["business"]
    S, p1 = value_terms(cfg, df, fc, ct)
    return -(psi * b["margin_rate"] * S - b["contact_cost"] - p1 * b["incentive_cost"])


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    plt = setup_mpl()
    rng = np.random.default_rng(cfg["seed"])
    proc = cfg["paths"]["processed"]
    camp = load_campaign(cfg)
    fc = pd.read_parquet(proc / "forecast_campaign.parquet")
    ct = pd.read_parquet(proc / "cate_campaign.parquet")
    psi_tr = pd.read_parquet(proc / "dr_pseudo_train.parquet")["psi"]
    cm = load_config  # noqa
    sp = cfg["spo"]
    n, k = sp["batch_size"], int(sp["batch_size"] * sp["k_frac"])
    e = float(camp.loc[camp.split == "train", "treatment_flg"].mean())
    b = cfg["business"]

    # pseudo-outcome ngoài mẫu cho val/locked/oot
    y, t = camp["target"].values.astype(float), camp["treatment_flg"].values.astype(float)
    mu0, mu1 = ct["_mu0"].values, ct["_mu1"].values
    psi_all = pd.Series(mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e), index=camp.index)
    psi_all.loc[psi_tr.index] = psi_tr

    feats = feature_cols(camp)
    Xdf = camp[feats].copy()
    Xdf["spend_hat"] = fc["spend_hat"]
    Xdf["p1_hat"] = ct["p1_hat"]
    Xdf["p_buy_cal"] = fc["p_buy_cal"]
    med = Xdf[camp.split == "train"].median()
    Xdf = Xdf.fillna(med)
    Xs = np.sign(Xdf) * np.log1p(np.abs(Xdf))
    mu_, sd_ = Xs[camp.split == "train"].mean(), Xs[camp.split == "train"].std().replace(0, 1)
    Xs = ((Xs - mu_) / sd_).values
    Xs = np.c_[Xs, np.ones(len(Xs))]

    sets = {s: np.where(camp.split.values == s)[0] for s in ["train", "val", "locked", "oot"]}
    C_all = proxy_cost(cfg, camp, fc, ct, psi_all.values)
    S_all, p1_all = value_terms(cfg, camp, fc, ct)

    # ---------- dự báo chi phí của các phương pháp
    preds = {}
    tr = sets["train"]
    ridge = Ridge(alpha=10.0, fit_intercept=False).fit(Xs[tr], C_all[tr])
    preds["PTO - Ridge (MSE)"] = Xs @ ridge.coef_
    preds["PTO - Plug-in CATE (LightGBM)"] = -(ct["tau_hat"].values * b["margin_rate"] * S_all - b["contact_cost"] - p1_all * b["incentive_cost"])
    preds["Heuristic - Xác suất mua (p_buy)"] = -fc["p_buy_cal"].values  # chọn khách có xác suất mua cao
    preds["Heuristic - Ngẫu nhiên"] = rng.normal(size=len(camp))

    # ---------- SPO+
    theta = ridge.coef_.copy()
    opt = Adam(theta.shape, sp["lr"])
    hist = []
    va_batches = make_batches(len(sets["val"]), n, np.random.default_rng(1), reps=3)
    best_theta, best_val, bad = theta.copy(), np.inf, 0

    def val_regret(th):
        idx = sets["val"][va_batches]
        Chat, C = Xs[idx] @ th, C_all[idx]
        r, z, _ = regret(Chat, C, k)
        return r.sum() / np.abs(z).sum()

    log.info("Huấn luyện SPO+ ...")
    for ep in range(sp["epochs"]):
        batches = make_batches(len(tr), n, rng)
        losses = []
        for i in range(0, len(batches), 16):  # mini-batch 16 bài toán
            idx = tr[batches[i:i + 16]]
            Xb, Cb = Xs[idx], C_all[idx]
            Chat = Xb @ theta
            loss, g = spo_plus_loss_grad(Chat, Cb, k)
            grad = np.einsum("bnp,bn->p", Xb, g) / (idx.size) + sp["l2"] * theta
            theta -= opt.step(grad)
            losses.append(loss.mean())
        vr = val_regret(theta)
        hist.append({"epoch": ep + 1, "spo_loss_train": np.mean(losses), "norm_regret_val": vr})
        if vr < best_val - 1e-6:
            best_val, best_theta, bad = vr, theta.copy(), 0
        else:
            bad += 1
            if bad >= sp["patience"]:
                break
    hist = pd.DataFrame(hist)
    save_table(hist, cfg, "t24_spo_training", "Lịch sử huấn luyện SPO+ (Adam, early stopping theo regret trên Validation)")
    preds["SPO+ (tuyến tính)"] = Xs @ best_theta
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(hist["epoch"], hist["spo_loss_train"], "o-", color=PALETTE[0]); ax[0].set_title("SPO+ loss (train)")
    ax[1].plot(hist["epoch"], hist["norm_regret_val"], "o-", color=PALETTE[1]); ax[1].set_title("Regret chuẩn hóa (Validation)")
    for a in ax: a.set_xlabel("Epoch")
    save_fig(fig, cfg, "f29_spo_training", "Đường huấn luyện SPO+: hàm mất mát và regret trên Validation")

    # ---------- kiểm tra đạo hàm bằng sai phân hữu hạn (hướng ngẫu nhiên, vùng khả vi)
    rng2 = np.random.default_rng(7)
    idx = tr[make_batches(len(tr), n, rng2)[:4]]
    Xb, Cb = Xs[idx], C_all[idx]
    fd_rows = []
    for h in [1e-2, 1e-3, 1e-4, 1e-5]:
        d = rng2.normal(size=theta.shape); d /= np.linalg.norm(d)
        f = lambda th: spo_plus_loss_grad(Xb @ th, Cb, k)[0].sum()
        num = (f(best_theta + h * d) - f(best_theta - h * d)) / (2 * h)
        _, g = spo_plus_loss_grad(Xb @ best_theta, Cb, k)
        ana = np.einsum("bnp,bn->p", Xb, g) @ d
        fd_rows.append({"Bước h": h, "Đạo hàm số": num, "Đạo hàm giải tích": ana,
                        "Sai số tương đối": abs(num - ana) / max(1e-12, abs(ana))})
    save_table(pd.DataFrame(fd_rows), cfg, "t25_finite_difference", "Kiểm tra đạo hàm dưới SPO+ bằng sai phân hữu hạn trung tâm")

    # ---------- đánh giá
    rows = []
    eval_batches = {s: make_batches(len(sets[s]), n, np.random.default_rng(11), reps=5) for s in ["val", "locked", "oot"]}
    for name, ch in preds.items():
        for s in ["val", "locked", "oot"]:
            idx = sets[s][eval_batches[s]]
            r, z, W = regret(ch[idx], C_all[idx], k)
            mse = np.mean((ch[sets[s]] - C_all[sets[s]]) ** 2) if not name.startswith("Heuristic") else np.nan
            rows.append({"Phương pháp": name, "Tập": s, "Regret TB / lô (RUB)": r.mean(),
                         "Regret chuẩn hóa": r.sum() / np.abs(z).sum(),
                         "Giá trị DR TB / lô (RUB)": -(C_all[idx] * W).sum(1).mean(),
                         "Giá trị oracle-proxy / lô (RUB)": -z.mean(), "Tỷ lệ chọn": W.mean(), "MSE chi phí": mse})
    ev = pd.DataFrame(rows)
    save_table(ev, cfg, "t26_pto_vs_spo", "So sánh Predict-Then-Optimize, SPO+ và heuristic theo regret và giá trị quyết định")
    m = {"k": k, "n": n, "spo_epochs_run": len(hist), "spo_best_val_regret": best_val, "eval": ev.to_dict(orient="records"),
         "finite_diff": fd_rows, "e_train": e}
    cand = ev[(ev["Tập"] == "val") & (~ev["Phương pháp"].str.startswith("Heuristic"))]
    best = cand.sort_values("Regret chuẩn hóa").iloc[0]["Phương pháp"]
    m["best_method"] = best

    fig, ax = plt.subplots(figsize=(9, 4))
    piv = ev.pivot(index="Phương pháp", columns="Tập", values="Regret chuẩn hóa")[["val", "locked", "oot"]]
    piv.plot.barh(ax=ax, color=PALETTE[:3]); ax.set_xlabel("Regret chuẩn hóa (thấp hơn = tốt hơn)")
    ax.set_title("Regret chuẩn hóa theo phương pháp và tập dữ liệu")
    save_fig(fig, cfg, "f30_regret_comparison", "Regret chuẩn hóa của các phương pháp ra quyết định")

    # ---------- quan hệ phi tuyến giữa sai số dự báo và regret
    base = preds["PTO - Plug-in CATE (LightGBM)"]
    idx = sets["locked"][eval_batches["locked"]]
    rows = []
    sd_c = np.std(base[sets["locked"]])
    for kind in ["Nhiễu ngẫu nhiên", "Lệch dịch chuyển (bias)", "Co giãn (scale)"]:
        for lvl in np.linspace(0, 2, 11):
            if kind == "Nhiễu ngẫu nhiên":
                ch = base + np.random.default_rng(3).normal(scale=lvl * sd_c, size=len(base))
            elif kind == "Lệch dịch chuyển (bias)":
                ch = base + lvl * sd_c
            else:
                ch = base * (1 + 2 * lvl)
            r, z, W = regret(ch[idx], C_all[idx], k)
            rows.append({"Loại sai lệch": kind, "Mức": lvl, "MSE so với plug-in": np.mean((ch - base)[sets["locked"]] ** 2),
                         "MSE so với proxy": np.mean((ch - C_all)[sets["locked"]] ** 2), "Regret chuẩn hóa": r.sum() / np.abs(z).sum(),
                         "Tỷ lệ chọn": W.mean()})
    nl = pd.DataFrame(rows)
    save_table(nl, cfg, "t27_error_vs_regret", "Thực nghiệm: cùng mức sai số dự báo nhưng tổn thất quyết định khác nhau")
    m["error_vs_regret"] = nl.to_dict(orient="records")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (kind, g) in enumerate(nl.groupby("Loại sai lệch")):
        ax[0].plot(g["MSE so với plug-in"], g["Regret chuẩn hóa"], "o-", color=PALETTE[i], label=kind)
        ax[1].plot(g["Mức"], g["Tỷ lệ chọn"], "o-", color=PALETTE[i], label=kind)
    ax[0].set_xscale("symlog"); ax[0].set_xlabel("MSE thêm vào (RUB²)"); ax[0].set_ylabel("Regret chuẩn hóa"); ax[0].legend()
    ax[0].set_title("Sai số dự báo ↔ tổn thất quyết định")
    ax[1].set_xlabel("Mức sai lệch (đơn vị độ lệch chuẩn)"); ax[1].set_ylabel("Tỷ lệ khách được chọn"); ax[1].legend()
    ax[1].set_title("Ảnh hưởng tới quyết định chọn")
    save_fig(fig, cfg, "f31_error_vs_regret", "Quan hệ phi tuyến giữa sai số dự báo và regret")

    # phân phối giá trị V̂ của phương pháp tốt nhất
    out = pd.DataFrame(index=camp.index)
    for name, ch in preds.items():
        out["chat_" + name] = ch
    out["value_hat"] = -preds[best]
    out["value_proxy"] = -C_all
    out["spend_cap"] = S_all
    out["p1_hat"] = p1_all
    out["cost_i"] = b["contact_cost"] + p1_all * b["incentive_cost"]
    out["psi"] = psi_all.values
    out.to_parquet(proc / "decision_scores.parquet")
    m["value_hat_locked"] = {"mean": out.loc[camp.split == "locked", "value_hat"].mean(),
                             "share_pos": (out.loc[camp.split == "locked", "value_hat"] > 0).mean()}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    v = out.loc[camp.split == "locked", "value_hat"]
    ax.hist(v.clip(v.quantile(.005), v.quantile(.995)), bins=80, color=PALETTE[0])
    ax.axvline(0, color="k", lw=1); ax.set_xlabel("Giá trị tăng thêm kỳ vọng V̂ (RUB)")
    ax.set_title(f"Phân phối điểm quyết định (Decision Score) - {best}")
    save_fig(fig, cfg, "f32_decision_score_dist", "Phân phối điểm quyết định (giá trị tăng thêm kỳ vọng) trên Locked Test")
    joblib.dump({"theta": best_theta, "ridge": ridge.coef_, "mu": mu_, "sd": sd_, "med": med, "cols": list(Xdf.columns)},
                cfg["paths"]["models"] / "spo.joblib")
    save_metrics(cfg, "spo", m)
    return m


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
