"""Giai đoạn 1 - Thiết kế bài toán hướng đến quyết định và hành động.

Xuất: Decision Contract, sơ đồ chu trình 8 giai đoạn, kiến trúc hệ thống, Graphic Abstract.
(Các hình ở đây là sơ đồ mô tả thiết kế, không chứa số liệu thực nghiệm.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import yaml

from utils import PALETTE, ROOT, load_config, save_fig, save_metrics, save_table, setup_mpl


def box(ax, x, y, w, h, text, fc, tc="white", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p1, p2, col="#333", style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, color=col, lw=lw, linestyle=ls))


def run(cfg: dict) -> None:
    plt = setup_mpl()
    b, mc, rc = cfg["business"], cfg["milp"], cfg["router"]
    save_metrics(cfg, "config", yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))

    contract = pd.DataFrame([
        ["Chủ thể quyết định", "Bộ phận CRM / Marketing giữ chân khách hàng của chuỗi bán lẻ X5"],
        ["Quyết định cần đưa ra", "Gửi hay không gửi SMS ưu đãi cho từng khách hàng trong chiến dịch"],
        ["Đơn vị phân tích", "Khách hàng (client_id) có thẻ thành viên"],
        ["Thời điểm quyết định (t0)", "Ngày kế tiếp sau giao dịch cuối cùng quan sát được"],
        ["Kỳ dự báo", f"H = {cfg['temporal']['horizon_days']} ngày (dự báo mua hàng); cửa sổ chiến dịch theo dữ liệu X5"],
        ["Biến can thiệp T", "treatment_flg ∈ {0, 1}: gửi SMS (ngẫu nhiên hóa trong dữ liệu lịch sử)"],
        ["Biến kết quả Y", "target ∈ {0, 1}: khách có mua hàng trong cửa sổ chiến dịch"],
        ["Đại lượng nhân quả", "CATE τ(x) = E[Y(1) − Y(0) | X = x]; ATE, ATT, ATC"],
        ["Hàm giá trị V", f"V_i = τ_i · m · S_i − c_contact − p1_i · c_incentive; m = {b['margin_rate']}, c_contact = {b['contact_cost']} RUB, c_incentive = {b['incentive_cost']} RUB"],
        ["Tập hành động A", "AUTO_SEND, HUMAN_REVIEW, DEFER, SUPPRESS, NO_ACTION"],
        ["Ràng buộc vận hành", f"Năng lực gửi K = {mc['capacity_frac']:.0%} số khách; ngân sách B = {mc['budget_frac_of_capacity_cost']:.0%}·K·chi phí TB; "
                               f"mỗi nhóm tuổi ≤ {mc['segment_max_share']:.0%}·K và ≥ {mc['segment_min_coverage']:.0%} số khách của nhóm; năng lực duyệt {rc['review_capacity']} ca"],
        ["Rào chắn nghiệp vụ", f"Chặn 'sleeping dog' (τ̂ < {rc['sleeping_dog_tau']}); hoãn khách vừa mua ≤ {rc['recent_purchase_days']} ngày; VIP & bất định cao → người duyệt"],
        ["Chỉ số đánh giá", "AUC, Brier, ECE, độ phủ CQR; Qini, AUUC, uplift@k; regret; lợi nhuận tăng thêm DR, ROI tăng thêm; P50/P95/P99; PSI, Action TVD"],
        ["Tiêu chí chấp nhận", "Xem bảng Release Gates (cổng phát hành) trong giai đoạn 8"],
        ["Hợp đồng dữ liệu", "uplift_train, clients, purchases (X5 RetailHero) - kiểm chứng SHA-256 khi nạp"],
    ], columns=["Thành phần", "Đặc tả"])
    save_table(contract, cfg, "t00_decision_contract", "Đặc tả bài toán quyết định - hành động (Decision Contract)")

    stages = pd.DataFrame([
        ["1", "Thiết kế bài toán quyết định - hành động", "src/s0_design.py", "Decision Contract, sơ đồ kiến trúc, Graphic Abstract"],
        ["2", "Thiết kế dữ liệu theo thời gian & nhận dạng nhân quả", "src/s1_data.py, src/s2_features.py", "Checksum, hồ sơ dữ liệu, snapshot S1–S3, chia tách 4 tập, DAG, bảng kiểm rò rỉ"],
        ["3", "Dự báo nâng cao, hiệu chỉnh, định lượng bất định", "src/s3_forecast.py", "LightGBM, Platt/Isotonic, ECE/Brier, CQR, net benefit"],
        ["4", "Phân tích nhân quả & uplift", "src/s4_causal.py", "SMD, ATE/ATT/ATC (Neyman, AIPW, DoubleML), phản bác, 6 meta-learner, Qini/AUUC"],
        ["5", "Predict-Then-Optimize & SPO+", "src/s5_spo.py", "Hàm giá trị, oracle top-k, SPO+ (Adam), regret, sai phân hữu hạn"],
        ["6", "Tối ưu phân bổ nguồn lực MILP", "src/s6_milp.py", "Cái túi đa chiều (SCIP), Greedy, nới lỏng LP, giá đối ngẫu, độ nhạy, độ ổn định"],
        ["7", "Định tuyến hành động & Human-in-the-loop", "src/s7_router.py", "Quy tắc ưu tiên, kiểm thử tình huống biên, idempotency, nhật ký, playbook"],
        ["8", "Triển khai, đo lường thực chứng, giám sát", "src/s8_mlops.py", "Checksum mô hình, độ trễ P50/P95/P99, PSI, TVD, độ trễ nhãn, Challenger, Release Gates"],
    ], columns=["Giai đoạn", "Nội dung", "Mô-đun mã nguồn", "Đầu ra chính"])
    save_table(stages, cfg, "t00b_stage_mapping", "Ánh xạ chu trình tám giai đoạn sang mô-đun mã nguồn và đầu ra")

    metrics = pd.DataFrame([
        ["Chất lượng dự báo", "AUC, PR-AUC, Log loss", "Khả năng phân biệt khách mua/không mua", "Validation, Locked, OOT"],
        ["Hiệu chỉnh xác suất", "Brier, ECE, MCE, biểu đồ độ tin cậy", "Xác suất dự báo khớp tần suất thực", "Locked, OOT"],
        ["Bất định", "Độ phủ & độ rộng khoảng CQR", "Khoảng chi tiêu đạt độ phủ mục tiêu 90%", "Locked, OOT"],
        ["Giá trị dự báo", "Net benefit theo ngưỡng", "Dự báo có hữu ích cho quyết định", "Locked"],
        ["Nhân quả trung bình", "ATE/ATT/ATC + KTC 95%", "Tác động trung bình của SMS", "Train+Val, Locked, OOT"],
        ["Uplift", "Qini chuẩn hóa, AUUC, uplift@k, thập phân vị", "Xếp hạng theo tác động biên", "Validation, Locked, OOT"],
        ["Tổn thất quyết định", "Regret, regret chuẩn hóa", "Chi phí của quyết định dựa trên dự báo", "Validation, Locked, OOT"],
        ["Giá trị thực hiện", "Lợi nhuận tăng thêm DR + KTC bootstrap, ROI tăng thêm", "Giá trị kinh tế của chính sách", "Locked (policy validation), OOT"],
        ["Tối ưu hóa", "Khoảng cách LP–MILP, Greedy–MILP, thời gian giải, Jaccard", "Chất lượng và độ ổn định nghiệm", "Locked"],
        ["Vận hành", "P50/P95/P99 độ trễ, idempotency", "Đáp ứng yêu cầu thời gian thực, không gửi trùng", "Mô phỏng trực tuyến"],
        ["Trôi dạt", "PSI, Brier/ECE theo thời gian, Action TVD, độ trễ nhãn", "Phát hiện dịch chuyển dữ liệu, mô hình, chính sách", "Train→Locked/OOT, S1→S3"],
        ["Tái huấn luyện", "Current vs Challenger (AUC, ECE, Qini)", "Có nên thăng hạng mô hình mới", "OOT"],
    ], columns=["Tầng", "Chỉ số", "Câu hỏi trả lời", "Tập đánh giá"])
    save_table(metrics, cfg, "t00c_metric_system", "Hệ thống chỉ số đo lường đa tầng")

    # --------- chu trình 8 giai đoạn
    fig, ax = plt.subplots(figsize=(9, 9)); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.axis("off"); ax.set_aspect("equal")
    stages = ["1. Thiết kế bài toán\nquyết định - hành động", "2. Dữ liệu theo thời gian\n& nhận dạng nhân quả",
              "3. Dự báo, hiệu chỉnh\n& bất định", "4. Nhân quả & uplift\n(ATE/CATE)", "5. Predict-Then-Optimize\n& SPO+",
              "6. Tối ưu phân bổ\nMILP", "7. Action Router\nGuardrails & HITL", "8. Triển khai, đo lường\n& giám sát (MLOps)"]
    pos = []
    for i, s in enumerate(stages):
        ang = np.pi / 2 - i * 2 * np.pi / 8
        x, y = np.cos(ang), np.sin(ang)
        pos.append((x, y))
        box(ax, x - .37, y - .16, .74, .32, s, PALETTE[i % len(PALETTE)], fs=8.5, bold=True,
            tc="black" if PALETTE[i % len(PALETTE)] in ("#F2CC8F",) else "white")
    for i in range(8):
        a1 = np.pi / 2 - i * 2 * np.pi / 8 - 0.2; a2 = np.pi / 2 - (i + 1) * 2 * np.pi / 8 + 0.2
        arrow(ax, (0.72 * np.cos(a1), 0.72 * np.sin(a1)), (0.72 * np.cos(a2), 0.72 * np.sin(a2)), lw=2)
    ax.text(0, 0.05, "Vòng phản hồi\nquyết định - hành động", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0, -0.25, "Dự báo → Nhân quả → Tối ưu → Hành động → Giám sát", ha="center", fontsize=8.5, style="italic")
    ax.set_title("Chu trình tám giai đoạn của hệ thống phân tích")
    save_fig(fig, cfg, "f00a_eight_stage_cycle", "Chu trình tám giai đoạn của hệ thống phân tích quyết định - hành động")

    # --------- kiến trúc hệ thống
    fig, ax = plt.subplots(figsize=(13, 7)); ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.3), 2.6, 6.3, boxstyle="round,pad=0.02", fc="#F4F6F8", ec="#999"))
    ax.text(1.4, 6.35, "TẦNG DỮ LIỆU", ha="center", fontweight="bold")
    for i, t in enumerate(["uplift_train.csv\n(T, Y)", "clients.csv\n(nhân khẩu học)", "purchases.csv\n(giao dịch mức SP)", "Kiểm chứng\nSHA-256", "Feature store\n(point-in-time)"]):
        box(ax, 0.35, 5.3 - i * 1.15, 2.1, 0.85, t, "#DDE6ED", tc="black", fs=8.5)
    ax.add_patch(FancyBboxPatch((3.0, 0.3), 5.6, 6.3, boxstyle="round,pad=0.02", fc="#EEF5F0", ec="#999"))
    ax.text(5.8, 6.35, "TẦNG PHÂN TÍCH & QUYẾT ĐỊNH", ha="center", fontweight="bold")
    box(ax, 3.2, 5.0, 2.5, 1.0, "Dự báo LightGBM\n+ hiệu chỉnh Isotonic/Platt\n+ CQR chi tiêu", PALETTE[0], fs=8)
    box(ax, 5.9, 5.0, 2.5, 1.0, "Nhân quả: AIPW, DML\nS/T/X/R/DR-learner\nQini, AUUC", PALETTE[1], fs=8)
    box(ax, 3.2, 3.3, 5.2, 1.1, "Hàm giá trị  V = τ·m·S − c_contact − p1·c_incentive\nPredict-Then-Optimize  |  SPO+ (Adam, regret)", PALETTE[3], fs=8.5)
    box(ax, 3.2, 1.6, 5.2, 1.1, "MILP cái túi đa chiều (OR-Tools SCIP)\nnăng lực · ngân sách · trần/sàn nhóm tuổi", PALETTE[2], fs=8.5)
    box(ax, 3.2, 0.5, 5.2, 0.8, "Action Router: Guardrails → HITL → Idempotency → Nhật ký", PALETTE[6], fs=8.5)
    ax.add_patch(FancyBboxPatch((8.9, 0.3), 4.0, 6.3, boxstyle="round,pad=0.02", fc="#FBF3EC", ec="#999"))
    ax.text(10.9, 6.35, "TẦNG VẬN HÀNH (MLOps)", ha="center", fontweight="bold")
    for i, t in enumerate(["Đóng băng mô hình\n+ checksum", "Locked Test / OOT\nRelease Gates", "Độ trễ P50/P95/P99", "Giám sát PSI, ECE,\nAction TVD, độ trễ nhãn", "Current vs Challenger\n→ tái huấn luyện"]):
        box(ax, 9.15, 5.3 - i * 1.15, 3.5, 0.85, t, "#F6D9C4", tc="black", fs=8.5)
    arrow(ax, (2.6, 3.5), (3.15, 3.8)); arrow(ax, (2.6, 4.5), (3.15, 5.3))
    arrow(ax, (4.45, 5.0), (4.45, 4.45)); arrow(ax, (7.15, 5.0), (7.15, 4.45))
    arrow(ax, (5.8, 3.3), (5.8, 2.75)); arrow(ax, (5.8, 1.6), (5.8, 1.35))
    arrow(ax, (8.45, 0.9), (9.1, 1.0)); arrow(ax, (8.45, 2.1), (9.1, 3.0))
    arrow(ax, (10.9, 0.65), (1.4, 0.15), col=PALETTE[6], ls="--", style="-|>")
    ax.text(6.5, 0.02, "vòng phản hồi: kết quả thực → nhãn mới → tái huấn luyện", ha="center", fontsize=8, color=PALETTE[6])
    ax.set_title("Kiến trúc hệ thống phân bổ ngân sách khuyến mãi theo uplift", fontsize=12)
    save_fig(fig, cfg, "f00b_system_architecture", "Kiến trúc tổng thể ba tầng của hệ thống")

    # --------- Graphic Abstract
    fig, ax = plt.subplots(figsize=(14, 4.2)); ax.set_xlim(0, 14); ax.set_ylim(0, 4.2); ax.axis("off")
    steps = [("DỮ LIỆU", "X5 RetailHero\nRCT gửi SMS\n+ lịch sử giao dịch", "#6C757D"),
             ("DỰ BÁO", "P(mua), chi tiêu S\nhiệu chỉnh + CQR", PALETTE[0]),
             ("NHÂN QUẢ", "ATE / CATE\nuplift τ(x)", PALETTE[1]),
             ("GIÁ TRỊ", "V = τ·m·S − chi phí\nSPO+ ↓ regret", PALETTE[3]),
             ("TỐI ƯU", "MILP: ngân sách,\nnăng lực, công bằng", PALETTE[2]),
             ("HÀNH ĐỘNG", "Router + guardrails\n+ người duyệt", PALETTE[6]),
             ("GIÁM SÁT", "Gates, PSI, P95,\nChallenger", "#1B998B")]
    w = 1.75
    for i, (h, t, c) in enumerate(steps):
        x = 0.2 + i * 1.98
        box(ax, x, 1.2, w, 1.9, "", c)
        ax.text(x + w / 2, 2.75, h, ha="center", color="white", fontweight="bold", fontsize=10)
        ax.text(x + w / 2, 1.85, t, ha="center", va="center", color="white", fontsize=8.5)
        if i < len(steps) - 1:
            arrow(ax, (x + w + 0.02, 2.15), (x + 1.98 - 0.02, 2.15), lw=2)
    arrow(ax, (13.0, 1.15), (1.0, 1.15), col="#1B998B", ls="--")
    ax.text(7, 0.75, "Vòng phản hồi: kết quả chiến dịch → nhãn mới → tái huấn luyện có kiểm soát", ha="center", fontsize=9, color="#1B998B")
    ax.text(7, 3.75, "Từ dự báo & nhân quả đến quyết định tối ưu và hành động có trách nhiệm", ha="center", fontsize=13, fontweight="bold")
    save_fig(fig, cfg, "f00c_graphic_abstract", "Graphic Abstract của nghiên cứu")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run(load_config())
