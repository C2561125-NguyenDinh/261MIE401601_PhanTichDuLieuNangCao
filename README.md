# Hệ thống phân bổ ngân sách khuyến mãi giữ chân khách hàng theo uplift (X5 RetailHero)

Mã nguồn tiểu luận cuối kỳ môn **Phân tích dữ liệu nâng cao** (ThS, Trường ĐH Kinh tế – Luật, ĐHQG-HCM).

Đây là một pipeline Python **xuyên suốt và tái lập được** theo chu trình 8 giai đoạn: từ dự báo và ước lượng nhân quả đến tối ưu phân bổ (MILP) và điều phối hành động, có giám sát sau triển khai. Báo cáo Word được **sinh tự động từ kết quả chạy code**, nên mọi con số trong báo cáo đều khớp với kết quả thực nghiệm.

```
Dữ liệu X5 (RCT gửi SMS) → Dự báo + hiệu chỉnh + CQR → Nhân quả (ATE/CATE, 6 meta-learner)
→ Hàm giá trị + SPO+ → MILP (OR-Tools SCIP) → Action Router (guardrails, HITL, idempotency)
→ MLOps (checksum, P95, PSI, TVD, Challenger, Release Gates) → báo cáo .docx
```

## 1. Cài đặt

Yêu cầu Python ≥ 3.10, khoảng 8 GB RAM và ~3 GB dung lượng trống (dữ liệu gốc khoảng 700 MB). Toàn bộ pipeline chạy khoảng 25–40 phút trên máy tính cá nhân.

```bash
git clone https://github.com/C2561125-NguyenDinh/261MIE401601_PhanTichDuLieuNangCao.git
cd 261MIE401601_PhanTichDuLieuNangCao
python -m venv .venv
# Windows: .venv\Scripts\activate    |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Chạy toàn bộ

```bash
python run_all.py
```

Lệnh này sẽ:
1. tự tải 3 tệp dữ liệu X5 (`uplift_train`, `clients`, `purchases`) từ kho công khai của scikit-uplift vào `data/raw/` và kiểm tra mã băm SHA-256;
2. chạy lần lượt 8 giai đoạn (`src/s0_design.py` … `src/s8_mlops.py`) và các mô-đun bổ trợ s9–s12;
3. ghi hình vào `results/figures/`, bảng vào `results/tables/`, chỉ số vào `results/metrics.json`;
4. sinh báo cáo `report/output/Tieu_luan_PTDLNC_X5_Uplift.docx`.

Mở báo cáo trong Word và nhấn **Ctrl+A rồi F9** để cập nhật mục lục, danh mục hình và danh mục bảng.

Các lệnh khác:

```bash
python run_all.py --from 4     # chạy lại từ giai đoạn 4
python run_all.py --only 6     # chỉ chạy giai đoạn 6 (MILP)
python run_all.py --report     # chỉ sinh lại báo cáo từ kết quả có sẵn
```

Dịch vụ, dashboard và MLflow (sau khi đã chạy pipeline):

```bash
uvicorn serve.api:app --port 8000          # dịch vụ chấm điểm
streamlit run dashboard/app.py             # dashboard giám sát
mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db
```

Nếu máy không tải được dữ liệu tự động, hãy tải thủ công 3 tệp sau rồi đặt vào `data/raw/`:

- https://sklift.s3.eu-west-2.amazonaws.com/uplift_train.csv.gz
- https://sklift.s3.eu-west-2.amazonaws.com/clients.csv.gz
- https://sklift.s3.eu-west-2.amazonaws.com/purchases.csv.gz

## 3. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `config.yaml` | Toàn bộ tham số và giả định kinh tế (biên lợi nhuận, chi phí, ngân sách, ngưỡng quy tắc, cổng phát hành) |
| `run_all.py` | Điều phối pipeline |
| `src/s0_design.py` | Giai đoạn 1: Decision Contract, sơ đồ kiến trúc, Graphic Abstract |
| `src/s1_data.py` | Giai đoạn 2a: tải dữ liệu, SHA-256, gộp giao dịch, hồ sơ dữ liệu |
| `src/s2_features.py` | Giai đoạn 2b: đặc trưng point-in-time, 3 snapshot, Train/Val/Locked/OOT, DAG |
| `src/s3_forecast.py` | Giai đoạn 3: LightGBM, hiệu chỉnh Platt/Isotonic, ECE, CQR, net benefit |
| `src/s4_causal.py` | Giai đoạn 4: SMD, ATE/ATT/ATC (Neyman, AIPW, DoubleML), refutation, S/T/X/R/DR-learner, Qini/AUUC |
| `src/s5_spo.py` | Giai đoạn 5: hàm giá trị, Predict-Then-Optimize, SPO+ (Adam), regret |
| `src/s6_milp.py` | Giai đoạn 6: MILP cái túi đa chiều, Greedy, LP relaxation, shadow price, độ nhạy, độ ổn định |
| `src/s7_router.py` | Giai đoạn 7: Action Router, guardrails, Human-in-the-loop, idempotency, playbook |
| `src/s8_mlops.py` | Giai đoạn 8: checksum mô hình, độ trễ P50/P95/P99, PSI, TVD, độ trễ nhãn, Challenger, Release Gates |
| `src/s9_extended.py` | Phân tích mở rộng & kiểm tra độ vững (không thay đổi lựa chọn của giai đoạn 1–8) |
| `src/s10_evidence.py` | Lực thống kê, mô phỏng SPO+ so với PTO và MILP so với Greedy |
| `src/s11_causal_xai.py` | CausalForestDML (EconML), giải thích SHAP cho CATE, cây chính sách (PolicyTree) |
| `src/s12_mlops_tools.py` | MLflow Tracking/Registry, báo cáo trôi dạt Evidently, kiểm thử dịch vụ FastAPI |
| `serve/api.py` | Dịch vụ chấm điểm FastAPI (`/health`, `/score`, `/score_batch`) |
| `dashboard/app.py` | Dashboard giám sát Streamlit (5 trang) |
| `mlops/` | `mlflow.db` (SQLite) và báo cáo `evidently_drift_train_vs_oot.html` |
| `report/build_report.py` | Bộ sinh báo cáo .docx từ `report/content/*.md` + kết quả |
| `report/update_toc_libreoffice.py` | (Tùy chọn) cập nhật mục lục và xuất PDF bằng LibreOffice |
| `tests/make_synthetic.py` | Dữ liệu tổng hợp để kiểm thử nhanh (KHÔNG dùng cho báo cáo) |

## 4. Kiểm thử nhanh (không cần dữ liệu thật)

```bash
python tests/make_synthetic.py 8000
# Windows PowerShell:  $env:X5_SMOKE="1"; python run_all.py
X5_SMOKE=1 python run_all.py
```

Chế độ smoke ghi kết quả vào `results_smoke/`, tách biệt hoàn toàn với kết quả thật.

## 5. Ghi chú quan trọng về dữ liệu

Tập khách hàng của chiến dịch X5 được chọn có điều kiện trên hoạt động gần đây: **mọi** khách hàng đều có giao dịch trong ~22 ngày cuối. Vì vậy các snapshot dự báo được đặt *trước* cửa sổ chọn mẫu này (xem `src/s2_features.py` và mục 3.4.1 của báo cáo); nếu không, nhãn dự báo sẽ suy biến (100% "mua").

## 6. Tính tái lập

- Hạt giống ngẫu nhiên cố định (`seed: 42`).
- Mã băm SHA-256 của dữ liệu gốc, mô hình và dữ liệu đã xử lý: `results/tables/t01_raw_checksums.csv`, `results/tables/t41_artifact_checksums.csv`.
- Phiên bản thư viện đã kiểm thử: `results/tables/t42_version_matrix.csv`.
- Thời gian chạy từng giai đoạn: `results/stage_timings.json`.

## 7. Nguồn dữ liệu

X5 Retail Group (2019). *X5 RetailHero Uplift Modeling Competition*. Open Data Science. Bản phân phối lại công khai qua thư viện scikit-uplift.

## 8. Tuyên bố sử dụng AI

Mã nguồn và bản thảo báo cáo được xây dựng với sự hỗ trợ của trợ lý AI (Claude). Mọi kết quả định lượng được sinh ra bằng cách chạy mã trên dữ liệu thật; tác giả chịu trách nhiệm kiểm chứng nội dung, trích dẫn và các quyết định phân tích.
