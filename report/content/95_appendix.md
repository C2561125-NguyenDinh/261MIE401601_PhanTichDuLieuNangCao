# PHỤ LỤC

## PL.1 Kho mã nguồn và hướng dẫn tái lập

Toàn bộ mã nguồn, cấu hình và bộ sinh báo cáo được lưu trong kho GitHub của học viên: https://github.com/C2561125-NguyenDinh/261MIE401601_PhanTichDuLieuNangCao. Kho được cấp quyền truy cập công khai để giảng viên có thể sao chép (clone) và chạy lại.

Các bước tái lập:

1. `git clone https://github.com/C2561125-NguyenDinh/261MIE401601_PhanTichDuLieuNangCao.git` và chuyển vào thư mục dự án.
2. Tạo môi trường ảo Python (phiên bản 3.10 trở lên) và chạy `pip install -r requirements.txt`.
3. Chạy `python run_all.py`. Pipeline tự tải ba tệp dữ liệu X5 từ kho công khai của scikit-uplift, kiểm chứng mã băm, chạy tám giai đoạn và sinh lại báo cáo này tại `report/output/`.
4. Đối chiếu mã băm dữ liệu gốc với [[tab:t01_raw_checksums]] và mã băm tạo tác với [[tab:t41_artifact_checksums]].
5. Khởi động dịch vụ chấm điểm: `uvicorn serve.api:app --port 8000`, rồi gọi `POST /score` với `{"client_id": "..."}`.
6. Mở dashboard giám sát: `streamlit run dashboard/app.py` (trang được chọn ở thanh bên).
7. Xem lần chạy và mô hình đã đăng ký: `mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db`; báo cáo trôi dạt Evidently nằm ở `mlops/evidently_drift_train_vs_oot.html`.

Thời gian chạy từng giai đoạn trên máy thực nghiệm được ghi ở bảng dưới đây.

{{tab:x_stage_timings|fs=10}}

## PL.2 Hợp đồng dữ liệu đầu vào – đầu ra

### PL.2.1 Từ điển đặc trưng

{{tab:t08_feature_dictionary|fs=8.5}}

### PL.2.2 Tệp dữ liệu trung gian và đầu ra

Các tệp trung gian được lưu trong `data/processed/`: `transactions.parquet` (giao dịch đã gộp), `campaign.parquet` (đặc trưng và nhãn của tập chiến dịch, kèm cột `split`), `forecast_snapshots.parquet` (ba snapshot dự báo), `forecast_campaign.parquet` (xác suất mua, chi tiêu dự báo và khoảng CQR), `cate_campaign.parquet` (CATE của sáu learner, độ bất đồng, mức cơ sở), `decision_scores.parquet` (điểm quyết định, chi phí, giả kết quả DR), `milp_solution_locked.csv` (nghiệm MILP) và `decision_log_*.csv` (nhật ký quyết định). Kết quả được lưu trong `results/`: `metrics.json` (toàn bộ chỉ số), `registry.json` (danh mục hình và bảng), `tables/*.csv` và `figures/*.png`.

## PL.3 Bảng mã băm kiểm chứng và ma trận tương thích phiên bản

{{tab:t41_artifact_checksums|fs=7.5}}

{{tab:t42_version_matrix|fs=9}}

## PL.4 Bảng kết quả chi tiết

### PL.4.1 Cân bằng hiệp biến đầy đủ

{{tab:t16_balance_smd|fs=8.5}}

### PL.4.2 PSI của toàn bộ đặc trưng

{{tab:t44_psi_drift|fs=7.5}}

### PL.4.3 Tầm quan trọng đặc trưng đầy đủ

{{tab:t13_forecast_importance|fs=9}}

{{tab:t23_cate_importance|fs=9}}

### PL.4.4 Độ ổn định nghiệm MILP (30 lần nhiễu hóa)

{{tab:t33_solution_stability|fs=9}}

### PL.4.5 Trích nhật ký quyết định

{{tab:t39_decision_log_sample|fs=7.5}}

### PL.4.6 Uplift theo thập phân vị của sáu learner (Locked Test)

{{tab:tx08_learner_deciles_locked|fs=8}}

### PL.4.7 Lợi nhuận DR theo tỷ lệ nhắm của từng learner

{{tab:tx10_policy_curves|fs=8}}

### PL.4.8 Độ nhạy của regret theo kích thước lô và tỷ lệ chọn

{{tab:tx12_spo_batch_sensitivity|fs=8}}

### PL.4.9 Đường cong quyết định đầy đủ

{{tab:t14_decision_curve|fs=9}}

### PL.4.10 Kiểm tra áp lực ràng buộc công bằng

{{tab:tx14_fairness_stress|fs=9}}

### PL.4.11 So sánh chính sách đầy đủ

{{tab:t28_policy_comparison_locked|fs=7.5}}

{{tab:t28_policy_comparison_oot|fs=7.5}}

### PL.4.12 Kết quả Predict-Then-Optimize và SPO+ đầy đủ

{{tab:t26_pto_vs_spo|fs=8}}
