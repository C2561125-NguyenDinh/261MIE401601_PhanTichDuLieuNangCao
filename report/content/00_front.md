{{cover|title=Hệ thống phân bổ ngân sách khuyến mãi giữ chân khách hàng theo tác động biên: từ dự báo và suy luận nhân quả đến tối ưu hóa MILP và điều phối hành động|course=Phân tích dữ liệu nâng cao|gvhd=TS. Nguyễn Thôn Dã|hvth=Đinh Nguyễn Tấn Nguyên – C25611252|date=TP.HCM, THÁNG 11/2026}}

# LỜI CẢM ƠN

Em xin gửi lời cảm ơn đến TS. Nguyễn Thôn Dã, giảng viên môn Phân tích dữ liệu nâng cao. Chu trình tám giai đoạn trong bài giảng, từ thiết kế bài toán hướng đến quyết định, thiết kế dữ liệu theo thời gian, dự báo có hiệu chỉnh, phân tích nhân quả và uplift, Predict-Then-Optimize và SPO+, tối ưu hóa MILP, định tuyến hành động đến triển khai và giám sát, là khung sườn để em thiết kế hệ thống và lựa chọn phương pháp cho tiểu luận này.

Em cũng cảm ơn X5 Retail Group đã công bố dữ liệu thí nghiệm ngẫu nhiên của chiến dịch SMS trong cuộc thi X5 RetailHero, nhóm phát triển scikit-uplift đã phân phối lại bộ dữ liệu, cùng cộng đồng phát triển pandas, scikit-learn, LightGBM, DoubleML, OR-Tools và PyArrow. Nhờ các công cụ mã nguồn mở này, toàn bộ pipeline với hơn tám triệu giao dịch có thể chạy trên một máy tính cá nhân.

Do giới hạn về thời gian và tài nguyên tính toán, tiểu luận chắc chắn còn thiếu sót. Em mong nhận được góp ý của thầy để hoàn thiện nghiên cứu.

# LỜI CAM KẾT

Em xin cam đoan tiểu luận này là công trình nghiên cứu do em thực hiện dưới sự hướng dẫn của TS. Nguyễn Thôn Dã trong khuôn khổ môn học Phân tích dữ liệu nâng cao. Toàn bộ dữ liệu được tải trực tiếp từ bộ dữ liệu công khai X5 RetailHero. Mọi con số, bảng biểu và hình ảnh trong báo cáo đều được tạo ra tự động từ các tệp kết quả của mã nguồn đính kèm, không có số liệu nào được nhập tay hay ước đoán.

Mã nguồn, tệp cấu hình, dữ liệu trung gian và các tệp kết quả được lưu đầy đủ trong thư mục đồ án để người chấm có thể chạy lại và đối chiếu. Các tài liệu tham khảo được trích dẫn rõ nguồn theo chuẩn APA phiên bản 7.

>|*Thành phố Hồ Chí Minh, năm 2026*
>|**Học viên thực hiện**
>|Đinh Nguyễn Tấn Nguyên

# TUYÊN BỐ SỬ DỤNG TRÍ TUỆ NHÂN TẠO

Theo yêu cầu của học phần, học viên tuyên bố việc sử dụng công cụ trí tuệ nhân tạo (AI) trong quá trình thực hiện tiểu luận như sau:

- **Công cụ sử dụng:** Claude (Anthropic), sử dụng qua ứng dụng Claude.
- **Mục đích sử dụng:** (i) hỗ trợ thiết kế kiến trúc và viết mã nguồn Python cho pipeline tám giai đoạn; (ii) hỗ trợ xây dựng bộ sinh báo cáo tự động; (iii) hỗ trợ soạn thảo bản nháp các phần lý thuyết, phương pháp và diễn giải kết quả; (iv) hỗ trợ tra cứu và kiểm tra thông tin thư mục của tài liệu tham khảo.
- **Cách kiểm chứng:** mọi kết quả định lượng được tạo ra bằng cách chạy mã nguồn trên dữ liệu thật và được đọc tự động vào báo cáo từ tệp kết quả, nên không phụ thuộc vào nội dung do AI sinh ra. Học viên đã rà soát mã nguồn, chạy lại pipeline, đối chiếu diễn giải với bảng và hình kết quả, và kiểm tra thông tin thư mục của các tài liệu tham khảo.
- **Trách nhiệm:** học viên chịu trách nhiệm cuối cùng về mọi quyết định phân tích, diễn giải và kết luận trong tiểu luận.

# TÓM TẮT

Tiểu luận xây dựng và kiểm định thực chứng một hệ thống phân tích khép kín theo chu trình tám giai đoạn nhằm hỗ trợ quyết định phân bổ ngân sách khuyến mãi giữ chân khách hàng trong bán lẻ và thương mại điện tử. Dữ liệu nghiên cứu là bộ X5 RetailHero gồm {{m:data_profile.n_train_clients|int}} khách hàng tham gia một chiến dịch SMS được phân bổ ngẫu nhiên và {{m:data_raw_rows.transactions|int}} giao dịch lịch sử. Hệ thống thiết kế dữ liệu đúng tại thời điểm quyết định với bốn tập Train, Validation, Locked Test và Out-of-Time, đồng thời phát hiện và xử lý một dạng rò rỉ do chọn mẫu trong dữ liệu. Mô hình LightGBM có hiệu chỉnh Platt dự báo xác suất mua với AUC {{tv:t12_calibration_results|Hiệu chỉnh=platt;Tập=Locked Test (S2)|AUC|.4f}} và ECE {{tv:t12_calibration_results|Hiệu chỉnh=platt;Tập=Locked Test (S2)|ECE|.4f}} trên tập khóa; hồi quy phân vị phù hợp (CQR) cho khoảng chi tiêu đạt độ phủ {{tv:t15_cqr_results|Tập=Out-of-Time (S3);Phương pháp=CQR|Độ phủ|pct1}} trên dữ liệu ngoài thời gian. Tác động trung bình của SMS lên xác suất mua là {{m:data_profile.naive_ate|.4f}}, nhất quán giữa ước lượng Neyman, AIPW và Double ML, và vững trước các kiểm định phản bác; tác động lớn nhất ở khách hàng lớn tuổi, ít mua và gần bằng 0 ở khách hàng mua thường xuyên. Sáu meta-learner ước lượng tác động không đồng nhất; điểm quyết định được xây dựng từ hàm giá trị có chi phí phụ thuộc phản hồi và học bằng Predict-Then-Optimize và SPO+. Bài toán phân bổ được giải bằng MILP (SCIP) dưới ràng buộc năng lực, ngân sách và công bằng theo nhóm tuổi trong vài giây; Action Router tích hợp rào chắn nghiệp vụ, con người tham gia và khóa lũy đẳng. Kết quả cho thấy lợi nhuận tăng thêm của chính sách tối ưu dương có ý nghĩa trên cohort ngoài thời gian ({{tv:t28_policy_comparison_oot|Chính sách=MILP (SCIP)|Lợi nhuận tăng thêm DR (RUB)|,.0f}} RUB) nhưng chưa có ý nghĩa trên tập khóa; hệ thống đạt {{m:mlops.gates_pass|int}}/{{m:mlops.gates_total|int}} cổng phát hành và được khuyến nghị triển khai dạng thí nghiệm có nhóm đối chứng. Phân tích lực thống kê cho thấy tập khóa chỉ đủ phát hiện chênh lệch từ khoảng {{tv:tx32_power_analysis|Tập=Locked Test;So sánh=MILP so với không gửi|MDE lực 80% (RUB)|,.0f}} RUB trở lên, nên kết quả chưa có ý nghĩa phản ánh cỡ mẫu chứ không phản ánh việc hệ thống không tạo giá trị. Mô phỏng tổng hợp có đáp án biết trước cho thấy SPO+ giảm mạnh regret khi mô hình sai đặc tả và nhãn chi phí ít nhiễu nhưng mất lợi thế khi nhãn nhiễu lớn như trên X5, còn MILP vượt thuật toán tham lam khoảng {{tv:tx34_milp_simulation|Kịch bản=S2. Ngân sách và năng lực cùng chặt|Khoảng cách Greedy TB (%)|.0f}}% khi năng lực gửi cũng là ràng buộc chặt. Đối chứng bằng CausalForestDML của EconML cho khả năng xếp hạng uplift tương đương, giải thích SHAP chỉ ra tuổi và chi tiêu gần đây là các đặc trưng dẫn dắt tác động, và cây chính sách học từ điểm thưởng kép vững đạt giá trị tương đương MILP khi cùng ngân sách. Tầng triển khai được hiện thực bằng MLflow, Evidently, dịch vụ FastAPI và dashboard giám sát Streamlit. Nghiên cứu cũng minh họa quan hệ phi tuyến giữa sai số dự báo và tổn thất quyết định, lời nguyền của người tối ưu, và rủi ro của rào chắn nghiệp vụ dựa trên đầu ra mô hình.

**Từ khóa:** uplift modeling, suy luận nhân quả, Double Machine Learning, Predict-Then-Optimize, SPO+, MILP, Human-in-the-Loop, MLOps, giữ chân khách hàng.

# MỤC LỤC

{{toc}}

# DANH MỤC HÌNH

{{lof}}

# DANH MỤC BẢNG

{{lot}}

# DANH MỤC TỪ VIẾT TẮT

{{tab:s_abbrev|fs=10}}

# KẾ HOẠCH THỰC HIỆN

Kế hoạch thực hiện đồ án được xây dựng bám theo lịch các buổi học của học phần, mỗi bước gắn với một chương lý thuyết và tạo ra các sản phẩm có thể kiểm chứng trong kho mã nguồn.

{{tab:s_plan|fs=9}}

