## PL.5 Nhật ký quyết định thiết kế

Phụ lục này ghi lại các quyết định thiết kế quan trọng được đưa ra trong quá trình xây dựng hệ thống, kèm bối cảnh, phương án được chọn và lý do, theo dạng "bản ghi quyết định kiến trúc" (architecture decision record). Mục đích là để người đọc hiểu vì sao hệ thống có hình dạng hiện tại, và để các thay đổi sau này có điểm tham chiếu.

**QĐ-01. Chọn bộ dữ liệu X5 RetailHero.** *Bối cảnh:* đề tài cần dữ liệu thương mại điện tử/bán lẻ có biến can thiệp để nhận dạng nhân quả và có trục thời gian để thiết kế kiểm định theo thời gian. *Quyết định:* dùng X5 RetailHero. *Lý do:* là một trong số ít bộ dữ liệu công khai vừa có thí nghiệm ngẫu nhiên vừa có lịch sử giao dịch có dấu thời gian. *Hệ quả:* chỉ có một chiến dịch, nên tập OOT của bài toán uplift phải xây dựng theo cohort.

**QĐ-02. Dời các snapshot ra khỏi cửa sổ chọn mẫu.** *Bối cảnh:* phiên bản đầu với H = 28 ngày và snapshot cuối kết thúc tại thời điểm quyết định cho tỷ lệ mua 100% ở snapshot S3. *Quyết định:* xác định cửa sổ chọn mẫu từ dữ liệu, đặt S3 sao cho cửa sổ nhãn kết thúc tại đầu cửa sổ chọn mẫu, giảm H xuống 14 ngày để có đủ lịch sử trước S1. *Lý do:* loại bỏ rò rỉ do chọn mẫu. *Hệ quả:* kỳ dự báo ngắn hơn; dữ liệu lịch sử trước S1 chỉ khoảng 53 ngày.

**QĐ-03. Dùng cửa sổ nhìn lại cố định cho mọi đặc trưng.** *Bối cảnh:* các snapshot có độ dài lịch sử khác nhau do dữ liệu bắt đầu tại một ngày cố định. *Quyết định:* tính mọi đặc trưng trong cửa sổ L = 53 ngày trước mốc cắt, kể cả cho tập chiến dịch. *Lý do:* tránh trôi dạt giả của các đặc trưng tích lũy và bảo đảm mô hình dự báo áp dụng được lên tập chiến dịch. *Hệ quả:* tập chiến dịch không dùng phần lịch sử cũ hơn 53 ngày.

**QĐ-04. Chọn learner và phương pháp ra quyết định chỉ trên Validation.** *Bối cảnh:* có sáu learner và năm phương pháp ra quyết định. *Quyết định:* chọn theo Qini và regret trên Validation, không thay đổi sau khi xem Locked Test. *Lý do:* giữ tính không chệch của đánh giá cuối cùng. *Hệ quả:* learner và phương pháp được chọn không phải là tốt nhất trên Locked Test; điều này được báo cáo nguyên trạng và trở thành căn cứ cho kiến nghị.

**QĐ-05. Dùng giả kết quả kép vững làm nhãn chi phí cho SPO+.** *Bối cảnh:* tác động cá nhân không quan sát được. *Quyết định:* dùng ψ từ cross-fitting trên Train và từ T-learner ngoài mẫu trên các tập khác. *Lý do:* ψ không chệch cho CATE khi điểm xu hướng đã biết (mục 2.14.3). *Hệ quả:* regret theo nhãn thay thế nhiễu lớn, chỉ có ý nghĩa tương đối.

**QĐ-06. Dùng OR-Tools với SCIP thay cho PuLP.** *Bối cảnh:* phiên bản PuLP trong môi trường thực nghiệm không kèm bộ giải CBC. *Quyết định:* dùng OR-Tools (cài đặt qua pip, kèm sẵn SCIP, CBC, GLOP). *Lý do:* cài đặt đơn giản trên mọi hệ điều hành, phù hợp với yêu cầu giảng viên clone và chạy lại. *Hệ quả:* có thể lấy giá đối ngẫu trực tiếp từ GLOP.

**QĐ-07. Siết dung sai tối ưu của MILP.** *Bối cảnh:* với dung sai mặc định, giá trị mục tiêu của MILP không ràng buộc công bằng có lúc thấp hơn một lượng rất nhỏ so với MILP có ràng buộc. *Quyết định:* đặt dung sai tối ưu tương đối 10⁻⁷. *Lý do:* các so sánh "cái giá của công bằng" và Greedy – MILP đòi hỏi độ chính xác cao hơn mức chênh lệch cần đo. *Hệ quả:* thời gian giải tăng nhẹ nhưng vẫn vài giây.

**QĐ-08. Bootstrap ghép cặp cho so sánh chính sách.** *Bối cảnh:* phiên bản đầu dùng các mẫu bootstrap khác nhau cho từng chính sách, khiến hai chính sách giống hệt nhau có khoảng tin cậy khác nhau. *Quyết định:* dùng cùng một tập chỉ số bootstrap cho mọi chính sách và báo cáo thêm khoảng tin cậy của chênh lệch MILP − chính sách. *Lý do:* so sánh ghép cặp chính xác hơn và nhất quán. *Hệ quả:* các so sánh trên cùng tập dữ liệu nhất quán với nhau.

**QĐ-09. Tách phân tích mở rộng thành giai đoạn riêng.** *Bối cảnh:* cần nhiều phân tích độ vững nhưng không được làm thay đổi hệ thống đã khóa. *Quyết định:* đặt mọi phân tích bổ trợ trong mô-đun `s9_extended.py`, chỉ đọc kết quả của các giai đoạn 1–8. *Lý do:* bảo đảm tính toàn vẹn của quy trình lựa chọn – đánh giá. *Hệ quả:* các cải tiến được đề xuất (tổ hợp learner, rào chắn đồng thuận, tối ưu bảo thủ) được trình bày như kiến nghị cho chu kỳ sau, không thay thế kết quả chính.

**QĐ-10. Sinh báo cáo tự động từ kết quả.** *Bối cảnh:* yêu cầu mọi số liệu trong báo cáo phải lấy từ kết quả thực nghiệm. *Quyết định:* viết nội dung văn bản với các thẻ tham chiếu tới `metrics.json` và bảng kết quả; bộ sinh báo cáo thay thế và đánh dấu mọi tham chiếu không tìm thấy. *Lý do:* loại bỏ khả năng sai lệch giữa kết quả và văn bản, và cho phép cập nhật báo cáo bằng một lệnh. *Hệ quả:* thay đổi dữ liệu hoặc giả định sẽ tự động cập nhật mọi con số; phần diễn giải định tính vẫn cần được rà soát lại nếu kết quả thay đổi về chiều hướng.

## PL.6 Hướng dẫn vận hành dành cho người dùng nghiệp vụ

Phụ lục này mô tả cách bộ phận CRM sử dụng đầu ra của hệ thống trong một chu kỳ chiến dịch, không đòi hỏi kiến thức lập trình.

**Trước chiến dịch.** Bộ phận CRM xác định ngân sách, năng lực gửi và các cam kết công bằng, rồi chuyển cho nhóm dữ liệu cập nhật vào tệp cấu hình. Nhóm dữ liệu chạy pipeline và gửi lại ba đầu ra: (i) danh sách khách hàng theo luồng hành động (tệp nhật ký quyết định); (ii) bảng cổng phát hành: nếu có cổng không đạt, chiến dịch chỉ được chạy ở chế độ có đối chứng; (iii) báo cáo tóm tắt gồm lợi nhuận tăng thêm ước lượng *kèm khoảng tin cậy* và phân bổ theo nhóm tuổi.

**Xử lý hàng chờ duyệt.** Với mỗi khách hàng trong luồng HUMAN_REVIEW, người duyệt xem mã lý do (VIP hay bất định cao), thông tin khách hàng và điểm quyết định, rồi chọn chấp thuận, từ chối hoặc chuyển sang chương trình chăm sóc riêng. Quyết định của người duyệt được ghi lại kèm khóa lũy đẳng để phục vụ đánh giá sau chiến dịch. Thời hạn duyệt cần kết thúc trước giờ gửi tin; các trường hợp chưa duyệt được xử lý như DEFER.

**Trong chiến dịch.** Hệ thống gửi tin chỉ thực thi các bản ghi AUTO_SEND và các bản ghi HUMAN_REVIEW đã được chấp thuận; mọi yêu cầu gửi mang khóa lũy đẳng nên việc thử lại khi lỗi mạng không tạo tin trùng. Nhóm đối chứng ngẫu nhiên không bao giờ nhận tin trong chiến dịch.

**Sau chiến dịch.** Sau khi hết cửa sổ đo kết quả, nhóm dữ liệu tính lợi nhuận tăng thêm thực tế bằng so sánh với nhóm đối chứng, cập nhật nhãn và chạy quy trình Current – Challenger. Bộ phận CRM nhận báo cáo so sánh giữa lợi nhuận dự kiến (kèm khoảng tin cậy) và lợi nhuận thực tế, cùng các phát hiện mới về phân khúc.

**Khi có cảnh báo.** Người trực tra playbook ngoại lệ theo mã cảnh báo và thực hiện các bước khoanh vùng trong thời hạn SLA. Mọi thay đổi chế độ vận hành (ví dụ chuyển toàn bộ sang duyệt thủ công) phải được ghi nhận.
