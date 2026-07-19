Bank Marketing Project - AI
Dự án phân tích và xây dựng mô hình Machine Learning nhằm dự đoán khả năng khách hàng đăng ký gửi tiết kiệm kỳ hạn (term deposit), dựa trên bộ dữ liệu Bank Marketing (UCI Machine Learning Repository) — dữ liệu từ các chiến dịch tiếp thị qua điện thoại của một ngân hàng tại Bồ Đào Nha.

Bối cảnh
Tiếp thị qua điện thoại (telemarketing) là một trong những phương thức chính để ngân hàng tư vấn khách hàng sử dụng dịch vụ tiền gửi có kỳ hạn. Tuy nhiên gọi điện diện rộng tốn nhiều chi phí nhân sự và có thể gây ấn tượng xấu với khách hàng không có nhu cầu. Dự án ứng dụng Machine Learning để dự đoán trước xác suất khách hàng đồng ý tham gia, giúp tối ưu nhân lực và nâng cao tỷ lệ chuyển đổi.

Phát biểu bài toán
Loại bài toán: Phân loại nhị phân (Binary Classification)
Đầu vào: 20 đặc trưng, chia 3 nhóm — thông tin nhân khẩu học/tài chính khách hàng, thông tin chiến dịch liên hệ, và các chỉ số kinh tế vĩ mô. Dữ liệu pha trộn giữa numeric và categorical.
Đầu ra: biến y nhị phân (yes/no) — khách hàng có đăng ký gửi tiết kiệm kỳ hạn hay không.
Mục tiêu tối ưu: tối thiểu hoá Log-Loss, tối đa hoá hiệu suất — đặc biệt chú trọng Recall và F1-score trên lớp thiểu số ("yes"), do dữ liệu mất cân bằng.
Mục tiêu và phạm vi
Mục tiêu:

Tiền xử lý dữ liệu và Feature Engineering cho hơn 41.000 mẫu.
Xử lý mất cân bằng lớp (tỷ lệ khoảng 88:12) bằng kỹ thuật SMOTE.
Huấn luyện, tinh chỉnh siêu tham số và đánh giá ít nhất 2 mô hình: Logistic Regression (baseline) và Random Forest/XGBoost (ensemble).
Đóng gói mô hình hiệu suất cao nhất thành model artifact.
Triển khai thành hệ thống hoàn chỉnh: REST API + giao diện web.
Phạm vi: Chỉ giới hạn phân tích và dự đoán trên tập dữ liệu lịch sử tĩnh (Bank Marketing Dataset). Không bao gồm tích hợp tổng đài gọi điện tự động thời gian thực (auto-dialer) hay thu thập thêm dữ liệu từ nguồn ngoài.

Bộ dữ liệu
Nguồn: UCI Machine Learning Repository - Bank Marketing Dataset
Số lượng bản ghi: hơn 41.000 mẫu, 20 đặc trưng đầu vào
Biến mục tiêu: y — khách hàng có đăng ký gửi tiết kiệm kỳ hạn hay không (yes/no)
Tỷ lệ mất cân bằng lớp: khoảng 88:12 (no:yes) — xử lý bằng SMOTE
Các nhóm đặc trưng chính: thông tin nhân khẩu học/tài chính, lịch sử liên hệ chiến dịch, chỉ số kinh tế vĩ mô

Cấu trúc thư mục
├── api/            # Backend API phục vụ mô hình
├── app/            # Ứng dụng/service liên quan
├── data/           # Dữ liệu (raw/processed)
├── frontend/        # Giao diện người dùng 
├── models/         # Mô hình đã huấn luyện + metrics đi kèm
├── notebooks/      # Notebook phân tích & thử nghiệm mô hình
├── reports/        # Báo cáo EDA, đánh giá mô hình
├── src/            # Source code dùng chung (tiền xử lý, pipeline, utils)
├── tests/          # Unit test
├── 1_EDA.ipynb     # Notebook phân tích khám phá dữ liệu
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

Các mô hình đã thử nghiệm
So sánh giữa mô hình tuyến tính cơ sở (baseline) và các mô hình ensemble nâng cao, sau khi xử lý mất cân bằng dữ liệu bằng SMOTE:

Model	Vai trò	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	Baseline	ư			
Random Forest	Ensemble				
XGBoost	Ensemble	
Điền số liệu thật từ reports/ hoặc notebook huấn luyện. Vì bài toán ưu tiên phát hiện đúng nhóm "yes" (lớp thiểu số), nên đánh giá mô hình tốt nhất nên dựa trên Recall và F1-score trên lớp "yes", không chỉ dựa vào Accuracy tổng thể (dễ gây hiểu nhầm với dữ liệu mất cân bằng).

Cài đặt & chạy thử
1. Clone repo
bash
git clone https://github.com/oanhnguyen0529/bank-marketing-project-AI.git
cd bank-marketing-project-AI
2. Cài đặt môi trường
bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Chạy notebook phân tích
bash
jupyter notebook 1_EDA.ipynb
4. Chạy bằng Docker (khi API sẵn sàng)
bash
docker-compose up --build
Công nghệ sử dụng
Python, pandas, scikit-learn, XGBoost/LightGBM
Jupyter Notebook cho EDA & thử nghiệm mô hình
Docker & Docker Compose để đóng gói triển khai
(Cập nhật thêm: framework API bạn dùng — FastAPI/Flask, framework frontend — React/Streamlit...)
Roadmap
 EDA và tiền xử lý dữ liệu
 Huấn luyện & so sánh nhiều mô hình
 Hoàn thiện API phục vụ mô hình
 Hoàn thiện frontend demo
 Deploy lên môi trường thực (Docker/Cloud)
 Viết unit test đầy đủ cho src/
