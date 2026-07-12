# Data Dictionary — `bank_final.csv`

**Nguồn gốc**: UCI Bank Marketing Dataset (Moro, Cortez, Rita, 2014)

**Pipeline sinh ra file này**: `raw/bank-additional-full.csv` → `clean_data()` (src/preprocess.py) → bucket `age`/`duration` + ordinal `poutcome` (03_FeatureEngineering.ipynb) → one-hot phần categorical còn lại (`encode_onehot()`) → xuất `bank_final.csv`

**Kích thước**: 41,172 dòng × 53 cột

**Chưa scale**: các cột numeric liệt kê ở mục 3 bên dưới **chưa được chuẩn hóa** (z-score/min-max). Việc scale phải làm **sau khi `train_test_split`**, fit `scaler` trên tập train rồi lưu `scaler.pkl` — không fit trên toàn bộ `bank_final.csv`.

---

## 1. Biến mục tiêu (target)

Cột : `y` -> subscribed  
Kiểu : int (0/1) : 0 = no, 1 = yes  
Giá trị : Khách hàng có đăng ký gói tiền gửi có kỳ hạn hay không.  
Ý nghĩa : **Mất cân bằng: ~88.7% = 0, ~11.3% = 1** → cần `class_weight` hoặc SMOTE khi train.  

---

## 2. Biến đã được feature-engineer (bucket / ordinal)

| Cột | Kiểu | Giá trị | Ý nghĩa |
|---|---|---|---|
| `age` | int, 1–4 | 1 (≤32 tuổi)  <br>2 [33-47] <br>3 [48-70]<br>4 [71-98]) | Bucket theo tứ phân vị (IQR) của tuổi khách hàng |
| `duration` | float, 1–5 | 1 (≤102s) <br>2 [103–180s] <br>3 [181–319s] <br>4 [320–644.5s] <br>5 [644.5s] | Bucket theo tứ phân vị của thời lượng cuộc gọi (giây).  <br>**Lưu ý rủi ro**: giá trị này chỉ biết được *sau khi* cuộc gọi kết thúc — nếu bài toán là dự đoán *trước khi gọi*, cần cân nhắc loại bỏ khi train mô hình dùng để dự đoán. |
| `poutcome` | int, 1–3 | 1: nonexistent (chưa từng liên hệ)<br>2: failure   <br>3: success | Kết quả chiến dịch trước đó — encode dạng ordinal vì có thứ tự ý nghĩa tăng dần |

---

## 3. Biến numeric gốc (cần scale trước khi train)

| Cột | Kiểu | Ý nghĩa | Ghi chú |
|---|---|---|---|
| `campaign` | float | Số lần liên hệ trong chiến dịch hiện tại (kể cả lần này) | |
| `pdays` | float | Số ngày kể từ lần liên hệ trước đó | **999 = sentinel**, nghĩa là "chưa từng được liên hệ trước đây", không phải số ngày thật. |
| `previous` | float | Số lần liên hệ trước chiến dịch hiện tại | |
| `emp_var_rate` | float | Employment variation rate (chỉ số kinh tế vĩ mô, theo quý) | |
| `cons_price_idx` | float | Consumer price index (chỉ số giá tiêu dùng, theo tháng) | |
| `cons_conf_idx` | float | Consumer confidence index (chỉ số niềm tin tiêu dùng, theo tháng) | |
| `euribor3m` | float | Lãi suất Euribor kỳ hạn 3 tháng (theo ngày) | |
| `nr_employed` | float | Số lượng nhân viên (chỉ số kinh tế vĩ mô, theo quý) | |

---

## 4. Biến one-hot encoding (drop_first=True)

Mỗi nhóm dưới đây được sinh từ 1 cột categorical gốc bằng `pd.get_dummies(..., drop_first=True)`. **Category bị drop** (không xuất hiện tên cột) chính là baseline/reference category — khi tất cả cột dummy trong nhóm = 0, nghĩa là khách hàng thuộc category đó.

| Cột gốc | Category bị drop (baseline) | Các cột dummy tương ứng |
|---|---|---|
| `job` | `admin.` | `job_blue-collar`, `job_entrepreneur`, `job_housemaid`, `job_management`, `job_retired`, `job_self-employed`, `job_services`, `job_student`, `job_technician`, `job_unemployed`, `job_unknown` |
| `marital` | `divorced` | `marital_married`, `marital_single`, `marital_unknown` |
| `education` | `basic.4y` | `education_basic.6y`, `education_basic.9y`, `education_high.school`, `education_illiterate`, `education_professional.course`, `education_university.degree`, `education_unknown` |
| `default` | `no` | `default_unknown`, `default_yes` |
| `housing` | `no` | `housing_unknown`, `housing_yes` |
| `loan` | `no` | `loan_unknown`, `loan_yes` |
| `contact` | `cellular` | `contact_telephone` |
| `month` | `apr` | `month_aug`, `month_dec`, `month_jul`, `month_jun`, `month_mar`, `month_may`, `month_nov`, `month_oct`, `month_sep` |
| `day_of_week` | `fri` | `day_of_week_mon`, `day_of_week_thu`, `day_of_week_tue`, `day_of_week_wed` |

Tất cả cột dummy có kiểu `bool` trong file CSV (True/False). Hầu hết thư viện ML (`scikit-learn`, `XGBoost` bản mới) tự hiểu là 0/1, nhưng nếu gặp lỗi kiểu dữ liệu, ép về int:
```python
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
```

---

## 5. Lưu ý khi dùng để train

1. **Split trước, xử lý sau**: `train_test_split(..., stratify=y)` → mọi bước scale/SMOTE thực hiện *sau đó*, chỉ fit trên tập train.
2. **SMOTE**: chỉ áp dụng lên `X_train` (đã scale), không áp dụng lên test. Có thể dùng `imblearn.pipeline.Pipeline` để tránh leakage khi cross-validate.
3. **1,692 dòng trùng lặp**: tồn tại trong file do hệ quả của việc bucket `age`/`duration` (làm mất độ chi tiết, khiến các dòng vốn khác nhau trở nên giống hệt). Không phải lỗi dữ liệu gốc — khuyến nghị **giữ nguyên**, không dedupe thêm lần nữa, nhưng cần ghi chú trong báo cáo.
4. **`duration`**: cân nhắc loại khỏi tập feature nếu mục tiêu là dự đoán.
5. Khi lưu model để dùng lại cho `predict.py`/API, cần lưu kèm: `scaler.pkl` (StandardScaler fit trên train) + danh sách cột sau one-hot (để đảm bảo input mới được align đúng thứ tự cột, tránh lệch khi 1 khách hàng mới có category chưa từng gặp lúc train).