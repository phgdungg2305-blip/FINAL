# 📈 Hệ Thống Kiểm Định Định Lượng: Chiến Lược Kết Hợp SMA & OBV (Sàn HOSE)

Ứng dụng Web tương tác trên nền tảng **Streamlit** dùng để kiểm định (backtest) và đánh giá tính hiệu quả thực nghiệm của chiến lược giao dịch định lượng kết hợp giữa **Đường trung bình động giản đơn (SMA)** và **Chỉ báo Dòng tiền Khối lượng (On-Balance Volume - OBV)** trên thị trường chứng khoán Việt Nam (dữ liệu sàn HOSE 2020 - 2023).

---

## 🌟 Tính Năng Nổi Bật của Ứng Dụng

- **Sàng lọc cổ phiếu tự động (TA Screening)**: Tự động xếp hạng toàn bộ cổ phiếu trên sàn HOSE dựa trên 4 trụ cột kỹ thuật: **Xu hướng (Trend - 30%)**, **Động lượng (Momentum - 25%)**, **Khối lượng (Volume - 25%)** và **Rủi ro (Risk - 20%)** để chọn ra Top cổ phiếu tiềm năng nhất (mặc định Top 5: `DIG, DPM, KBC, MSN, DCM`).
- **Chiến lược kết hợp SMA + OBV theo cổng OR**:
  - Tín hiệu **MUA**: Kích hoạt khi **SMA MUA** hoặc **OBV MUA**.
  - Tín hiệu **BÁN**: Kích hoạt khi **SMA BÁN** hoặc **OBV BÁN**.
  - Xử lý xung đột: Giữ vị thế trung lập (0) nếu hai chỉ báo đối nghịch nhau trong cùng ngày.
- **Loại bỏ hoàn toàn Look-ahead Bias**:
  - Tín hiệu phát sinh ở ngày $t$ chỉ được thực thi và tính lãi/lỗ từ phiên $t+1$.
  - Cơ chế **Warm-up 349 phiên giao dịch** trước ngày kiểm định Test để tính toán chính xác chỉ báo SMA dài hạn (lên tới 350 phiên) ngay từ phiên đầu tiên của năm 2022 mà không sử dụng dữ liệu tương lai.
- **Tối ưu hóa danh mục hiện đại (MPT - Markowitz)**: Tự động tối đa hóa tỷ lệ Sharpe trên tập Train để xác định tỷ trọng vốn tối ưu cho từng cổ phiếu và kiểm định tính bền vững ngoài mẫu (Out-Of-Sample) so với phương pháp Đẳng trọng số (Equal Weight) và Mua & Nắm giữ (Buy & Hold).
- **Trực quan hóa tương tác cao cấp (Interactive Plotly)**: Biểu đồ nến giá, chỉ báo kỹ thuật, điểm mua/bán, đường cong tăng trưởng tài sản (Equity Curve), mức sụt giảm (Drawdown), và cơ cấu tỷ trọng danh mục.

---

## 📊 Kết Quả Thực Nghiệm Nổi Bật (Kỳ Test 2022 - Downtrend)

Năm 2022 là năm thị trường chứng khoán Việt Nam sụt giảm khốc liệt (VN-Index giảm hơn 30%). Kết quả thực nghiệm cho thấy sự vượt trội của chiến lược:

| Chỉ số Hiệu Suất | Equal Weight Buy & Hold | SMA + OBV Equal Weight | SMA + OBV MPT Tối Ưu |
| :--- | :---: | :---: | :---: |
| **Total Return [%]** | **-50.41%** | **-0.85%** | **+8.08%** |
| **Sharpe Ratio** | **-1.3821** | **+0.0874** | **+0.4407** |
| **Max Drawdown [%]** | **-64.30%** | **-28.08%** | **-21.19%** |
| **Annual Volatility [%]**| 44.19% | 24.64% | 24.75% |

> **💡 Kết luận thực nghiệm:** Chiến lược SMA+OBV (OR) phát huy năng lực **phòng thủ và bảo toàn vốn tuyệt đối** trong downtrend bằng cách chủ động thoát hàng về tiền mặt khi xu hướng hoặc dòng tiền gãy đổ. Danh mục tối ưu MPT đạt **lợi nhuận dương +8.08%**, vượt trội hơn **58.49%** so với thị trường mua nắm giữ truyền thống.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                      # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── README.md                   # Tài liệu hướng dẫn chi tiết dự án
├── HOSE_2020_2023_in.csv       # File dữ liệu lịch sử giá sàn HOSE
└── nhom_5.ipynb                # Jupyter Notebook phân tích gốc
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local Machine)

### Bước 1: Clone kho lưu trữ hoặc tải mã nguồn về máy
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### Bước 2: Tạo môi trường ảo Python (Khuyến nghị)
- **Trên Windows:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Deploy Miễn Phí Lên Streamlit Community Cloud

Bạn có thể đưa Web App này lên Internet hoàn toàn miễn phí chỉ với vài bước đơn giản:

### Bước 1: Đẩy mã nguồn lên GitHub
1. Đăng nhập vào [GitHub](https://github.com/) và tạo một Repository mới (ví dụ: `quant-sma-obv-hose`).
2. Mở terminal tại thư mục chứa dự án và chạy các lệnh:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: SMA + OBV Quantitative Backtest App"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo-name>.git
   git push -u origin main
   ```
*(Lưu ý: File `HOSE_2020_2023_in.csv` có dung lượng ~5.7MB, hoàn toàn nằm trong giới hạn cho phép 100MB của GitHub).*

### Bước 2: Deploy trên Streamlit Cloud
1. Truy cập vào [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"New app"** (hoặc "Create app").
3. Chọn các thông số:
   - **Repository**: Chọn repository bạn vừa tạo (ví dụ: `<username>/quant-sma-obv-hose`).
   - **Branch**: `main`.
   - **Main file path**: `app.py`.
4. Nhấn nút **"Deploy!"**.
5. Đợi 1-2 phút để hệ thống tự động cài đặt các gói trong `requirements.txt`. Sau khi hoàn tất, ứng dụng sẽ có một liên kết web công khai (URL) để bạn chia sẻ cho mọi người!

---

## 🛠️ Công Nghệ Sử Dụng

- **Python 3.9+**
- **Streamlit**: Framework xây dựng ứng dụng web tương tác nhanh chóng cho Khoa học dữ liệu.
- **Plotly**: Trực quan hóa dữ liệu tài chính chuyên sâu với biểu đồ đa lớp tương tác.
- **Pandas & NumPy**: Xử lý dữ liệu chuỗi thời gian lớn và tính toán đại số ma trận.
- **SciPy (SLSQP)**: Giải thuật tối ưu hóa phi tuyến có ràng buộc cho mô hình danh mục Markowitz.
- **TA (Technical Analysis Library)**: Thư viện chỉ báo phân tích kỹ thuật chuẩn quốc tế.
