# 🛡️ DDoS Detection System — Streamlit Demo

Hệ thống phát hiện tấn công DDoS sử dụng Machine Learning (Random Forest).

## Cài đặt và chạy

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Chạy ứng dụng
python -m streamlit run app.py
```

Mở trình duyệt tại: http://localhost:8501

## Tính năng

| Trang | Mô tả |
|---|---|
| 🏠 Tổng quan | Metrics tổng thể + thử nghiệm nhanh với slider |
| 📁 Phân tích CSV | Upload file log mạng → phân loại hàng loạt |
| 🔴 Giám sát realtime | Mô phỏng luồng gói tin liên tục, chặn IP nghi ngờ |
| 📊 Hiệu năng mô hình | Confusion Matrix, Feature Importance, phân phối xác suất |

## Dùng với dataset thực (CICIDS2017)

1. Tải dataset tại: https://www.unb.ca/cic/datasets/ids-2017.html
2. Lấy file `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
3. Upload vào trang **Phân tích CSV**

## Cấu trúc project

```
ddos-detection/
├── app.py              # Ứng dụng Streamlit chính
├── requirements.txt    # Thư viện cần thiết
├── README.md
└── data/               # Thư mục chứa dataset CSV (tùy chọn)
```
