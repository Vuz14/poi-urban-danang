# 🗺️ HỌC BIỂU DIỄN VÙNG ĐÔ THỊ ĐA PHƯƠNG THỨC 
**(Multimodal Urban Region Representation Learning)**

> **Đồ án Chuyên ngành 1 - Ngành Trí tuệ Nhân tạo** > **Trường Đại học CNTT & TT Việt-Hàn (VKU)** > **Sinh viên thực hiện:** Lê Tiến Vũ (23AI055), Nguyễn Minh Nhật (23AI033) & Nhóm  
> **Giảng viên hướng dẫn:** ThS. Lê Đình Nguyên

---

## 📖 Giới thiệu (Overview)
Dự án này tập trung xây dựng một hệ thống Trí tuệ Nhân tạo Không gian (Spatial AI) có khả năng **"hiểu"** và **phân cụm** các khu vực đô thị (ví dụ: TP. Đà Nẵng) mà không cần con người gán nhãn (Unsupervised Learning). 

Hệ thống sử dụng **Mô hình Ngôn ngữ Lớn (LLM)** để làm sạch dữ liệu, kết hợp **ResNet** và **CLIP** để trích xuất đặc trưng Đa phương thức (Hình dáng Tòa nhà + Ảnh quán ăn + Text Review). Sau đó, mô hình **Distance-biased Transformer** sẽ học cách phân cụm các địa điểm này và gợi ý vị trí kinh doanh tối ưu (Site Selection).

---

## 🛠️ Cài đặt Môi trường (Installation)

Dự án yêu cầu Python 3.9 trở lên. Khuyến khích sử dụng môi trường ảo (`venv` hoặc `conda`).

1. **Clone repository về máy:**
   ```bash
   git clone <link-github-của-bạn>
   cd poi-urban-danang
2. **Cài đặt thư viện:**
pip install -r requirements.txt
3. **Giai đoạn 1: Chuẩn bị Dữ liệu (Data Preprocessing)**
**Dữ liệu thô ban đầu là danh sách 553 quán ăn từ Foody/Google Places. Chúng ta cần bổ sung thêm dữ liệu không gian.**

python src/precompute/pds_sampler.py
python src/precompute/prepare_road_network.py
python src/precompute/crop_buildings.py
python main.py