# 🗺️ HỌC BIỂU DIỄN VÙNG ĐÔ THỊ ĐA PHƯƠNG THỨC 
**(Multimodal Urban Region Representation Learning & Domain Adaptation)**

## 📖 Giới thiệu

Dự án xây dựng hệ thống **Trí tuệ Nhân tạo Không gian (Spatial AI)** có khả năng phân cụm các khu vực đô thị (tại Đà Nẵng) mà không cần gán nhãn (Unsupervised Learning). Hệ thống kết hợp các kiến trúc tiên tiến gồm **LLM**, **ResNet** và **CLIP** để trích xuất đặc trưng Đa phương thức:
* 🏢 Hình dáng đa giác Tòa nhà
* 📸 Ảnh chụp thực tế quán ăn/địa điểm
* 📝 Văn bản đánh giá (Text Review)

Sau khi trích xuất, mô hình sử dụng **Distance-biased Transformer** để phân tích và đưa ra các gợi ý vị trí kinh doanh tối ưu (Site Selection).

🌟 **Điểm nhấn Nghiên cứu:** Ứng dụng **Domain Adaptation** để chứng minh mô hình thực sự học được *tri thức cốt lõi (Core Representation)* của không gian đô thị. Cụ thể, mô hình được huấn luyện hoàn toàn trên **Google Maps (Source Domain)** và kiểm thử chéo trực tiếp trên **Foody (Target Domain)** thông qua phương pháp Zero-shot Inference.

---

## 🛠️ Cài đặt Môi trường

**Yêu cầu hệ thống:** Python 3.9 trở lên. Khuyến khích sử dụng môi trường ảo (`venv` hoặc `conda`) để tránh xung đột thư viện.

```bash
# Clone repository
git clone <link-github-của-bạn>
cd poi-urban-danang

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
🚀 Pipeline Thực thi Dự án
Dưới đây là luồng chạy chuẩn của toàn bộ dự án từ khâu xử lý dữ liệu đến lúc xuất báo cáo.

## 🚀 Pipeline Thực thi Dự án (End-to-End)

Dự án đã được tự động hóa cao độ. Bạn chỉ cần chạy các lệnh sau theo thứ tự:

### Bước 1: Chuẩn bị Dữ liệu (Data Preprocessing)
Tính toán không gian, lấy mẫu vùng trống và thu thập hình dáng tòa nhà, hình ảnh từ Google Maps và Foody. Tọa độ được đồng bộ nhưng dữ liệu được tách làm 2 tập (Source và Target) độc lập.

```bash
python dataset/processed/prepare_data.py
python src/precompute/pds_sampler.py
python src/precompute/prepare_road_network.py
python src/precompute/crop_buildings.py
python src/data/download_poi_images.py
Bước 2: Phân tích Dịch chuyển Miền (Domain Shift Analysis)
Đo lường sự khác biệt về phân phối (ví dụ: Rating, Price) giữa hai nền tảng bằng các độ đo thống kê như KL Divergence & Wasserstein.

Bash
python research_pipeline/domain_analysis.py
Bước 3: Huấn luyện & Trực quan hóa Tự động (One-click Run)
File main.py là trung tâm điều phối của toàn bộ hệ thống. Chỉ với một lệnh duy nhất, hệ thống sẽ tự động thực hiện chuỗi tác vụ:

Huấn luyện mô hình Multimodal (Google Maps).

Kiểm thử chéo Zero-shot (Foody).

Tính toán tự động các metrics (Silhouette, Recall@5, Pos/Neg Similarity).

Xuất bảng dữ liệu kết quả ra file .csv.

Vẽ và lưu toàn bộ biểu đồ không gian (Loss curve, t-SNE, UMAP...).

💡 Tip: Để thay đổi cấu hình (Batch size, LR, Version...), bạn chỉ cần chỉnh sửa trong file config.py.

Bash
python main.py
📁 Cấu trúc Output (Sau khi chạy Bước 3)
Toàn bộ kết quả sẽ được tự động đóng gói gọn gàng theo từng phiên bản (ví dụ v4) để bạn dễ dàng đưa vào báo cáo:

🧠 Trọng số AI tốt nhất: results/v4/models_saved/

📑 Bảng số liệu Metrics (.csv): results/v4/reports/metrics/

📊 Biểu đồ & Hình ảnh t-SNE/UMAP: results/v4/reports/figures/