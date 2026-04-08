# 🗺️ HỌC BIỂU DIỄN VÙNG ĐÔ THỊ ĐA PHƯƠNG THỨC 
**(Multimodal Urban Region Representation Learning & Domain Adaptation)**

## 📖 Giới thiệu
Dự án xây dựng hệ thống **Trí tuệ Nhân tạo Không gian (Spatial AI)** có khả năng phân cụm các khu vực đô thị (Đà Nẵng) không cần gán nhãn (Unsupervised Learning). Hệ thống kết hợp **LLM**, **ResNet** và **CLIP** để trích xuất đặc trưng Đa phương thức (Hình dáng Tòa nhà + Ảnh quán ăn + Text Review), sau đó dùng **Distance-biased Transformer** để gợi ý vị trí kinh doanh tối ưu (Site Selection).

🌟 **Điểm nhấn Nghiên cứu:** Ứng dụng **Domain Adaptation** để chứng minh mô hình thực sự học được *tri thức cốt lõi (Core Representation)*. Mô hình được huấn luyện trên **Google Maps (Source Domain)** và kiểm thử chéo trực tiếp trên **Foody (Target Domain)** thông qua Zero-shot Inference.

---

## 🛠️ Cài đặt Môi trường
Yêu cầu: Python 3.9 trở lên. Khuyến khích sử dụng môi trường ảo (`venv` hoặc `conda`).

```bash
git clone <link-github-của-bạn>
cd poi-urban-danang
pip install -r requirements.txt
📂 Dữ liệu Hệ thống
File dữ liệu chính thô:

D:\poi_urban\dataset\processed\poi_data_foody.csv

D:\poi_urban\dataset\processed\poi_data_ggmap.csv

**File dữ liệu chính sau khi chạy prepare_data.py:**

Tập huấn luyện (Gmap): dataset/processed/poi_processed_gmap.csv

Tập kiểm thử (Foody): dataset/processed/poi_processed_foody.csv

🚀 Quy trình Chạy Dự án (Execution Pipeline)
1. Chuẩn bị Dữ liệu (Data Preprocessing)
Gộp và Chuẩn hóa Dữ liệu (Data Merging): Bước đầu tiên bắt buộc phải chạy để gộp 2 file thô (poi_data_foody.csv và poi_data_ggmap.csv) thành một file duy nhất đồng bộ các cột. Sau đó tiến hành tính toán không gian và thu thập hình dáng đa giác tòa nhà từ tọa độ 553 POI.

 
python dataset/processed/prepare_data.py
python src/precompute/pds_sampler.py
python src/precompute/prepare_road_network.py
python src/data/get_building_footprints.py
python src/precompute/crop_buildings.py
2. Phân tích Dịch chuyển Miền (Domain Shift Analysis)
Đo lường sự khác biệt phân phối (Rating, Price) giữa Google Maps và Foody bằng KL Divergence & Wasserstein.

 
python research_pipeline/domain_analysis.py
3. Huấn luyện AI (Model Training)
Huấn luyện Multimodal Encoder & Distance-Biased Transformer trên dữ liệu Google Maps.

 
python main.py
4. Kiểm thử Chéo (Zero-shot Inference)
Đóng băng trọng số (Freeze Model) và trích xuất embedding cho cả 2 tập dữ liệu.

 
python research_pipeline/freeze_model.py
5. Đánh giá & Trực quan hóa (Evaluation & Visualization)
Tính toán các độ đo phân cụm (Silhouette, Davies-Bouldin) và vẽ bản đồ t-SNE, UMAP không gian ngữ nghĩa chéo.

 
python research_pipeline/evaluation.py
python research_pipeline/visualization.py
6. Xuất Báo Cáo Tự động (Auto-Conclusion)
Tổng hợp metrics thành bảng .csv và xuất file text đánh giá mô hình có bị Overfitting hay không.

 
python research_pipeline/generate_report.py
🌐 Triển khai Web (Production Readiness)
Toàn bộ biểu đồ (reports/figures/), bảng số liệu (reports/tables/) và trọng số AI (models_saved/multimodal_best.pth) đã sẵn sàng để tích hợp vào hệ thống Web phân tích (Frontend Next.js & Backend FastAPI).