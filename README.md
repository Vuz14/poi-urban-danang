# 🗺️ HỌC BIỂU DIỄN VÙNG ĐÔ THỊ ĐA PHƯƠNG THỨC 
**(Multimodal Urban Region Representation Learning & Domain Adaptation)**


## 📖 Giới thiệu
Dự án xây dựng hệ thống **Trí tuệ Nhân tạo Không gian (Spatial AI)** có khả năng phân cụm các khu vực đô thị (Đà Nẵng) không cần gán nhãn (Unsupervised Learning). Hệ thống kết hợp **LLM**, **ResNet** và **CLIP** để trích xuất đặc trưng Đa phương thức (Hình dáng Tòa nhà + Ảnh quán ăn + Text Review), sau đó dùng **Distance-biased Transformer** để gợi ý vị trí kinh doanh tối ưu (Site Selection).

🌟 **Điểm nhấn Nghiên cứu:** Ứng dụng **Domain Adaptation** để chứng minh mô hình thực sự học được *tri thức cốt lõi (Core Representation)*. Mô hình được huấn luyện trên **Google Maps (Source Domain)** và kiểm thử chéo trực tiếp trên **Foody (Target Domain)** thông qua Zero-shot Inference.


## 🛠️ Cài đặt Môi trường
Yêu cầu: Python 3.9 trở lên. Khuyến khích sử dụng môi trường ảo (`venv` hoặc `conda`).

git clone <link-github-của-bạn>
cd poi-urban-danang
pip install -r requirements.txt




1. Chuẩn bị Dữ liệu (Data Preprocessing) 
Tính toán không gian và thu thập hình dáng đa giác tòa nhà từ tọa độ 553 POI.

Xử lý và Chuẩn hóa Dữ liệu ĐỘC LẬP (Data Cleaning): Bước này sẽ đọc 2 file thô (`poi_data_foody.csv` và `poi_data_ggmap.csv`), làm sạch, đồng bộ tên các cột (Lat/Lon) nhưng KHÔNG GỘP CHUNG. Đầu ra tạo thành 2 file xử lý riêng biệt để phục vụ Domain Adaptation.

python dataset/processed/prepare_data.py



python src/precompute/pds_sampler.py
python src/precompute/prepare_road_network.py
python src/precompute/crop_buildings.py
2. Phân tích Dịch chuyển Miền (Domain Shift Analysis) Đo lường sự khác biệt phân phối (Rating, Price) giữa Google Maps và Foody bằng KL Divergence & Wasserstein.


python research_pipeline/domain_analysis.py

3. Huấn luyện AI (Model Training) Huấn luyện Multimodal Encoder & Distance-Biased Transformer trên dữ liệu Google Maps.

split_dataset.py : để lọc bỏ các điểm đen thui trong file building_images_ggmap
thiết lập cấu hình TRAINING_VERSION (từ 1 đến 4 tương ứng 4 chế độ Ablation Study ở main.py) và chạy lệnh:
python main.py
 3 file dataset, multimodal_encoder, main.py là những file liên quan nhau tinh chỉnh chiến thuật train thì đọc cả 3 hoặc chỉnh trong main

4. Kiểm thử Chéo (Zero-shot Inference) Đóng băng trọng số (Freeze Model) và trích xuất embedding cho cả 2 tập dữ liệu.


python research_pipeline/freeze_model.py

5. Đánh giá & Trực quan hóa (Evaluation & Visualization) Tính toán các độ đo phân cụm (Silhouette, Davies-Bouldin) và vẽ bản đồ t-SNE, UMAP không gian ngữ nghĩa chéo.


python research_pipeline/evaluation.py
python research_pipeline/visualization.py

6. Xuất Báo Cáo Tự động (Auto-Conclusion) Tổng hợp metrics thành bảng .csv và xuất file text đánh giá mô hình có bị Overfitting hay không.


python research_pipeline/generate_report.py

🌐 Triển khai Web (Production Readiness)
Toàn bộ biểu đồ (reports/figures/), bảng số liệu (reports/tables/) và trọng số AI (models_saved/multimodal_best.pth) đã sẵn sàng để tích hợp: