import os
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def generate_auto_conclusion(metrics, domain_shift):
    gmap = metrics['GoogleMaps']
    foody = metrics['Foody']
    
    silhouette_gap = gmap['Silhouette'] - foody['Silhouette']
    domain_gap = domain_shift['Wasserstein'] + domain_shift['JS_Divergence']
    
    report = ["=== KẾT LUẬN TỰ ĐỘNG TỪ AI CHUYÊN GIA ===", ""]
    
    report.append(f"1. Phân tích Dịch chuyển Miền (Domain Gap: {domain_gap:.4f}):")
    if domain_gap > 0.5:
        report.append("  -> RẤT LỚN: Phân phối của Google Maps và Foody có sự khác biệt cấu trúc mạnh mẽ.")
    else:
        report.append("  -> VỪA PHẢI: Dữ liệu hai nguồn có sự tương đồng nhất định.")
        
    report.append(f"\n2. Đánh giá Zero-shot Generalization (Khả năng khái quát hóa):")
    report.append(f"  -> Điểm Silhouette Google Maps: {gmap['Silhouette']:.4f}")
    report.append(f"  -> Điểm Silhouette Foody      : {foody['Silhouette']:.4f}")
    
    if abs(silhouette_gap) < 0.15:
        report.append("\n✅ KẾT LUẬN: MÔ HÌNH ĐÃ TRÁNH ĐƯỢC OVERFITTING!")
        report.append("  -> Gap rất nhỏ. Mô hình Spatial AI đã học được tri thức Không gian đô thị cốt lõi (Core Representation) thay vì chỉ học thuộc Google Maps.")
    else:
        report.append("\n❌ KẾT LUẬN: MÔ HÌNH CÓ DẤU HIỆU OVERFIT.")
        report.append("  -> Hiệu năng sụt giảm nghiêm trọng khi mang sang Foody. Cần kiểm tra lại các bước Augmentation hoặc tăng cường Negative Samples.")
        
    with open(os.path.join(PROJECT_ROOT, "reports/conclusion.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print("\n".join(report))

if __name__ == "__main__":
    print("\n📝 ĐANG XUẤT BÁO CÁO TỰ ĐỘNG...\n")
    
    with open(os.path.join(PROJECT_ROOT, "reports/metrics/clustering_metrics.json"), "r") as f:
        metrics = json.load(f)
        
    with open(os.path.join(PROJECT_ROOT, "reports/metrics/domain_shift.json"), "r") as f:
        domain_shift = json.load(f)
        
    generate_auto_conclusion(metrics, domain_shift)
    
    # Lưu CSV Summary
    df = pd.DataFrame({
        "Metric": ["Silhouette Score", "Davies-Bouldin"],
        "Google Maps (Source)": [metrics['GoogleMaps']['Silhouette'], metrics['GoogleMaps']['Davies-Bouldin']],
        "Foody (Target/Zero-shot)": [metrics['Foody']['Silhouette'], metrics['Foody']['Davies-Bouldin']]
    })
    
    os.makedirs(os.path.join(PROJECT_ROOT, "reports/tables"), exist_ok=True)
    df.to_csv(os.path.join(PROJECT_ROOT, "reports/tables/clustering_metrics.csv"), index=False)
    print("\n📁 Đã lưu bảng metrics tại: reports/tables/clustering_metrics.csv")