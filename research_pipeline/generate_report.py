import pandas as pd
import json

def generate_auto_conclusion(gmap_metrics, foody_metrics, domain_shift_scores):
    """Hệ chuyên gia tự động kết luận dựa trên Metrics"""
    
    silhouette_gap = gmap_metrics['Silhouette'] - foody_metrics['Silhouette']
    domain_gap = domain_shift_scores['Wasserstein'] + domain_shift_scores['JS_Divergence']
    
    report = []
    report.append("=== AUTO-ANALYSIS CONCLUSION ===")
    
    # 1. Đánh giá Domain Shift
    if domain_gap > 0.5:
        report.append("📌 Domain Shift: RẤT LỚN. Google Maps và Foody có sự khác biệt cấu trúc dữ liệu rõ rệt.")
    else:
        report.append("📌 Domain Shift: VỪA PHẢI. Dữ liệu hai nguồn có sự tương đồng nhất định.")
        
    # 2. Đánh giá Overfitting & Generalization
    if silhouette_gap < 0.1:
        report.append("✅ OVERFITTING CHECK: PASSED. Generalization Gap nhỏ (Dropout = {:.3f}). Mô hình đã học được Tri thức cốt lõi (Core Representation) thay vì chỉ học thuộc Google Maps.".format(silhouette_gap))
    else:
        report.append("❌ OVERFITTING CHECK: FAILED. Mô hình bị overfit nặng vào Google Maps (Gap = {:.3f}), hiệu năng sụt giảm nghiêm trọng trên Foody.".format(silhouette_gap))
        
    with open("reports/conclusion.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print("\n".join(report))

# Bảng Summary Table (Yêu cầu 10)
def export_summary_table(gmap_metrics, foody_metrics):
    df = pd.DataFrame({
        "Metric": ["Silhouette Score", "Davies-Bouldin", "Precision@10"],
        "Google Maps (Train)": [gmap_metrics['Silhouette'], gmap_metrics['Davies-Bouldin'], gmap_metrics['Precision@10']],
        "Foody (Test)": [foody_metrics['Silhouette'], foody_metrics['Davies-Bouldin'], foody_metrics['Precision@10']]
    })
    df.to_csv("reports/tables/clustering_metrics.csv", index=False)