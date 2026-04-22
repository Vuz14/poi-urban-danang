import os
import warnings
import multiprocessing
from requests import RequestsDependencyWarning

from config import VERSIONS_TO_TRAIN
from src.training.engine import train_urban_ai
from research_pipeline.visualization import (
    plot_training_loss, plot_silhouette_scores, plot_recall_at_5,
    plot_similarity_curves, plot_tsne_clusters, plot_group_tsne_clusters, plot_umap_clusters
)

warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

def get_output_paths(version):
    """Tự động sinh các thư mục chứa Logs, Models và Figures theo Version."""
    base_dir = f"results/v{version}"
    paths = {
        "models": f"{base_dir}/models_saved",
        "metrics": f"{base_dir}/reports/metrics",
        "figures": f"{base_dir}/reports/figures"
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # CHẠY LẦN LƯỢT CÁC VERSION ĐƯỢC CHỈ ĐỊNH (Theo Config.py)
    for v in VERSIONS_TO_TRAIN:
        print(f"\n{'='*80}")
        print(f"🚀 BẮT ĐẦU CHUỖI HUẤN LUYỆN ĐỘC LẬP CHO VERSION {v}")
        print(f"{'='*80}")
        
        # 1. Tạo thư mục chứa riêng cho Version này
        paths = get_output_paths(v)
        
        # 2. Train và lưu vào thư mục riêng
        train_urban_ai(v, paths)
        
        # 3. Trực quan hóa và lưu vào thư mục riêng
        print(f"🎨 Đang vẽ biểu đồ kết quả cho Version {v}...")
        plot_training_loss(v, paths)
        plot_silhouette_scores(v, paths)
        plot_recall_at_5(v, paths)
        plot_similarity_curves(v, paths)
        plot_tsne_clusters(v, paths)
        plot_group_tsne_clusters(v, paths)
        plot_umap_clusters(v, paths)
        
        print(f"📦 Toàn bộ báo cáo, file csv, hình ảnh và Model V{v} đã được đóng gói tại: results/v{v}/")