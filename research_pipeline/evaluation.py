import os
import json
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def evaluate_clustering(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    preds = kmeans.labels_
    
    # Bắt lỗi nếu chỉ có 1 cụm
    if len(set(preds)) > 1:
        return {
            "Silhouette": float(silhouette_score(embeddings, preds)),
            "Davies-Bouldin": float(davies_bouldin_score(embeddings, preds)),
        }
    return {"Silhouette": 0.0, "Davies-Bouldin": 0.0}

if __name__ == "__main__":
    print("📊 ĐANG TÍNH TOÁN METRICS PHÂN CỤM...")
    
    gmap_emb = np.load(os.path.join(PROJECT_ROOT, "reports/embeddings/gmap_emb.npy"))
    foody_emb = np.load(os.path.join(PROJECT_ROOT, "reports/embeddings/foody_emb.npy"))
    
    gmap_metrics = evaluate_clustering(gmap_emb)
    foody_metrics = evaluate_clustering(foody_emb)
    
    print(f"📍 Google Maps: Silhouette = {gmap_metrics['Silhouette']:.4f} | DB = {gmap_metrics['Davies-Bouldin']:.4f}")
    print(f"📍 Foody:       Silhouette = {foody_metrics['Silhouette']:.4f} | DB = {foody_metrics['Davies-Bouldin']:.4f}")
    
    results = {"GoogleMaps": gmap_metrics, "Foody": foody_metrics}
    
    os.makedirs(os.path.join(PROJECT_ROOT, "reports/metrics"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "reports/metrics/clustering_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("📁 Đã lưu Metrics tại reports/metrics/clustering_metrics.json")