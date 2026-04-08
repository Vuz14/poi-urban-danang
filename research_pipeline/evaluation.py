from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

def evaluate_clustering(embeddings, labels, n_clusters=10):
    """Đánh giá chất lượng phân cụm"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    preds = kmeans.labels_
    
    return {
        "Silhouette": silhouette_score(embeddings, preds),
        "Davies-Bouldin": davies_bouldin_score(embeddings, preds),
        "Calinski-Harabasz": calinski_harabasz_score(embeddings, preds)
    }

def calculate_generalization_gap(metrics_gmap, metrics_foody):
    """Tính độ sụt giảm hiệu năng (Càng nhỏ càng chứng tỏ Model Generalized tốt)"""
    gap = {}
    for key in metrics_gmap.keys():
        gap[key] = metrics_gmap[key] - metrics_foody[key]
    return gap

def evaluate_retrieval_precision_at_k(query_embeddings, corpus_embeddings, corpus_labels, query_labels, k=10):
    """Đánh giá khả năng Recommend / Retrieval (Precision@K)"""
    nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(corpus_embeddings)
    distances, indices = nn.kneighbors(query_embeddings)
    
    precision_scores = []
    for i, query_label in enumerate(query_labels):
        retrieved_labels = corpus_labels[indices[i]]
        # Nếu Category của kết quả trả về trùng với Query Category -> True
        hits = np.sum(retrieved_labels == query_label)
        precision_scores.append(hits / k)
        
    return np.mean(precision_scores)