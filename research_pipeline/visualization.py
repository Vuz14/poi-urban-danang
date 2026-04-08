import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def plot_cross_domain_umap(gmap_emb, foody_emb, save_path="reports/figures/umap_embedding.png"):
    """Vẽ UMAP để xem hai domain có overlap (hiểu chung ngữ nghĩa) hay không"""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    
    all_emb = np.vstack((gmap_emb, foody_emb))
    labels = ['Google Maps'] * len(gmap_emb) + ['Foody'] * len(foody_emb)
    
    embedding = reducer.fit_transform(all_emb)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, alpha=0.6, palette="Set1")
    plt.title('UMAP Cross-Domain Representation (Google Maps vs Foody)')
    plt.savefig(save_path)
    plt.close()

def plot_embedding_distance_heatmap(gmap_emb, foody_emb, save_path="reports/figures/distance_heatmap.png"):
    """Heatmap đo sự tương đồng chéo (Bổ sung 4)"""
    # Lấy sample 100 quán mỗi bên để vẽ cho nhẹ
    sim_matrix = cosine_similarity(gmap_emb[:100], foody_emb[:100])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap='viridis')
    plt.title('Semantic Alignment Heatmap: Google Maps ↔ Foody')
    plt.xlabel('Foody Samples')
    plt.ylabel('Google Maps Samples')
    plt.savefig(save_path)
    plt.close()