import os
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def plot_cross_domain_umap(gmap_emb, foody_emb, save_path):
    print("🌌 Đang chiếu không gian UMAP Cross-Domain...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    
    # Lấy sample 500 điểm mỗi bên cho nhẹ máy
    g_sample = gmap_emb[:500]
    f_sample = foody_emb[:500]
    
    all_emb = np.vstack((g_sample, f_sample))
    labels = ['Google Maps (Source)'] * len(g_sample) + ['Foody (Target)'] * len(f_sample)
    
    embedding = reducer.fit_transform(all_emb)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, alpha=0.7, palette="Set1")
    plt.title('UMAP Cross-Domain Representation')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    gmap_emb = np.load(os.path.join(PROJECT_ROOT, "reports/embeddings/gmap_emb.npy"))
    foody_emb = np.load(os.path.join(PROJECT_ROOT, "reports/embeddings/foody_emb.npy"))
    
    os.makedirs(os.path.join(PROJECT_ROOT, "reports/figures"), exist_ok=True)
    
    plot_cross_domain_umap(gmap_emb, foody_emb, os.path.join(PROJECT_ROOT, "reports/figures/umap_cross_domain.png"))
    print("✅ Đã vẽ và lưu UMAP tại reports/figures/umap_cross_domain.png")