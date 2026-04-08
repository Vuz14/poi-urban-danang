import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_domain_shift(gmap_feature, foody_feature):
    """Tính các metric đo lường Domain Shift giữa 2 phân phối"""
    # 1. Wasserstein Distance (Earth Mover's Distance)
    w_dist = wasserstein_distance(gmap_feature, foody_feature)
    
    # Chuẩn hóa để tính KL và JS
    hist_gmap, bin_edges = np.histogram(gmap_feature, bins=20, density=True)
    hist_foody, _ = np.histogram(foody_feature, bins=bin_edges, density=True)
    
    # Add epsilon để tránh log(0)
    hist_gmap = hist_gmap + 1e-10
    hist_foody = hist_foody + 1e-10
    
    # 2. Kullback-Leibler Divergence
    kl_div = entropy(hist_gmap, hist_foody)
    
    # 3. Jensen-Shannon Divergence
    js_div = jensenshannon(hist_gmap, hist_foody)
    
    return w_dist, kl_div, js_div

def plot_feature_distributions(df_gmap, df_foody, feature_col, save_path):
    """Vẽ Histogram so sánh Distribution"""
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df_gmap[feature_col], label='Google Maps (Train)', fill=True, color='blue')
    sns.kdeplot(df_foody[feature_col], label='Foody (Test)', fill=True, color='orange')
    plt.title(f'Domain Distribution: {feature_col}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()