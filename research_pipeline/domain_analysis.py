import os
import numpy as np
import pandas as pd
import json
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def calculate_domain_shift(gmap_feature, foody_feature):
    w_dist = wasserstein_distance(gmap_feature, foody_feature)
    hist_gmap, bin_edges = np.histogram(gmap_feature, bins=20, density=True)
    hist_foody, _ = np.histogram(foody_feature, bins=bin_edges, density=True)
    
    hist_gmap = hist_gmap + 1e-10
    hist_foody = hist_foody + 1e-10
    
    kl_div = entropy(hist_gmap, hist_foody)
    js_div = jensenshannon(hist_gmap, hist_foody)
    
    return w_dist, kl_div, js_div

def plot_feature_distributions(df_gmap, df_foody, feature_col, save_path):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(df_gmap[feature_col].dropna(), label='Google Maps (Source)', fill=True, color='blue')
    sns.kdeplot(df_foody[feature_col].dropna(), label='Foody (Target)', fill=True, color='orange')
    plt.title(f'Domain Distribution Shift: {feature_col}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    print("🔍 BẮT ĐẦU PHÂN TÍCH DOMAIN SHIFT...")
    os.makedirs(os.path.join(PROJECT_ROOT, "reports/figures"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "reports/metrics"), exist_ok=True)

    # Đọc dữ liệu (bỏ qua void để so sánh POI thật)
    df_gmap = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes_google_maps.csv"))
    df_foody = pd.read_csv(os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes_foody.csv"))
    
    df_gmap = df_gmap[~df_gmap['Source'].str.contains('void')]
    df_foody = df_foody[~df_foody['Source'].str.contains('void')]

    # So sánh phân phối Không gian (Lat/Lon)
    w_dist, kl_div, js_div = calculate_domain_shift(df_gmap['Lat'].values, df_foody['Lat'].values)
    
    print(f"✅ Vĩ độ (Lat) - Wasserstein: {w_dist:.4f} | KL Divergence: {kl_div:.4f} | JS Divergence: {js_div:.4f}")
    
    plot_feature_distributions(df_gmap, df_foody, 'Lat', os.path.join(PROJECT_ROOT, "reports/figures/domain_shift_lat.png"))
    plot_feature_distributions(df_gmap, df_foody, 'Lon', os.path.join(PROJECT_ROOT, "reports/figures/domain_shift_lon.png"))

    # Lưu kết quả để generate_report.py đọc
    scores = {"Wasserstein": w_dist, "KL_Divergence": kl_div, "JS_Divergence": js_div}
    with open(os.path.join(PROJECT_ROOT, "reports/metrics/domain_shift.json"), "w") as f:
        json.dump(scores, f)
    
    print("📁 Đã lưu biểu đồ tại: reports/figures/domain_shift_*.png")