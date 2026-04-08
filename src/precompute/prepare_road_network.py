import os
import osmnx as ox
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import networkx as nx

def compute_street_distances(poi_path, void_path, output_file):
    print(f"\n🚙 Đang xử lý Mạng lưới giao thông cho: {poi_path}...")
    ox.settings.timeout = 600
    ox.settings.use_cache = True
    
    try:
        G = ox.graph_from_place('Da Nang, Vietnam', network_type='drive')
    except Exception as e:
        print(f"Lỗi tải Đà Nẵng, chuyển sang Quận Hải Châu... ({e})")
        G = ox.graph_from_place('Hai Chau District, Da Nang, Vietnam', network_type='drive')
    
    if not os.path.exists(poi_path):
        print(f"❌ Không tìm thấy: {poi_path}")
        return

    df_poi = pd.read_csv(poi_path)
    lats = list(df_poi['Lat'])
    lons = list(df_poi['Lon'])
    n_pois = len(df_poi)
    
    if os.path.exists(void_path):
        df_void = pd.read_csv(void_path)
        lats += list(df_void['Lat'])
        lons += list(df_void['Lon'])
        n_voids = len(df_void)
    else:
        n_voids = 0
        
    n_total = n_pois + n_voids
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    
    dist_matrix = np.zeros((n_pois, n_total), dtype=np.float32)
    
    for i in tqdm(range(n_pois)):
        try:
            lengths = nx.single_source_dijkstra_path_length(G, nodes[i], weight='length')
            for j in range(n_total):
                dist_matrix[i][j] = lengths.get(nodes[j], 15000.0)
        except Exception:
            dist_matrix[i, :] = 15000.0
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(torch.tensor(dist_matrix, dtype=torch.float32), output_file)
    print(f"✅ Đã lưu ma trận tại: {output_file}")

if __name__ == "__main__":
    datasets = [
        ("dataset/processed/poi_processed_gmap.csv", "dataset/sampling/urban_voids_gmap.csv", "dataset/processed/street_dist_matrix_gmap.pt"),
        ("dataset/processed/poi_processed_foody.csv", "dataset/sampling/urban_voids_foody.csv", "dataset/processed/street_dist_matrix_foody.pt")
    ]
    for p, v, o in datasets:
        compute_street_distances(p, v, o)