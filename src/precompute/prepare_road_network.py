import os
import osmnx as ox
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import networkx as nx

def compute_street_distances():
    POI_PATH = "dataset/processed/poi_processed_data.csv"
    VOID_PATH = "dataset/sampling/urban_voids.csv"
    OUTPUT_FILE = "dataset/processed/street_dist_matrix.pt"
    
    ox.settings.timeout = 600
    ox.settings.use_cache = True
    
    print("🌎 Đang tải mạng lưới giao thông Đà Nẵng (OSMnx)...")
    try:
        G = ox.graph_from_place('Da Nang, Vietnam', network_type='drive')
    except Exception as e:
        print(f"Lỗi tải Đà Nẵng, chuyển sang Quận Hải Châu... ({e})")
        G = ox.graph_from_place('Hai Chau District, Da Nang, Vietnam', network_type='drive')
    
    if not os.path.exists(POI_PATH):
        return

    df_poi = pd.read_csv(POI_PATH)
    lats = list(df_poi['Lat'])
    lons = list(df_poi['Lon'])
    n_pois = len(df_poi)
    
    if os.path.exists(VOID_PATH):
        df_void = pd.read_csv(VOID_PATH)
        lats += list(df_void['Lat'])
        lons += list(df_void['Lon'])
        n_voids = len(df_void)
    else:
        n_voids = 0
        
    n_total = n_pois + n_voids
    print(f"📍 Mapping: {n_pois} POIs + {n_voids} Voids TOÀN BỘ lên đồ thị (Tổng {n_total})...")
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    
    # TINH CHỈNH MA TRẬN 
    # Ma trận (553 x 125,672). Tránh tạo (125k x 125k) gây tràn 63GB RAM.
    dist_matrix = np.zeros((n_pois, n_total), dtype=np.float32)
    
    print("🚗 Đang tính toán Dijkstra từ POIs đến toàn bộ Voids (125k+)...")
    for i in tqdm(range(n_pois)):
        try:
            lengths = nx.single_source_dijkstra_path_length(G, nodes[i], weight='length')
            for j in range(n_total):
                dist_matrix[i][j] = lengths.get(nodes[j], 15000.0)
        except Exception:
            dist_matrix[i, :] = 15000.0
            
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    torch.save(torch.tensor(dist_matrix, dtype=torch.float32), OUTPUT_FILE)
    print(f"✅ Đã lưu kết quả tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_street_distances()