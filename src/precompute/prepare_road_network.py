import pandas as pd
import numpy as np
import torch
import osmnx as ox
import networkx as nx
from tqdm import tqdm
import os

# Cấu hình đường dẫn
POI_PATH = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\processed\poi_processed_data.csv"
VOID_PATH = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\sampling\urban_voids.csv"
OUTPUT_FILE = r"D:\python\ChuyenDe2\poi-urban-danang\dataset\processed\street_dist_matrix.pt"

def compute_street_distances():
    if not os.path.exists(POI_PATH) or not os.path.exists(VOID_PATH):
        print("❌ Lỗi: Thiếu file POI hoặc file Urban Voids. Hãy chạy pds_sampler.py trước!")
        return

    print("🌍 Tải mạng lưới giao thông Đà Nẵng...")
    G = ox.graph_from_place('Da Nang, Vietnam', network_type='drive')
    
    # 1. Gộp tọa độ của cả POI và Voids lại thành một danh sách duy nhất
    df_poi = pd.read_csv(POI_PATH)
    df_void = pd.read_csv(VOID_PATH)
    
    # Kết hợp tọa độ: POI sẽ nằm từ index 0 đến 552, sau đó là các điểm Voids
    combined_lats = list(df_poi['Lat']) + list(df_void['Lat'])
    combined_lons = list(df_poi['Lon']) + list(df_void['Lon'])
    n_total = len(combined_lats)
    
    print(f"📍 Tổng cộng: {len(df_poi)} POIs + {len(df_void)} Voids = {n_total} điểm.")
    
    print("📍 Ánh xạ tất cả các điểm vào nút giao thông...")
    nodes = ox.distance.nearest_nodes(G, X=combined_lons, Y=combined_lats)
    
    dist_matrix = np.zeros((n_total, n_total))
    
    print("🚗 Tính toán Dijkstra cho ma trận tổng hợp...")
    for i in tqdm(range(n_total)):
        try:
            lengths = nx.single_source_dijkstra_path_length(G, nodes[i], weight='length')
            for j in range(n_total):
                dist_matrix[i][j] = lengths.get(nodes[j], 15000.0)
        except Exception:
            dist_matrix[i, :] = 15000.0

    torch.save(torch.tensor(dist_matrix, dtype=torch.float32), OUTPUT_FILE)
    print(f"✅ Đã lưu ma trận tổng hợp tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    compute_street_distances()