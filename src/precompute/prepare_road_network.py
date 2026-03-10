import pandas as pd
import numpy as np
import torch
import osmnx as ox
import networkx as nx
from tqdm import tqdm
import os

def compute_street_distances():
    print("🌍 Bắt đầu tải Bản đồ Giao thông Đà Nẵng từ OpenStreetMap...")
    # Tải mạng lưới đường đi ô tô/xe máy của Đà Nẵng
    G = ox.graph_from_place('Da Nang, Vietnam', network_type='drive')
    print(f"Đã tải xong mạng lưới với {len(G.nodes)} nút giao thông!")

    df = pd.read_csv("dataset/processed/poi_processed_data.csv")
    n = len(df)
    
    print("📍 Đang gieo (map) 553 quán ăn vào các con đường gần nhất...")
    # Tìm node giao thông gần nhất với vĩ độ/kinh độ của từng quán ăn
    nodes = ox.distance.nearest_nodes(G, X=df['Lon'].values, Y=df['Lat'].values)
    
    dist_matrix = np.zeros((n, n))
    
    print("🚗 Đang tính toán đường đi ngắn nhất giữa tất cả các quán ăn...")
    for i in tqdm(range(n)):
        for j in range(n):
            if i == j:
                dist_matrix[i][j] = 0.0
            elif i < j:
                try:
                    # Tính khoảng cách thực tế (theo mét)
                    l = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')
                    dist_matrix[i][j] = l
                    dist_matrix[j][i] = l
                except nx.NetworkXNoPath:
                    # Nếu 2 điểm bị chia cắt (VD: 1 điểm ngoài đảo), phạt khoảng cách lớn
                    dist_matrix[i][j] = 15000.0 
                    dist_matrix[j][i] = 15000.0

    # Lưu ma trận lại để dùng cho file main.py
    os.makedirs("dataset/processed", exist_ok=True)
    torch.save(torch.tensor(dist_matrix, dtype=torch.float32), "dataset/processed/street_dist_matrix.pt")
    print("✅ HOÀN TẤT! Đã lưu ma trận khoảng cách thực tế tại: dataset/processed/street_dist_matrix.pt")

if __name__ == "__main__":
    compute_street_distances()