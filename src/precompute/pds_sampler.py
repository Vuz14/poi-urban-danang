import numpy as np
import pandas as pd
import folium
import os
import math
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

def bridson_pds(width, height, r, k=30):
    """
    Thuật toán Bridson cho Poisson Disk Sampling 2D.
    r: khoảng cách tối thiểu giữa các điểm.
    k: số lần thử trước khi từ bỏ một điểm anchor.
    """
    grid_size = r / math.sqrt(2)
    cols = int(math.ceil(width / grid_size))
    rows = int(math.ceil(height / grid_size))
    
    grid = [None] * (cols * rows)
    active_list = []
    points = []

    start_pt = np.array([np.random.uniform(0, width), np.random.uniform(0, height)])
    points.append(start_pt)
    active_list.append(start_pt)
    
    grid[int(start_pt[0]/grid_size) + int(start_pt[1]/grid_size) * cols] = start_pt

    while active_list:
        idx = np.random.randint(len(active_list))
        anchor = active_list[idx]
        found = False
        
        for _ in range(k):
            angle = 2 * math.pi * np.random.random()
            dist = np.random.uniform(r, 2*r)
            new_pt = anchor + np.array([dist * math.cos(angle), dist * math.sin(angle)])
            
            if 0 <= new_pt[0] < width and 0 <= new_pt[1] < height:
                col = int(new_pt[0] / grid_size)
                row = int(new_pt[1] / grid_size)
                
                too_close = False
                for r_off in range(max(0, row-1), min(rows, row+2)):
                    for c_off in range(max(0, col-1), min(cols, col+2)):
                        neighbor = grid[c_off + r_off * cols]
                        if neighbor is not None:
                            if np.linalg.norm(new_pt - neighbor) < r:
                                too_close = True
                                break
                    if too_close: break
                
                if not too_close:
                    points.append(new_pt)
                    active_list.append(new_pt)
                    grid[col + row * cols] = new_pt
                    found = True
                    break
        
        if not found:
            active_list.pop(idx)
            
    return np.array(points)

def generate_urban_negatives(poi_csv, output_csv, min_dist_m=200):
    """
    Tạo các điểm 'Negative samples' bằng PDS tránh các POI hiện có.
    min_dist_m: Khoảng cách tối thiểu giữa các điểm PDS (mét).
    """
    print("📍 Đang đọc dữ liệu POI hiện có...")
    df_poi = pd.read_csv(poi_csv)
    
    min_lat, max_lat = df_poi['Lat'].min(), df_poi['Lat'].max()
    min_lon, max_lon = df_poi['Lon'].min(), df_poi['Lon'].max()
    
    r_lat = min_dist_m / 111000.0
    r_lon = min_dist_m / 105000.0
    r_norm = (r_lat + r_lon) / 2.0
    
    width = max_lon - min_lon
    height = max_lat - min_lat
    
    print(f"🔄 Đang chạy Poisson Disk Sampling (r ≈ {min_dist_m}m)...")
    raw_points = bridson_pds(width, height, r_norm)
    
    pds_lons = raw_points[:, 0] + min_lon
    pds_lats = raw_points[:, 1] + min_lat
    
    print("🧹 Đang lọc các điểm nằm trong vùng đô thị hiện hữu...")
    poi_points = MultiPoint([Point(lon, lat) for lon, lat in zip(df_poi['Lon'], df_poi['Lat'])])
    
    filtered_points = []
    exclusion_radius = 100 / 111000.0 
    
    for lon, lat in zip(pds_lons, pds_lats):
        p = Point(lon, lat)
        nearest_poi = nearest_points(p, poi_points)[1]
        if p.distance(nearest_poi) > exclusion_radius:
            filtered_points.append([lat, lon])
            
    df_neg = pd.DataFrame(filtered_points, columns=['Lat', 'Lon'])
    df_neg['Category'] = 'Empty_Space'
    df_neg['Name'] = 'Urban_Void'
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_neg.to_csv(output_csv, index=False)
    print(f"✅ Đã tạo {len(df_neg)} điểm vùng trống. Lưu tại: {output_csv}")
    
    return df_poi, df_neg

def visualize_pds_map(df_poi, df_neg, map_path="reports/urban_voids_map.html"):
    print("🗺️ Đang vẽ bản đồ kiểm tra...")
    m = folium.Map(location=[df_poi['Lat'].mean(), df_poi['Lon'].mean()], zoom_start=13)
    
    for _, row in df_poi.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.5,
            popup="POI Hiện hữu"
        ).add_to(m)
        
    for _, row in df_neg.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup="Vùng trống (Negative)"
        ).add_to(m)
        
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    m.save(map_path)
    print(f"✅ Bản đồ đã lưu tại: {map_path}")

if __name__ == "__main__":
    poi_file = "dataset/processed/poi_processed_data.csv"
    output_file = "dataset/sampling/urban_voids.csv"
    
    if os.path.exists(poi_file):
        df_p, df_n = generate_urban_negatives(poi_file, output_file)
        visualize_pds_map(df_p, df_n)
    else:
        print(f"❌ Không tìm thấy file {poi_file}")
