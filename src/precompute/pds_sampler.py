import numpy as np
import pandas as pd
import folium
import os
import math
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Cấu hình độc lập 2 Domain
DATASETS = [
    {
        "name": "Google Maps POI (Source Domain)",
        "csv_path": os.path.join(PROJECT_ROOT, "dataset/processed/poi_processed_gmap.csv"),
        "source_name": "google_maps"
    },
    {
        "name": "Foody POI (Target Domain)",
        "csv_path": os.path.join(PROJECT_ROOT, "dataset/processed/poi_processed_foody.csv"),
        "source_name": "foody"
    }
]

def bridson_pds(width, height, r, k=30):
    # [Giữ nguyên code hàm bridson_pds của bạn]
    grid_size = r / math.sqrt(2)
    cols = int(math.ceil(width / grid_size))
    rows = int(math.ceil(height / grid_size))
    
    grid = [None] * (cols * rows)
    active_list, points = [], []

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
                col, row = int(new_pt[0] / grid_size), int(new_pt[1] / grid_size)
                too_close = False
                for r_off in range(max(0, row-1), min(rows, row+2)):
                    for c_off in range(max(0, col-1), min(cols, col+2)):
                        neighbor = grid[c_off + r_off * cols]
                        if neighbor is not None and np.linalg.norm(new_pt - neighbor) < r:
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

def process_single_domain(ds_info, min_dist_m=200):
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG XỬ LÝ DOMAIN: {ds_info['name']}")
    print(f"{'='*50}")
    
    if not os.path.exists(ds_info['csv_path']):
        print(f"❌ Không tìm thấy file: {ds_info['csv_path']}")
        return
        
    df_poi = pd.read_csv(ds_info['csv_path'])
    
    # 1. Đảm bảo có cột Global_ID và Source cho POI
    if 'Global_ID' not in df_poi.columns:
        # Tự động chọn cột ID phù hợp dựa trên nguồn
        id_col = 'place_id' if ds_info['source_name'] == 'google_maps' else 'RestaurantID'
        df_poi['Global_ID'] = df_poi.apply(lambda row: f"{ds_info['source_name']}_{str(row.get(id_col, row.name)).replace(':', '_')}", axis=1)
    
    df_poi['Source'] = ds_info['source_name']
    
    # 2. Lọc Bounding Box Đà Nẵng
    dn_min_lat, dn_max_lat = 15.90, 16.20
    dn_min_lon, dn_max_lon = 108.00, 108.35
    initial_len = len(df_poi)
    df_poi = df_poi[
        (df_poi['Lat'] >= dn_min_lat) & (df_poi['Lat'] <= dn_max_lat) &
        (df_poi['Lon'] >= dn_min_lon) & (df_poi['Lon'] <= dn_max_lon)
    ].reset_index(drop=True)
    print(f"🧹 Đã lọc ngoại lai, giữ lại {len(df_poi)}/{initial_len} điểm POI hợp lệ.")
    
    # 3. Tính toán Bounding Box để PDS
    min_lat, max_lat = df_poi['Lat'].min(), df_poi['Lat'].max()
    min_lon, max_lon = df_poi['Lon'].min(), df_poi['Lon'].max()
    r_norm = ((min_dist_m / 111000.0) + (min_dist_m / 105000.0)) / 2.0
    
    print(f"🔄 Chạy PDS sinh vùng trống (Void) cho {ds_info['source_name']}...")
    raw_points = bridson_pds(max_lon - min_lon, max_lat - min_lat, r_norm)
    pds_lons = raw_points[:, 0] + min_lon
    pds_lats = raw_points[:, 1] + min_lat
    
    # 4. Lọc Void không bị đè lên POI
    poi_points = MultiPoint([Point(lon, lat) for lon, lat in zip(df_poi['Lon'], df_poi['Lat'])])
    filtered_points = []
    exclusion_radius = 100 / 111000.0 
    
    for lon, lat in tqdm(zip(pds_lons, pds_lats), total=len(pds_lons)):
        p = Point(lon, lat)
        nearest_poi = nearest_points(p, poi_points)[1]
        if p.distance(nearest_poi) > exclusion_radius:
            filtered_points.append([lat, lon])
            
    df_neg = pd.DataFrame(filtered_points, columns=['Lat', 'Lon'])
    max_voids = len(df_poi) * 5 
    if len(df_neg) > max_voids:
        df_neg = df_neg.sample(n=max_voids, random_state=42).reset_index(drop=True)
        
    df_neg['Global_ID'] = [f"void_{ds_info['source_name']}_{i}" for i in range(len(df_neg))]
    df_neg['Source'] = f"urban_voids_{ds_info['source_name']}"
    
    # 5. Lưu kết quả RIÊNG BIỆT cho từng Domain
    voids_path = os.path.join(PROJECT_ROOT, f"dataset/sampling/urban_voids_{ds_info['source_name']}.csv")
    master_path = os.path.join(PROJECT_ROOT, f"dataset/processed/master_nodes_{ds_info['source_name']}.csv")
    map_path = os.path.join(PROJECT_ROOT, f"reports/urban_voids_map_{ds_info['source_name']}.html")
    
    os.makedirs(os.path.dirname(voids_path), exist_ok=True)
    df_neg.to_csv(voids_path, index=False)
    
    df_master = pd.concat([df_poi, df_neg], ignore_index=True)
    df_master.to_csv(master_path, index=False)
    
    print(f"✅ Đã tạo Master Node cho {ds_info['source_name']}: {master_path} (Tổng {len(df_master)} dòng)")
    
    # 6. Vẽ bản đồ riêng
    m = folium.Map(location=[df_poi['Lat'].mean(), df_poi['Lon'].mean()], zoom_start=13)
    for _, row in df_poi.iterrows():
        folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=2, color='blue', fill=True, popup=f"POI").add_to(m)
    for _, row in df_neg.iterrows():
        folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=3, color='red', fill=True, popup="Void").add_to(m)
    
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    m.save(map_path)

if __name__ == "__main__":
    for ds in DATASETS:
        process_single_domain(ds, min_dist_m=200)