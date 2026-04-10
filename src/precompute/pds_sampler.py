import numpy as np
import pandas as pd
import folium
import os
import math
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DATASETS = [
    {
        "name": "Foody POI",
        "csv_path": os.path.join(PROJECT_ROOT, "dataset/processed/poi_processed_data.csv"),
        "lat_col": "Lat",
        "lon_col": "Lon",
        "id_col": "RestaurantID",
        "source": "foody"
    },
    {
        "name": "Google Maps POI",
        "csv_path": os.path.join(PROJECT_ROOT, "dataset/processed/poi_data_ggmap.csv"),
        "lat_col": "lat",
        "lon_col": "lng",
        "id_col": "place_id",
        "source": "google_maps"
    }
]

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

def load_and_merge_datasets():
    combined_records = []
    print("📍 Đang đọc và hợp nhất các datasets POI...")
    
    for ds in DATASETS:
        if not os.path.exists(ds['csv_path']):
            print(f"⚠️ Không tìm thấy file: {ds['csv_path']}")
            continue
            
        df = pd.read_csv(ds['csv_path'])
        print(f"   - {ds['name']}: {len(df)} điểm.")
        for idx, row in df.iterrows():
            lat = row.get(ds['lat_col'])
            lon = row.get(ds['lon_col'])
            
            if pd.isna(lat) or pd.isna(lon):
                continue
                
            item_id = row.get(ds['id_col'], idx)
            clean_id = str(item_id).replace(':', '_')
            
            combined_records.append({
                'Global_ID': f"{ds['source']}_{clean_id}",
                'Lat': float(lat),
                'Lon': float(lon),
                'Source': ds['source']
            })
            
    df_combined = pd.DataFrame(combined_records)
    print(f"✅ Tổng cộng thu được {len(df_combined)} điểm dữ liệu thực tế.")
    return df_combined

def generate_urban_negatives(df_poi, output_csv, master_csv, min_dist_m=200):
    """
    Tạo các điểm 'Negative samples' bằng PDS tránh TẤT CẢ POI hiện có.
    """
    print(f"🧹 Đang dọn dẹp tọa độ rác trước khi lấy mẫu...")
    initial_len = len(df_poi)
    
    # Khóa cứng Bounding Box khu vực đô thị Đà Nẵng
    # (Loại bỏ các điểm có tọa độ 0,0 hoặc nằm ở tỉnh/quốc gia khác)
    dn_min_lat, dn_max_lat = 15.90, 16.20
    dn_min_lon, dn_max_lon = 108.00, 108.35
    
    df_poi = df_poi[
        (df_poi['Lat'] >= dn_min_lat) & (df_poi['Lat'] <= dn_max_lat) &
        (df_poi['Lon'] >= dn_min_lon) & (df_poi['Lon'] <= dn_max_lon)
    ].reset_index(drop=True)
    
    if len(df_poi) < initial_len:
        print(f"⚠️ Đã loại bỏ {initial_len - len(df_poi)} điểm POI có tọa độ rác (nằm ngoài Đà Nẵng).")
        
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
    
    print("🧹 Đang lọc các điểm nằm trong vùng đô thị hiện hữu (né chùm POI)...")
    poi_points = MultiPoint([Point(lon, lat) for lon, lat in zip(df_poi['Lon'], df_poi['Lat'])])
    
    filtered_points = []
    exclusion_radius = 100 / 111000.0 
    
    for lon, lat in tqdm(zip(pds_lons, pds_lats), total=len(pds_lons)):
        p = Point(lon, lat)
        nearest_poi = nearest_points(p, poi_points)[1]
        if p.distance(nearest_poi) > exclusion_radius:
            filtered_points.append([lat, lon])
            
    df_neg = pd.DataFrame(filtered_points, columns=['Lat', 'Lon'])
    
    # [FIX] Lấy số lượng điểm gấp 5 lần số POI thay vì 1:1
    max_voids = len(df_poi) * 5 
    
    if len(df_neg) > max_voids:
        print(f"\n⚠️ Phát hiện {len(df_neg)} điểm Voids!")
        print(f"👉 Đang lấy mẫu ngẫu nhiên giảm xuống còn {max_voids} điểm (Tỉ lệ 1:5 với POI)...")
        df_neg = df_neg.sample(n=max_voids, random_state=42).reset_index(drop=True)
    else:
        print(f"\n✅ Đã sinh được {len(df_neg)} điểm Voids hợp lệ.")
        
    df_neg['Global_ID'] = [f"void_{i}" for i in range(len(df_neg))]
    df_neg['Source'] = 'urban_voids'
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    # Lưu file urban_voids gốc
    df_neg.to_csv(output_csv, index=False)
    print(f"✅ Đã tạo {len(df_neg)} điểm vùng trống. Lưu tại: {output_csv}")
    
    # Nối thành siêu dải Master Nodes
    os.makedirs(os.path.dirname(master_csv), exist_ok=True)
    df_master = pd.concat([df_poi, df_neg], ignore_index=True)
    df_master.to_csv(master_csv, index=False)
    print(f"🌟 Đã xuất tệp tọa độ siêu vũ trụ: {master_csv} (Tổng {len(df_master)} dòng)")
    
    return df_neg, df_master

def visualize_pds_map(df_poi, df_neg, map_path):
    print("🗺️ Đang vẽ bản đồ kiểm tra không gian...")
    m = folium.Map(location=[df_poi['Lat'].mean(), df_poi['Lon'].mean()], zoom_start=13)
    
    for _, row in df_poi.iterrows():
        color = 'blue' if row['Source'] == 'foody' else 'green'
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.5,
            popup=f"POI: {row['Source']}"
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
    print(f"✅ Bản đồ HTML đã lưu tại: {map_path}")

if __name__ == "__main__":
    output_voids = os.path.join(PROJECT_ROOT, "dataset/sampling/urban_voids.csv")
    output_master = os.path.join(PROJECT_ROOT, "dataset/processed/master_nodes.csv")
    map_html = os.path.join(PROJECT_ROOT, "reports/urban_voids_map.html")
    
    df_poi_combined = load_and_merge_datasets()
    if len(df_poi_combined) > 0:
        # Sử dụng min_dist_m=55 để đảm bảo sinh ra đủ nhiều điểm trước khi bị cắt giảm (sample)
        df_negative, df_master = generate_urban_negatives(df_poi_combined, output_voids, output_master, min_dist_m=55)
        visualize_pds_map(df_poi_combined, df_negative, map_html)
    else:
        print("❌ Không có dữ liệu POI để rải điểm.")