import osmnx as ox
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import warnings

# Tắt cảnh báo DeprecationWarning của Geopandas
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unary_union.*")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_danang_environment():
    buildings_cache = os.path.join(PROJECT_ROOT, "dataset", "danang_buildings_cache.gpkg")
    boundary_cache = os.path.join(PROJECT_ROOT, "dataset", "danang_boundary_cache.gpkg")
    
    if os.path.exists(boundary_cache):
        print(f"📦 Đang đọc Ranh giới đất liền từ Local...")
        gdf_boundary = gpd.read_file(boundary_cache)
    else:
        print("🌊 Đang tải Ranh giới đất liền Đà Nẵng...")
        gdf_boundary = ox.geocode_to_gdf("Da Nang, Vietnam")
        os.makedirs(os.path.dirname(boundary_cache), exist_ok=True)
        gdf_boundary.to_file(boundary_cache, driver="GPKG")

    if os.path.exists(buildings_cache):
        print(f"📦 Đang đọc dữ liệu Tòa nhà từ Local...")
        gdf_buildings = gpd.read_file(buildings_cache)
    else:
        print("🏙️ Đang tải bản đồ Tòa nhà toàn thành phố Đà Nẵng...")
        gdf_buildings = ox.features_from_place("Da Nang, Vietnam", tags={'building': True})
        gdf_buildings = gdf_buildings[gdf_buildings.geometry.notnull()][['geometry']]
        gdf_buildings.to_file(buildings_cache, driver="GPKG")
        
    return gdf_buildings, gdf_boundary

def main():
    plt.ioff()
    print("🛡️ Cơ chế: CHỈ LỌC ĐIỂM RỚT BIỂN - Giữ lại các khu vực trống/chưa có map.")

    gdf_buildings, gdf_boundary = get_danang_environment()
    
    print("🗺️ Đang chuẩn hóa Hệ tọa độ về dạng Mét (Metric UTM CRS)...")
    gdf_buildings_proj = ox.projection.project_gdf(gdf_buildings)
    gdf_boundary_proj = ox.projection.project_gdf(gdf_boundary)
    
    # Cập nhật hàm union_all() thay cho unary_union để tránh cảnh báo
    try:
        land_polygon = gdf_boundary_proj.geometry.union_all()
    except AttributeError:
        land_polygon = gdf_boundary_proj.geometry.unary_union
    
    print("🔎 Đang xây dựng R-tree Spatial Index...")
    sidx = gdf_buildings_proj.sindex

    print("="*50)
    domains = ["google_maps", "foody"]
    
    SAFE_BACKGROUND = '#111111' 

    for domain in domains:
        master_csv = os.path.join(PROJECT_ROOT, f"dataset/processed/master_nodes_{domain}.csv")
        out_dir = os.path.join(PROJECT_ROOT, f"dataset/building_images_{domain}")
        
        if not os.path.exists(master_csv):
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(master_csv)
        print(f"\n🚀 Domain: {domain.upper()} (Tổng {len(df)} điểm gốc)")
        
        valid_rows = [] 
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            lat, lon, global_id = row.get("Lat"), row.get("Lon"), row.get("Global_ID")
            
            if pd.isna(lat) or pd.isna(lon) or pd.isna(global_id):
                continue
            
            fname = os.path.join(out_dir, f"{global_id}.png")
            
            try:
                pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                pt_proj = pt.to_crs(gdf_buildings_proj.crs)
                point_geom = pt_proj.iloc[0]
                
                # BƯỚC LỌC DUY NHẤT: Bỏ những quán nằm ngoài đất liền (rớt biển)
                if not land_polygon.contains(point_geom):
                    if os.path.exists(fname):
                        os.remove(fname)
                    continue # Bỏ qua điểm này, không đưa vào valid_rows
                
                # Nếu nằm trên đất liền, tạo ảnh tòa nhà (hoặc ảnh xám trống)
                if not os.path.exists(fname):
                    # Tăng buffer lên 150m để quét rộng hơn, dễ trúng nhà hơn
                    buffer_poly = point_geom.buffer(150) 
                    possible_matches_index = list(sidx.intersection(buffer_poly.bounds))
                    possible_matches = gdf_buildings_proj.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(buffer_poly)]
                    
                    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
                    fig.patch.set_facecolor(SAFE_BACKGROUND)
                    ax.set_facecolor(SAFE_BACKGROUND)
                    
                    if not precise_matches.empty:
                        precise_matches.plot(ax=ax, facecolor='white', edgecolor='none')
                    
                    minx, miny, maxx, maxy = buffer_poly.bounds
                    ax.set_xlim(minx, maxx)
                    ax.set_ylim(miny, maxy)
                    ax.set_axis_off()
                    
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
                    plt.savefig(fname, facecolor=SAFE_BACKGROUND, pad_inches=0)
                    plt.close(fig)
                    
                    with Image.open(fname) as img:
                        img_resized = img.convert('RGB').resize((224, 224))
                    img_resized.save(fname)

                # Lưu POI vào danh sách hợp lệ
                valid_rows.append(row)

            except Exception:
                plt.close('all')
                continue

        # Lưu lại file CSV các tọa độ đã được làm sạch
        if valid_rows:
            df_clean = pd.DataFrame(valid_rows)
            clean_csv_path = os.path.join(PROJECT_ROOT, f"dataset/processed/master_nodes_{domain}_clean.csv")
            df_clean.to_csv(clean_csv_path, index=False)
            print(f"✅ Đã dọn dẹp {domain.upper()}! Giữ lại {len(df_clean)}/{len(df)} điểm hợp lệ.")
            print(f"📁 Đã lưu: {clean_csv_path}")

if __name__ == "__main__":
    main()