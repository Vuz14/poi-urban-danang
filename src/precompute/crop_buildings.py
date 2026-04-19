import osmnx as ox
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_danang_environment():
    """
    Tải toàn bộ Tòa nhà VÀ Ranh giới đất liền (Land Boundary) của Đà Nẵng.
    """
    buildings_cache = os.path.join(PROJECT_ROOT, "dataset", "danang_buildings_cache.gpkg")
    boundary_cache = os.path.join(PROJECT_ROOT, "dataset", "danang_boundary_cache.gpkg")
    
    # 1. Tải ranh giới hành chính (Chống rớt xuống biển)
    if os.path.exists(boundary_cache):
        print(f"📦 Đang đọc Ranh giới đất liền từ Local: {boundary_cache}")
        gdf_boundary = gpd.read_file(boundary_cache)
    else:
        print("🌊 Đang tải Ranh giới đất liền Đà Nẵng (Loại bỏ biển/đảo xa)...")
        # geocode_to_gdf lấy chính xác đa giác bao quanh đất liền Đà Nẵng
        gdf_boundary = ox.geocode_to_gdf("Da Nang, Vietnam")
        os.makedirs(os.path.dirname(boundary_cache), exist_ok=True)
        gdf_boundary.to_file(boundary_cache, driver="GPKG")

    # 2. Tải Tòa nhà
    if os.path.exists(buildings_cache):
        print(f"📦 Đang đọc dữ liệu Tòa nhà từ Local: {buildings_cache}")
        gdf_buildings = gpd.read_file(buildings_cache)
    else:
        print("🏙️ Đang tải bản đồ Tòa nhà toàn thành phố Đà Nẵng...")
        gdf_buildings = ox.features_from_place("Da Nang, Vietnam", tags={'building': True})
        gdf_buildings = gdf_buildings[gdf_buildings.geometry.notnull()][['geometry']]
        gdf_buildings.to_file(buildings_cache, driver="GPKG")
        
    return gdf_buildings, gdf_boundary

def main():
    plt.ioff()
    print("🛡️ Cơ chế an toàn kích hoạt: Xử lý rớt biển, lỗi GPS & Dead Gradients.")

    # 1. Khởi tạo Không gian bản đồ
    gdf_buildings, gdf_boundary = get_danang_environment()
    
    print("🗺️ Đang chuẩn hóa Hệ tọa độ về dạng Mét (Metric UTM CRS)...")
    gdf_buildings_proj = ox.projection.project_gdf(gdf_buildings)
    gdf_boundary_proj = ox.projection.project_gdf(gdf_boundary)
    
    # Lấy đa giác đất liền duy nhất
    land_polygon = gdf_boundary_proj.geometry.unary_union
    
    print("🔎 Đang xây dựng R-tree Spatial Index (Truy vấn hình học siêu tốc)...")
    sidx = gdf_buildings_proj.sindex

    print("="*50)
    domains = ["google_maps", "foody"]
    
    # [GIẢI PHÁP 3]: Background xám cực tối chống Dead Gradient cho ResNet
    # Mã màu #111111 (RGB: 17, 17, 17) giúp Model hiểu là "Đất trống hợp lệ" thay vì ảnh Lỗi.
    SAFE_BACKGROUND = '#111111' 
    ERROR_BACKGROUND = '#000000' # Đen tuyệt đối cho Lỗi/Rớt xuống biển

    for domain in domains:
        master_csv = os.path.join(PROJECT_ROOT, f"dataset/processed/master_nodes_{domain}.csv")
        out_dir = os.path.join(PROJECT_ROOT, f"dataset/building_images_{domain}")
        
        if not os.path.exists(master_csv):
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(master_csv)
        print(f"\n🚀 Domain: {domain.upper()} (Tổng {len(df)} điểm)")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            lat, lon, global_id = row.get("Lat"), row.get("Lon"), row.get("Global_ID")
            
            if pd.isna(lat) or pd.isna(lon) or pd.isna(global_id):
                continue
            
            fname = os.path.join(out_dir, f"{global_id}.png")
            
            if not os.path.exists(fname):
                try:
                    # Chuyển đổi điểm vào hệ UTM
                    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                    pt_proj = pt.to_crs(gdf_buildings_proj.crs)
                    point_geom = pt_proj.iloc[0]
                    
                    # [GIẢI PHÁP 1]: KIỂM TRA ĐIỂM CÓ NẰM TRÊN ĐẤT LIỀN KHÔNG?
                    # Nếu điểm rớt xuống biển hoặc ra ngoài Đà Nẵng -> Ảnh đen tuyệt đối (Error)
                    if not land_polygon.contains(point_geom):
                        Image.new('RGB', (224, 224), (0, 0, 0)).save(fname)
                        continue

                    # Buffer mổ ra chu vi hình tròn 100 mét quanh POI
                    buffer_poly = point_geom.buffer(100)
                    
                    possible_matches_index = list(sidx.intersection(buffer_poly.bounds))
                    possible_matches = gdf_buildings_proj.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(buffer_poly)]
                    
                    # Canvas plot
                    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
                    
                    # Set nền an toàn (Safe Background)
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
                    
                    # Resize về chuẩn ResNet
                    Image.open(fname).convert('RGB').resize((224, 224)).save(fname)
                    
                except Exception:
                    # [GIẢI PHÁP 4]: Lỗi tính toán/Toán học -> Đen tuyệt đối
                    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(fname)
                    plt.close()

if __name__ == "__main__":
    main()