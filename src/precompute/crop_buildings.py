import osmnx as ox
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# Lấy đường dẫn gốc của project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def get_danang_buildings():
    """Tải và lưu trữ offline (cache) toàn phần tòa nhà thuộc Đà Nẵng."""
    cache_file = os.path.join(PROJECT_ROOT, "dataset", "danang_buildings_cache.gpkg")
    if os.path.exists(cache_file):
        print(f"📦 Đang đọc dữ liệu tòa nhà từ bộ đệm Local: {cache_file}")
        gdf = gpd.read_file(cache_file)
    else:
        print("🌐 Đang tải bản đồ toàn thành phố Đà Nẵng từ mạng (Chỉ mất 1-2 phút duy nhất vòng đời)...")
        gdf = ox.features_from_place("Da Nang, Vietnam", tags={'building': True})
        
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[['geometry']]
        
        print(f"💾 Đã tải xong! Đang phân giải và lưu vào cache {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        gdf.to_file(cache_file, driver="GPKG")
        
    return gdf

def main():
    plt.ioff() # Tắt chế độ vẽ không tương tác
    
    print("✅ Cơ chế an toàn hoạt động: Không vẽ lại/xóa các ảnh đã tồn tại trước đây.")

    # 1. Khởi tạo Không gian bản đồ (Spatial Environment)
    gdf_buildings = get_danang_buildings()
    
    print("🗺️ Đang chuẩn hóa Hệ tọa độ về dạng Mét (Metric UTM CRS)...")
    gdf_buildings_proj = ox.projection.project_gdf(gdf_buildings)
    
    print("🔎 Đang xây dựng R-tree Spatial Index (Truy vấn hình học siêu tốc)...")
    sidx = gdf_buildings_proj.sindex

    print("="*50)
    
    # Thiết lập 2 Domain độc lập
    domains = ["google_maps", "foody"]
    
    for domain in domains:
        # File Master nodes chứa cả POI và Voids đã sinh ra từ bước PDS
        master_csv = os.path.join(PROJECT_ROOT, f"dataset/processed/master_nodes_{domain}.csv")
        # Thư mục chứa ảnh tách biệt cho từng domain
        out_dir = os.path.join(PROJECT_ROOT, f"dataset/building_images_{domain}")
        
        if not os.path.exists(master_csv):
            print(f"⚠️ Bỏ qua {domain}: Không tìm thấy file {master_csv}")
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(master_csv)
        print(f"\n🚀 Bắt đầu trích xuất ảnh Building cho Domain: {domain.upper()} (Tổng {len(df)} điểm)")
        print(f"📁 Lưu tại: {out_dir}")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            lat = row.get("Lat")
            lon = row.get("Lon")
            global_id = row.get("Global_ID")
            
            # Bỏ qua nếu dòng bị lỗi thiếu data
            if pd.isna(lat) or pd.isna(lon) or pd.isna(global_id):
                continue
            
            # Sử dụng Global_ID làm tên file (ví dụ: foody_123.png hoặc void_foody_45.png)
            fname = os.path.join(out_dir, f"{global_id}.png")
            
            # TRỌNG TÂM: Ảnh đã có thì Skip qua ngay lập tức -> Tiết kiệm hàng giờ chạy lại!
            if not os.path.exists(fname):
                try:
                    # Chuyển đổi điểm POI vào trong hệ UTM của bản đồ
                    pt = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
                    pt_proj = pt.to_crs(gdf_buildings_proj.crs)
                    
                    # Buffer mổ ra chu vi hình tròn 100 mét quanh POI
                    buffer_poly = pt_proj.iloc[0].buffer(100)
                    
                    # Lấy những Tòa nhà giao cắt nằm lọt trong hoặc cắt ngang Buffer
                    possible_matches_index = list(sidx.intersection(buffer_poly.bounds))
                    possible_matches = gdf_buildings_proj.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(buffer_poly)]
                    
                    # Canvas plot
                    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    
                    if not precise_matches.empty:
                        precise_matches.plot(ax=ax, facecolor='white', edgecolor='none')
                    
                    # Gán Camera khung hình cho ax bằng đúng kích thước của chu vi 100 mét
                    minx, miny, maxx, maxy = buffer_poly.bounds
                    ax.set_xlim(minx, maxx)
                    ax.set_ylim(miny, maxy)
                    ax.set_axis_off()
                    
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
                    plt.savefig(fname, facecolor='black', pad_inches=0)
                    plt.close(fig)
                    
                    # Resize về chuẩn ResNet input shape
                    Image.open(fname).convert('RGB').resize((224, 224)).save(fname)
                    
                except Exception:
                    # Lỗi hình học (vd điểm nằm ngoài biển vv) > fallback đen thui
                    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(fname)
                    plt.close()
                
    print("\n✅ Hoàn thành toàn bộ quá trình vẽ Building Footprints cho cả 2 Domains!")

if __name__ == "__main__":
    main()