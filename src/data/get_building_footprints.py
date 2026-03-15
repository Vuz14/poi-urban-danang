import os
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt

def download_building_footprints():
    print("🌍 Đang khởi động hệ thống lấy Đa giác Tòa nhà từ OpenStreetMap...")
    
    # 1. Đọc dữ liệu tọa độ 553 POI
    csv_path = "dataset/processed/poi_processed_data.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.read_csv("poi_processed_data.csv") # Dự phòng nếu chạy sai thư mục
        
    # 2. Tạo thư mục chứa ảnh
    save_dir = "dataset/processed/building_footprints"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📍 Tổng số POI cần lấy ảnh tòa nhà: {len(df)}")
    
    # Cấu hình osmnx không in ra log dư thừa
    ox.settings.log_console = False

    success_count = 0
    fail_count = 0

    # 3. Quét từng quán ăn
    for idx, row in df.iterrows():
        poi_id = row['RestaurantID']
        lat, lon = row['Lat'], row['Lon']
        
        # Đường dẫn lưu ảnh
        img_path = os.path.join(save_dir, f"{poi_id}.png")
        if os.path.exists(img_path):
            continue # Nếu đã tải rồi thì bỏ qua
            
        try:
            # Lấy đa giác tòa nhà trong bán kính 50 mét xung quanh quán ăn
            tags = {'building': True}
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=50)
            
            if gdf.empty:
                raise ValueError("Không tìm thấy tòa nhà")

            # 4. Vẽ đa giác thành ảnh Trắng Đen chuẩn 224x224
            # figsize 2.24 * 100 dpi = 224 pixels
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            
            # Nền màu đen (để AI hiểu là không gian trống)
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Đa giác tòa nhà màu trắng (để AI bắt đặc trưng góc cạnh)
            gdf.plot(ax=ax, color='white', edgecolor='none')
            
            # Tắt hiển thị trục tọa độ
            ax.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            
            # Lưu ảnh
            plt.savefig(img_path, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
            plt.close()
            success_count += 1
            
            print(f"✅ Đã tải thành công ảnh đa giác cho POI {poi_id}")
            
        except Exception as e:
            # Nếu khu vực đó quá hẻo lánh, không có dữ liệu tòa nhà trên bản đồ
            # Chúng ta vẽ một bức ảnh đen hoàn toàn (Ảnh rỗng) để ResNet không bị lỗi
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            fig.patch.set_facecolor('black')
            ax.axis('off')
            plt.savefig(img_path, facecolor='black', bbox_inches='tight', pad_inches=0)
            plt.close()
            fail_count += 1
            print(f"⚠️ POI {poi_id}: Vùng trống (Lưu ảnh đen rỗng)")

    print("-" * 30)
    print(f"🎉 HOÀN TẤT THU THẬP! Thành công: {success_count} | Vùng trống: {fail_count}")
    print(f"📁 Ảnh đã được lưu tại: {save_dir}/")

if __name__ == "__main__":
    download_building_footprints()