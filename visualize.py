import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import os

# Cấu hình font chữ cho biểu đồ
plt.rcParams['font.family'] = 'sans-serif'

def plot_poi_distribution(df, save_dir):
    """Vẽ biểu đồ cột thống kê số lượng quán theo Danh mục và Quận"""
    plt.figure(figsize=(12, 6))
    
    # 1. Biểu đồ theo Quận
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, y='District', order=df['District'].value_counts().index, palette='viridis')
    plt.title('Số lượng Quán ăn theo từng Quận')
    plt.xlabel('Số lượng')
    plt.ylabel('Quận')

    # 2. Biểu đồ theo Danh mục
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, y='Category', order=df['Category'].value_counts().index, palette='magma')
    plt.title('Số lượng POI theo Danh mục')
    plt.xlabel('Số lượng')
    plt.ylabel('Danh mục')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'poi_distribution.png'), dpi=300)
    print("✅ Đã lưu biểu đồ phân bố: poi_distribution.png")

def create_interactive_map(df, save_dir):
    """Vẽ bản đồ Đà Nẵng và chấm các POI lên đó"""
    # Lấy tọa độ trung tâm Đà Nẵng
    danang_coords = [16.0544, 108.2022]
    
    # Tạo bản đồ nền
    m = folium.Map(location=danang_coords, zoom_start=13, tiles='CartoDB positron')
    
    # Bảng màu cho các danh mục
    colors = {'Quán ăn': 'blue', 'Café/Dessert': 'brown', 'Nhà hàng': 'red', 'Ăn vặt/vỉa hè': 'orange'}
    
    # Thêm Marker cho từng quán
    for idx, row in df.iterrows():
        cat = row['Category']
        color = colors.get(cat, 'gray') # Nếu không có trong bảng màu thì cho màu xám
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=3,
            popup=f"<b>{row['Restaurant Name']}</b><br>{cat}",
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Thêm Heatmap (Bản đồ nhiệt)
    heat_data = [[row['Lat'], row['Lon']] for index, row in df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10).add_to(m)

    # Lưu thành file HTML
    map_path = os.path.join(save_dir, 'danang_poi_map.html')
    m.save(map_path)
    print(f"✅ Đã lưu bản đồ tương tác: {map_path} (Hãy mở bằng trình duyệt Chrome để xem)")

def visualize_all():
    print("Đang tạo các biểu đồ trực quan hóa cho Báo cáo...")
    
    # Tạo thư mục chứa hình ảnh báo cáo
    report_dir = "reports/figures"
    os.makedirs(report_dir, exist_ok=True)
    
    # Đọc dữ liệu
    df = pd.read_csv("dataset/processed/poi_processed_data.csv")
    
    # Gọi các hàm vẽ
    plot_poi_distribution(df, report_dir)
    create_interactive_map(df, report_dir)

if __name__ == "__main__":
    visualize_all()