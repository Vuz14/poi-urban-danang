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


def plot_training_metrics(version=1, metrics_dir="reports/metrics", figure_dir="reports/figures"):
    """Vẽ Loss + Accuracy/Precision/Recall/F1 từ CSV huấn luyện."""
    csv_path = os.path.join(metrics_dir, f"training_loss_v{version}.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Không tìm thấy file metrics: {csv_path}")
        return

    os.makedirs(figure_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if {'epoch', 'train_loss', 'test_loss'}.issubset(df.columns):
        plt.figure(figsize=(9, 5))
        plt.plot(df['epoch'], df['train_loss'], marker='o', linewidth=2.2, label='Train Loss')
        plt.plot(df['epoch'], df['test_loss'], marker='s', linewidth=2.2, linestyle='--', label='Test Loss')
        plt.title(f'Loss Curve - V{version}')
        plt.xlabel('Epoch')
        plt.ylabel('InfoNCE Loss')
        plt.xticks(df['epoch'])
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_loss = os.path.join(figure_dir, f'loss_curve_v{version}.png')
        plt.savefig(out_loss, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Đã lưu biểu đồ loss: {out_loss}")

    metric_cols = [
        ('Loss', 'train_loss', 'test_loss'),
        ('AUC', 'train_auc', 'test_auc'),
        ('Accuracy', 'train_acc', 'test_acc'),
        ('F1-score', 'train_f1', 'test_f1'),
        ('Precision', 'train_precision', 'test_precision'),
        ('Recall', 'train_recall', 'test_recall'),
        ('Specificity', 'train_spec', 'test_spec'),
    ]

    has_all_metric_cols = all(train_col in df.columns and test_col in df.columns for _, train_col, test_col in metric_cols)
    if not has_all_metric_cols:
        print("⚠️ File CSV chưa có đủ cột Loss/AUC/Accuracy/F1/Precision/Recall/Specificity để vẽ metrics.")
        return

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()

    for ax, (metric_name, train_col, test_col) in zip(axes, metric_cols):
        ax.plot(df['epoch'], df[train_col], marker='o', linewidth=2.0, label=f'Train {metric_name}')
        ax.plot(df['epoch'], df[test_col], marker='s', linewidth=2.0, linestyle='--', label=f'Test {metric_name}')
        ax.set_title(metric_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_xticks(df['epoch'])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(metric_cols):]:
        ax.axis('off')

    fig.suptitle(f'All Metrics Curve - V{version}', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_metrics = os.path.join(figure_dir, f'metrics_curve_v{version}.png')
    fig.savefig(out_metrics, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Đã lưu biểu đồ metrics: {out_metrics}")

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
    plot_training_metrics(version=1, metrics_dir="reports/metrics", figure_dir=report_dir)

if __name__ == "__main__":
    visualize_all()