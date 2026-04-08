import os
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt

def download_building_footprints(csv_path):
    print(f"🌍 Đang lấy Đa giác Tòa nhà cho: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {csv_path}")
        return
        
    save_dir = "dataset/processed/building_footprints"
    os.makedirs(save_dir, exist_ok=True)
    ox.settings.log_console = False
    success_count, fail_count = 0, 0

    for idx, row in df.iterrows():
        poi_id = row['RestaurantID']
        lat, lon = row['Lat'], row['Lon']
        img_path = os.path.join(save_dir, f"{poi_id}.png")
        
        if os.path.exists(img_path):
            continue 
            
        try:
            tags = {'building': True}
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=50)
            if gdf.empty:
                raise ValueError("Không tìm thấy tòa nhà")

            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            gdf.plot(ax=ax, color='white', edgecolor='none')
            ax.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0,0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(img_path, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
            plt.close()
            success_count += 1
        except Exception:
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            fig.patch.set_facecolor('black')
            ax.axis('off')
            plt.savefig(img_path, facecolor='black', bbox_inches='tight', pad_inches=0)
            plt.close()
            fail_count += 1

    print(f"🎉 HOÀN TẤT ({csv_path})! Thành công: {success_count} | Vùng trống: {fail_count}")

if __name__ == "__main__":
    datasets = [
        "dataset/processed/poi_processed_gmap.csv",
        "dataset/processed/poi_processed_foody.csv"
    ]
    for ds in datasets:
        download_building_footprints(ds)