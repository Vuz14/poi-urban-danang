import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os
from PIL import Image

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Đà Nẵng Urban AI",
    page_icon="🗺️",
    layout="wide"
)

# ==========================================
# HÀM TẢI DỮ LIỆU (Cache để web chạy nhanh)
# ==========================================
@st.cache_data
def load_data():
    file_path = "dataset/processed/poi_processed_data.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data()

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🗺️ Hệ thống AI Phân tích Không gian Đô thị Đà Nẵng")
st.markdown("**Đồ án Chuyên ngành 1** | Xây dựng Bản đồ tương tác và Khai phá dữ liệu bằng Deep Learning (CLIP & Spatial Attention).")

if df is None:
    st.error("❌ Không tìm thấy dữ liệu! Vui lòng kiểm tra lại file `dataset/processed/poi_processed_data.csv`.")
    st.stop()

# --- SIDEBAR (THANH ĐIỀU CHUYỂN BÊN TRÁI) ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Da_Nang_logo.svg/1200px-Da_Nang_logo.svg.png", width=100)
st.sidebar.header("Lọc Dữ Liệu")

# Lọc theo Quận
all_districts = df['District'].dropna().unique().tolist()
selected_districts = st.sidebar.multiselect("Chọn Quận/Huyện:", options=all_districts, default=all_districts)

# Lọc theo Danh mục
all_categories = df['Category'].dropna().unique().tolist()
selected_categories = st.sidebar.multiselect("Chọn Danh mục:", options=all_categories, default=all_categories)

# Áp dụng bộ lọc
filtered_df = df[(df['District'].isin(selected_districts)) & (df['Category'].isin(selected_categories))]

# --- HIỂN THỊ THÔNG SỐ (METRICS) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tổng số Địa điểm (POI)", f"{len(filtered_df)} quán")
col2.metric("Số lượng Quận/Huyện", len(selected_districts))
col3.metric("Số lượng Danh mục", len(selected_categories))
col4.metric("Điểm đánh giá trung bình", f"{filtered_df['Overall Rating'].mean():.1f} / 10")

st.markdown("---")

# --- CHIA TAB (THẺ) ĐỂ HIỂN THỊ ---
tab1, tab2, tab3 = st.tabs(["📍 Bản đồ Không gian Đô thị", "📊 Bảng Dữ liệu Chi tiết", "🧠 Kết quả Trí tuệ Nhân tạo (AI)"])

# TAB 1: BẢN ĐỒ TƯƠNG TÁC
with tab1:
    st.subheader(f"Bản đồ phân bố {len(filtered_df)} địa điểm tại Đà Nẵng")
    
    # Tạo bản đồ nền
    m = folium.Map(location=[16.0544, 108.2022], zoom_start=13, tiles='CartoDB positron')
    
    # Định nghĩa màu sắc cho các danh mục phổ biến
    color_map = {
        'Quán ăn': 'blue', 
        'Café/Dessert': 'orange', 
        'Nhà hàng': 'red', 
        'Ăn vặt/vỉa hè': 'green'
    }
    
    # Chấm các điểm lên bản đồ
    for idx, row in filtered_df.iterrows():
        cat = row['Category']
        color = color_map.get(cat, 'gray') # Nếu không có trong danh sách thì màu xám
        
        # Nội dung khi click vào điểm (Popup)
        html_popup = f"""
        <div style="width:200px">
            <h4>{row['Restaurant Name']}</h4>
            <b>Danh mục:</b> {cat}<br>
            <b>Quận:</b> {row['District']}<br>
            <b>Đánh giá:</b> {row['Overall Rating']}/10 ⭐<br>
            <b>Mức giá:</b> {row['Price']}
        </div>
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=6,
            popup=folium.Popup(html_popup, max_width=250),
            tooltip=row['Restaurant Name'], # Hiển thị tên khi di chuột qua
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)
        
    # Render bản đồ lên Web
    st_folium(m, width="100%", height=600)

# TAB 2: BẢNG DỮ LIỆU
with tab2:
    st.subheader("Dữ liệu Gốc đã Tiền xử lý (Đã ẩn các cột hệ thống)")
    display_cols = ['RestaurantID', 'Restaurant Name', 'District', 'Category', 'Price', 'Overall Rating', 'Total_Reviews_Scraped']
    st.dataframe(filtered_df[display_cols], use_container_width=True, height=500)

# TAB 3: KẾT QUẢ AI (SHOW ẢNH T-SNE)
with tab3:
    st.subheader("Trực quan hóa Phân cụm Đa phương thức (Multimodal t-SNE)")
    st.markdown("""
    Biểu đồ dưới đây được sinh ra từ **Mô hình AI Đa phương thức (CLIP kết hợp Attention Không gian)**. 
    Mỗi dấu chấm là một quán ăn. AI đã tự động đọc Hình ảnh, Text Review và Tọa độ để "kéo" các quán có tính chất giống nhau lại gần nhau mà không cần con người can thiệp.
    """)
    
    tsne_path = "reports/figures/tsne_category_clusters.png"
    if os.path.exists(tsne_path):
        img = Image.open(tsne_path)
        st.image(img, caption="t-SNE Embeddings Visualization", use_container_width=True)
    else:
        st.info("💡 Chưa tìm thấy ảnh t-SNE. Vui lòng chạy lệnh `python main.py` ở Terminal để AI huấn luyện và vẽ ảnh này nhé!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Đồ án Chuyên ngành 1 - Khoa Khoa học Máy tính - ĐH CNTT & TT Việt - Hàn (VKU)</p>", unsafe_allow_html=True)