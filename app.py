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
# --- SỬA LẠI DÒNG NÀY TRONG APP.PY ---
tab1, tab2, tab3, tab4 = st.tabs(["📍 Bản đồ Đô thị", "📊 Bảng Dữ liệu", "🧠 Phân cụm AI", "🎯 AI Gợi ý Mở Quán (Site Selection)"])
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

# ==========================================
# TAB 4: AI GỢI Ý ĐỊA ĐIỂM (SITE SELECTION)
# ==========================================
with tab4:
    st.subheader("🎯 Trợ lý AI Khảo sát và Gợi ý Địa điểm Kinh doanh")
    st.markdown("""
    Chức năng này sử dụng **Mô hình Ngôn ngữ & Thị giác (CLIP)** để phân tích ý tưởng kinh doanh của bạn. 
    Sau đó, nó tìm kiếm những khu vực tại Đà Nẵng có sự tương đồng cao nhất về mặt văn hóa, không gian và tệp khách hàng để gợi ý bạn đặt mặt bằng.
    """)
    
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        # 1. Khách hàng nhập ý tưởng (Text)
        user_concept = st.text_area(
            "📝 Nhập ý tưởng quán bạn muốn mở (Ví dụ: Quán cafe lãng mạn, yên tĩnh...):", 
            value="Quán nhậu vỉa hè, hải sản tươi sống, không gian mở, ồn ào náo nhiệt, giá bình dân",
            key="user_concept_input",
            height=130
        )
        
    with col_input2:
        # 2. Khách hàng tải ảnh lên (Image)
        uploaded_file = st.file_uploader("🖼️ Tải lên ảnh thiết kế/phong cách quán:", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Ảnh phong cách tham khảo", use_container_width=True)
    
    col_btn, _ = st.columns([1, 2])
    with col_btn:
        search_clicked = st.button("🔍 Phân tích & Gợi ý Vị trí")
        
    # --- PHẦN 1: NẾU BẤM NÚT -> CHẠY AI VÀ LƯU VÀO BỘ NHỚ ---
    if search_clicked:
        with st.spinner("🧠 AI đang phân tích Đa phương thức (Ảnh + Chữ) và quét toàn bộ bản đồ Đà Nẵng..."):
            try:
                import os
                import torch
                from PIL import Image
                from torchvision import transforms
                from src.encoder.multimodal import MultimodalEncoder
                import torch.nn.functional as F
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_path = os.path.abspath(os.path.join(os.getcwd(), "models_saved", "multimodal_best.pth"))
                
                if not os.path.exists(model_path):
                    st.error(f"❌ Khẩn cấp: Không tìm thấy file trọng số AI tại:\n`{model_path}`")
                    st.stop()
                
                # Nạp mô hình AI
                model = MultimodalEncoder().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                model.eval()
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                if uploaded_file is not None:
                    img_pil = Image.open(uploaded_file).convert("RGB")
                    concept_img_tensor = transform(img_pil).unsqueeze(0).to(device)
                else:
                    concept_img_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(device)

                with torch.no_grad():
                    concept_feature = model(images=concept_img_tensor, texts=[user_concept]) 
                
                all_texts = df['LLM_Input_Text'].tolist()
                all_poi_features = []
                batch_size = 32
                
                for i in range(0, len(all_texts), batch_size):
                    batch_texts = all_texts[i:i+batch_size]
                    dummy_imgs = torch.zeros((len(batch_texts), 3, 224, 224), dtype=torch.float32).to(device)
                    
                    with torch.no_grad():
                        feats = model(images=dummy_imgs, texts=batch_texts)
                        all_poi_features.append(feats)
                
                all_poi_features = torch.cat(all_poi_features, dim=0)
                similarities = F.cosine_similarity(concept_feature, all_poi_features, dim=-1)
                
                # BÍ KÍP: Lưu Top 5 kết quả vào Session State (Bộ nhớ tạm của Web)
                st.session_state['ai_top_idx'] = torch.topk(similarities, 5).indices.cpu().numpy()
                st.session_state['ai_top_scores'] = torch.topk(similarities, 5).values.cpu().numpy()
                
            except Exception as e:
                st.error(f"⚠️ Có lỗi xảy ra trong quá trình tính toán AI: {e}")

    # --- PHẦN 2: LẤY TỪ BỘ NHỚ RA ĐỂ VẼ BẢN ĐỒ (Giúp bản đồ không bao giờ bị mất) ---
    if 'ai_top_idx' in st.session_state:
        st.success("✅ Đã tìm thấy các khu vực tiềm năng nhất cho ý tưởng của bạn!")
        
        top_5_idx = st.session_state['ai_top_idx']
        top_5_scores = st.session_state['ai_top_scores']
        
        m_recommend = folium.Map(location=[16.0544, 108.2022], zoom_start=13, tiles='CartoDB positron')
        
        for rank, (idx, score) in enumerate(zip(top_5_idx, top_5_scores)):
            poi = df.iloc[idx]
            
            st.markdown(f"**Top {rank+1}: Khu vực gần `{poi['Restaurant Name']}` (Độ phù hợp: {score*100:.1f}%)**")
            st.write(f"- 📍 Vị trí: {poi['District']}")
            st.write(f"- 🔎 Đặc trưng khu vực: {poi['LLM_Input_Text'][:150]}...")
            
            folium.Marker(
                location=[poi['Lat'], poi['Lon']],
                popup=f"Vị trí đề xuất Top {rank+1}<br>Độ phù hợp: {score*100:.1f}%",
                icon=folium.Icon(color='red', icon='star')
            ).add_to(m_recommend)
            
            folium.Circle(
                location=[poi['Lat'], poi['Lon']],
                radius=500,
                color='crimson',
                fill=True,
                fill_color='crimson',
                fill_opacity=0.2
            ).add_to(m_recommend)

        # Trực quan hóa bản đồ
        st_folium(m_recommend, width="100%", height=500)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Đồ án Chuyên ngành 1 - Khoa Khoa học Máy tính - ĐH CNTT & TT Việt - Hàn (VKU)</p>", unsafe_allow_html=True)