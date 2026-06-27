import streamlit as st
import os
import pandas as pd
import time
from datetime import datetime
from PIL import Image
import torch
import numpy as np

from reid_core import ReIDExtractor, search_topk, compute_cosine_similarity

# Page Config
st.set_page_config(page_title="Person ReID Web Studio", page_icon="👤", layout="wide")

# Determine Gallery Directory dynamically
PRIMARY_GALLERY = os.path.join("data", "reid_mini", "gallery")
DEFAULT_GALLERY = "gallery"

if os.path.exists(PRIMARY_GALLERY) and len(os.listdir(PRIMARY_GALLERY)) > 0:
    GALLERY_DIR = PRIMARY_GALLERY
elif os.path.exists(DEFAULT_GALLERY):
    GALLERY_DIR = DEFAULT_GALLERY
else:
    GALLERY_DIR = DEFAULT_GALLERY
    os.makedirs(GALLERY_DIR, exist_ok=True)

RESULTS_FILE = "reid_log.xlsx"

# Sidebar Controls
st.sidebar.header("⚙️ Model & Search Settings")
model_choice = st.sidebar.selectbox("Select ReID Model", ["osnet_x0_25", "osnet_x0_5", "osnet_x1_0"], index=0)
top_k = st.sidebar.slider("Top K Matches", min_value=1, max_value=10, value=5)
threshold = st.sidebar.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.40, step=0.05)

# Load ReID Extractor Model
@st.cache_resource
def get_extractor(model_name: str):
    return ReIDExtractor(model_name=model_name, use_gpu=True)

extractor = get_extractor(model_choice)

# Load Gallery Database Features
@st.cache_data
def get_gallery_features(gallery_dir: str, model_name: str):
    if not os.path.exists(gallery_dir):
        return [], [], np.empty((0, 512))
    
    gallery_files = [f for f in sorted(os.listdir(gallery_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    gallery_paths = [os.path.join(gallery_dir, f) for f in gallery_files]
    
    feats = []
    for path in gallery_paths:
        img = cv2_img = Image.open(path).convert("RGB")
        # Extract using core extractor
        img_np = np.array(img)[:, :, ::-1]  # RGB to BGR for extractor
        feat = extractor.extract_feature(img_np)
        feats.append(feat)
        
    feats = np.array(feats) if feats else np.empty((0, 512))
    return gallery_files, gallery_paths, feats

gallery_files, gallery_paths, gallery_feats = get_gallery_features(GALLERY_DIR, model_choice)

# UI Main Content
st.title("👤 Person Re-Identification Studio")
st.markdown("Upload a query image of a person to perform automated visual search across camera gallery feeds.")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📤 Query Person Input")
    uploaded = st.file_uploader("Upload Query Image", type=["jpg", "png", "jpeg", "bmp"])
    
    if uploaded:
        query_image = Image.open(uploaded).convert("RGB")
        st.image(query_image, caption="Uploaded Query Image", use_column_width=True)

with col_right:
    st.subheader("🔍 Match Results & Analysis")
    if uploaded:
        if len(gallery_paths) == 0:
            st.warning(f"No images found in gallery directory (`{GALLERY_DIR}`). Please add images to search.")
        else:
            # Extract query feature
            query_np = np.array(query_image)[:, :, ::-1]  # RGB to BGR
            start_time = time.time()
            q_feat = extractor.extract_feature(query_np)
            
            # Compute similarities
            sims = np.dot(gallery_feats, q_feat.T).reshape(-1)
            
            # Sort and filter
            sorted_indices = np.argsort(-sims)
            valid_matches = [(idx, sims[idx]) for idx in sorted_indices if sims[idx] >= threshold][:top_k]
            elapsed = time.time() - start_time
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Search Time", f"{elapsed*1000:.1f} ms")
            m2.metric("Total Matches", len(valid_matches))
            top_score = f"{valid_matches[0][1]*100:.1f}%" if valid_matches else "N/A"
            m3.metric("Top Confidence", top_score)
            
            st.divider()
            
            if not valid_matches:
                st.info("No matching individuals found above the selected similarity threshold.")
            else:
                grid_cols = st.columns(min(len(valid_matches), 4))
                for i, (g_idx, score) in enumerate(valid_matches):
                    col_target = grid_cols[i % len(grid_cols)]
                    with col_target:
                        st.image(gallery_paths[g_idx], use_column_width=True)
                        conf = score * 100
                        badge_color = "🟢" if conf >= 70 else "🟡" if conf >= 50 else "⚪"
                        st.caption(f"**Rank {i+1}** {badge_color}\nScore: `{conf:.1f}%`\nFile: `{gallery_files[g_idx]}`")
                
                # Log best match to excel
                top_g_idx, top_sim = valid_matches[0]
                matched_pid = gallery_files[top_g_idx].split("_")[0]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                row = {
                    "Query File": uploaded.name,
                    "Matched Person ID": matched_pid,
                    "Similarity Score": float(top_sim),
                    "Timestamp": timestamp,
                    "Model": model_choice
                }
                
                if not os.path.exists(RESULTS_FILE):
                    df = pd.DataFrame([row])
                else:
                    try:
                        df = pd.read_excel(RESULTS_FILE)
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    except Exception:
                        df = pd.DataFrame([row])
                
                df.to_excel(RESULTS_FILE, index=False)
                st.success("✅ Match automatically recorded into system log!")

st.sidebar.divider()
if os.path.exists(RESULTS_FILE):
    st.sidebar.subheader("📊 System Audit Log")
    log_df = pd.read_excel(RESULTS_FILE)
    st.sidebar.dataframe(log_df.tail(5))
