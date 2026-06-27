import os
import cv2
import numpy as np
from reid_core import ReIDExtractor, load_feature_cache, search_topk

def match_query(query_img_path: str, top_k: int = 3):
    if not os.path.exists(query_img_path):
        raise FileNotFoundError(f"Query image not found: {query_img_path}")
        
    cache_path = os.path.join("outputs", "gallery_features.pkl")
    if not os.path.exists(cache_path):
        # Fallback to npy
        cache_path_npy = os.path.join("outputs", "gallery_features.npy")
        if os.path.exists(cache_path_npy):
            gallery = np.load(cache_path_npy, allow_pickle=True).item()
        else:
            raise FileNotFoundError("Gallery features index not found. Run extract_features.py or build index in app.py first.")
    else:
        gallery = load_feature_cache(cache_path)

    extractor = ReIDExtractor(model_name="osnet_x0_25", use_gpu=True)
    img_bgr = cv2.imread(query_img_path)
    query_feat = extractor.extract_feature(img_bgr)

    results = search_topk(query_feat, gallery, top_k=top_k, threshold=0.0)
    return results

if __name__ == "__main__":
    query_sample = os.path.join("query", "0001_01.jpg")
    if not os.path.exists(query_sample):
        # Check any image in query directory
        q_files = [os.path.join("query", f) for f in os.listdir("query") if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if q_files:
            query_sample = q_files[0]

    if os.path.exists(query_sample):
        print(f"Running match query on: {query_sample}")
        results = match_query(query_sample, top_k=3)
        print("\n--- TOP MATCHES ---")
        for rank, (match_path, score) in enumerate(results, start=1):
            print(f"Rank {rank}: {os.path.basename(match_path)} | Confidence: {score * 100:.2f}%")
    else:
        print("Please place a query image in the 'query' folder to test.")
