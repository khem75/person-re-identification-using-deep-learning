import os
import numpy as np
from reid_core import ReIDExtractor, build_gallery_index, save_feature_cache

def main():
    gallery_dir = "gallery"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("Building model...")
    extractor = ReIDExtractor(model_name="osnet_x0_25", use_gpu=True)

    print(f"Extracting gallery features from '{gallery_dir}'...")
    features_db = build_gallery_index(extractor, gallery_dir)

    # Save pickle cache
    cache_pickle = os.path.join(output_dir, "gallery_features.pkl")
    save_feature_cache(features_db, cache_pickle)

    # Save npy cache for compatibility
    cache_npy = os.path.join(output_dir, "gallery_features.npy")
    np.save(cache_npy, features_db)

    print(f"[SUCCESS] Extracted features for {len(features_db)} images successfully!")
    print(f"Saved to '{cache_pickle}' and '{cache_npy}'")

if __name__ == "__main__":
    main()
