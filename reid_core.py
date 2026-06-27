import os
import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchreid
from typing import Dict, List, Tuple, Optional, Union

# Torchreid standard input size
INPUT_SIZE = (256, 128)  # Height, Width

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Preprocess OpenCV BGR image for ReID models:
    1. Resize to (256, 128)
    2. Convert BGR to RGB
    3. Normalize with ImageNet mean and std
    4. Convert to float tensor CHW and add batch dimension NCHW
    """
    img_resized = cv2.resize(img_bgr, (INPUT_SIZE[1], INPUT_SIZE[0]))
    img_rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0
    img_norm = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor

def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D feature vectors."""
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(feat1, feat2) / (norm1 * norm2))

class ReIDExtractor:
    """Wrapper class for building model and extracting L2-normalized ReID features."""
    def __init__(self, model_name: str = "osnet_x0_25", use_gpu: bool = True):
        self.model_name = model_name
        self.device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_feature(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extract L2-normalized 1D feature vector from a single BGR image."""
        tensor = preprocess_image(img_bgr).to(self.device)
        feat = self.model(tensor)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

def build_gallery_index(extractor: ReIDExtractor, gallery_dir: str) -> Dict[str, np.ndarray]:
    """Extract features for all valid images in gallery directory."""
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in sorted(os.listdir(gallery_dir)) if f.lower().endswith(valid_exts)]
    db = {}
    for f in files:
        path = os.path.join(gallery_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
        feat = extractor.extract_feature(img)
        db[path] = feat
    return db

def save_feature_cache(db: Dict[str, np.ndarray], cache_path: str):
    """Save feature database dictionary to pickle file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(db, f)

def load_feature_cache(cache_path: str) -> Dict[str, np.ndarray]:
    """Load feature database dictionary from pickle file."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)

def search_topk(
    query_feat: np.ndarray,
    gallery_db: Dict[str, np.ndarray],
    top_k: int = 5,
    threshold: float = 0.0
) -> List[Tuple[str, float]]:
    """Search gallery database and return list of (image_path, similarity_score) sorted descending."""
    results = []
    for path, g_feat in gallery_db.items():
        score = compute_cosine_similarity(query_feat, g_feat)
        if score >= threshold:
            results.append((path, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
