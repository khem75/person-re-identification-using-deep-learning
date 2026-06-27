import argparse, os, glob, csv
import numpy as np
import cv2
from tqdm import tqdm
from reid_core import ReIDExtractor, search_topk

def load_paths(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(exts)])

def parse_pid(name):
    import re
    m = re.match(r"^([-\d]+)_c\d+s\d+_.+\.jpg$", os.path.basename(name), re.I)
    return m.group(1) if m and m.group(1) != "-1" else None

def main():
    ap = argparse.ArgumentParser(description="Run Person ReID evaluation script.")
    ap.add_argument("--gallery", default="gallery", help="Path to gallery folder")
    ap.add_argument("--query", default="query", help="Path to query folder")
    ap.add_argument("--model", default="osnet_x0_25", help="ReID model name")
    ap.add_argument("--topk", type=int, default=5, help="Top-K matches")
    ap.add_argument("--out", default=os.path.join("outputs", "reid_results.csv"), help="Output CSV path")
    args = ap.parse_args()

    print(f"Initializing ReID Extractor with model: {args.model}")
    extractor = ReIDExtractor(model_name=args.model, use_gpu=True)

    g_paths = load_paths(args.gallery)
    q_paths = load_paths(args.query)
    
    if not g_paths or not q_paths:
        print("Warning: Empty gallery or query directory.")
        return

    print(f"Found {len(g_paths)} gallery images and {len(q_paths)} query images.")

    # Extract gallery features
    g_db = {}
    print("Extracting gallery features...")
    for gp in tqdm(g_paths, desc="Gallery"):
        img = cv2.imread(gp)
        if img is not None:
            g_db[gp] = extractor.extract_feature(img)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    hits, total = 0, 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "q_pid", "rank", "gallery", "g_pid", "similarity_score"])
        
        print("Searching and evaluating queries...")
        for qp in tqdm(q_paths, desc="Queries"):
            img = cv2.imread(qp)
            if img is None:
                continue
            q_feat = extractor.extract_feature(img)
            results = search_topk(q_feat, g_db, top_k=args.topk, threshold=0.0)
            
            qpid = parse_pid(qp)
            for rank, (gp, sim) in enumerate(results, start=1):
                gpid = parse_pid(gp)
                w.writerow([qp, qpid, rank, gp, gpid, f"{sim:.4f}"])
                
            if qpid is not None and results:
                total += 1
                top1_gpid = parse_pid(results[0][0])
                if top1_gpid == qpid:
                    hits += 1

    if total > 0:
        print(f"Hit@1 Accuracy: {hits}/{total} ({hits/total*100:.1f}%)")
    print(f"[SUCCESS] Saved evaluation results to {args.out}")

if __name__ == "__main__":
    main()
