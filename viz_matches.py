# viz_matches.py
import os, csv, cv2
import numpy as np
from pathlib import Path

CSV = "reid_results.csv"
OUT = "viz_topk"

os.makedirs(OUT, exist_ok=True)

rows = {}
with open(CSV, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for d in r:
        rows.setdefault(d["query"], []).append(d)

for q, lst in rows.items():
    img_q = cv2.imread(q)
    if img_q is None: 
        print("skip missing", q); 
        continue
    tiles = [img_q]
    for d in lst:
        gp = d["gallery"]
        sim = float(d["cosine_similarity"])
        img_g = cv2.imread(gp)
        if img_g is None: 
            continue
        # annotate sim
        cv2.putText(img_g, f"sim:{sim:.3f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        tiles.append(img_g)
    # make a horizontal strip
    h = max(t.shape[0] for t in tiles)
    resized = [cv2.resize(t, (int(t.shape[1]*h/t.shape[0]), h)) for t in tiles]
    strip = np.concatenate(resized, axis=1)
    out_path = Path(OUT)/ (Path(q).stem + "_topk.jpg")
    cv2.imwrite(str(out_path), strip)
    print("wrote", out_path)
