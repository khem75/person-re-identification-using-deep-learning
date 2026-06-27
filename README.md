# Person Re-Identification Using Deep Learning (OSNet) 👤

An end-to-end Deep Learning Person Re-Identification (ReID) system powered by PyTorch and Torchreid. Features both an interactive **Tkinter Desktop Studio** and a **Streamlit Web GUI**, supporting dynamic similarity threshold tuning, multi-model selection, and automated match audit logging.

---

## 🌟 Key Features

- **High-Accuracy Embedding Extraction**: Built using `OSNet` architecture with standard ImageNet normalization and L2 feature normalization.
- **Interactive Desktop Studio (`app.py`)**: Tkinter GUI with live query navigation, interactive similarity threshold slider (0.0 – 1.0), top-K rank selection, and CSV audit logging.
- **Streamlit Web Application (`app_reid_gui.py`)**: Modern browser dashboard with instant file upload, performance metrics, visual match confidence badges, and Excel exports.
- **Unified Engine (`reid_core.py`)**: Centralized module managing feature indexing, database caching, and rank matching across CLI tools and GUIs.

---

## 📁 Project Structure

```
person_reid_project/
├── app.py                 # Tkinter Desktop GUI application
├── app_reid_gui.py        # Streamlit Web application
├── reid_core.py           # Unified feature extraction & matching core
├── extract_features.py    # CLI tool for gallery feature extraction
├── match.py               # Quick query matching script
├── run_reid_simple.py     # Batch search & evaluation pipeline
├── gallery/               # Gallery image database
├── query/                 # Query images directory
└── outputs/               # Saved feature caches and match logs
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/khem75/person-re-identification-using-deep-learning.git
cd person-re-identification-using-deep-learning
pip install -r requirements.txt
```

### 2. Run Desktop GUI
Launch the interactive Tkinter application:
```bash
python app.py
```

### 3. Run Web GUI
Launch the Streamlit web application:
```bash
streamlit run app_reid_gui.py
```

### 4. CLI Feature Extraction & Quick Match
Extract gallery features and run search from command line:
```bash
python extract_features.py
python match.py
```

---

## 🛠️ Tech Stack
- **Deep Learning**: PyTorch, Torchreid (OSNet)
- **Computer Vision**: OpenCV, PIL
- **GUI Frameworks**: Tkinter, Streamlit
- **Data Analysis**: NumPy, Pandas, Scikit-learn
