# XAI Face Clustering

## API Usage
To run the FastAPI inference server locally:
```bash
python -m uvicorn inference_api:app --reload
```
Then open your browser and go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the API interactively.

### Example cURL Requests
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@/path/to/AI-Generated_Images/000003.jpg"
```
Make sure the path points to a valid image file on your machine.

---

## 📦 Environment Setup

### 1. Conda (Recommended)
```bash
conda create -n aml-env python=3.10
conda activate aml-env
conda config --add channels conda-forge
conda install --yes --file requirements.txt
```

### 2. pip (if not using conda)
Ensure `requirements.txt` uses compatible names (e.g., `torch`, `opencv-python`):
```bash
pip install -r requirements.txt
```



## Adding Large Files with Git LFS
```bash
git lfs install
git lfs track "*.npz"
git add .gitattributes
git add scripts/xai_face_clustering/features/embeddings.npz
```


## Project Structure

```
AppliedML_project/
├── main.py
├── inference_api.py
├── README.md
├── requirements.txt
├── tests/
│   ├── test_main.py
│   └── (unit tests by module)
├── scripts/
│   └── xai_face_clustering/
│       ├── data/
│       │   ├── loader.py
│       │   └── Human_Faces_Dataset/
│       ├── features/
│       │   ├── cnn_embeddings.py
│       │   ├── pca.py
│       │   ├── pca_variance_plot.py
│       │   └── exploratory_plots/
│       ├── models/
│       │   ├── clustering.py
│       │   ├── surrogate.py
│       │   └── xai.py
```

##  Module Overview

### `main.py`
Controls the full pipeline via CLI:
- Load + preprocess images
- Extract CNN embeddings
- Apply PCA
- Cluster with DBSCAN/KMeans/GMM
- Train surrogate model
- Generate SHAP explanations

### `data/loader.py`
- Loads and normalizes images from dataset
- Supports Real vs AI-labeled folders

### `features/`
- `cnn_embeddings.py`: extracts embeddings via pretrained Facenet model
- `pca.py`: applies PCA (with caching)
- `pca_variance_plot.py`: visualizes explained variance

### `models/`
- `clustering.py`: clustering logic (KMeans, DBSCAN, GMM)
- `surrogate.py`: trains interpretable classifier (LogReg/SVM/Tree)
- `xai.py`: runs SHAP on surrogate model

### `tests/`
- Covers each component: data loading, features, models, SHAP outputs

---

## 📊 Visual Flow of Pipeline

![Flowchart part 1](flowchart_1.png)
![Flowchart part 2](flowchart_2.png)

---

## 🧠 Notes
- SHAP explanations are saved under: `scripts/xai_face_clustering/features/shap_explanations/`
- Embedding is cached on first run to avoid recomputation, which takes time.
- You can switch models or clustering method via CLI flags(e.g.: --pca_components=50).

