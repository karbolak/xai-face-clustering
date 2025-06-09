# XAI Face Clustering

## Environment Setup

### 1. Environment creation
```bash
conda create -n aml-env python=3.10
conda activate aml-env
```

### 2. pip (recommended)
```bash
pip install -r requirements.txt
```

### 2. conda (if not using pip)

```bash
conda config --add channels conda-forge
# Ensure `requirements.txt` uses compatible names (e.g., `pytorch`, `opencv`):
conda install --yes --file requirements.txt
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


## Visual Flow of Pipeline

![Flowchart part 1](flowchart_1.png)
![Flowchart part 2](flowchart_2.png)


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

## Docker File

There is an image provided in the repository which can be run. Alternatively, if you wish to build your own image, ensure docker is running on your machine(via desktop application: https://www.docker.com/products/docker-desktop/), and then build with the command:
```bash
docker build -t xai-face-clustering .
```
This command will take the docker file and build everything which is required for the ml application to run, starting with acquiring conda, to creating an environment and adding the necessary requirements.
Once the build is done, the image can be run on an interactive shell(defualt -> CMD ["bash"]) through:
```bash
docker run -it xai-face-clustering
```

###Dockerised web
To run the fastapi, enter this command in your CLI:
```bash
docker run -it -p 8000:8000 xai-face-clustering python -m uvicorn inference_api:app --host 0.0.0.0 --port 8000
```
Once built, access the link http://0.0.0.0:8000/docs on your default browser in order to see the web GUI.

To run streamlit, a similar command is required to be typed in the CLI:
```bash
docker run -it -p 8000:8000 xai-face-clustering streamlit run streamlit_app.py --server.port=8000 --server.address=0.0.0.0
```
Then access the url provided http://0.0.0.0:8000. This will redirect you to the streamlit app which allows you to upload images from your computer and test whether they are real or AI generated.

## Notes
- SHAP explanations are saved under: `scripts/xai_face_clustering/features/shap_explanations/`
- Embedding is cached on first run to avoid recomputation, which takes time.
- You can switch models or clustering method via CLI flags(e.g.: --pca_components=50).

