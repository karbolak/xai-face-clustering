#!/usr/bin/env python
import os
import torch
import argparse
from types import SimpleNamespace
from xai_face_clustering.pipeline import PipelineRunner

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="xai_face_clustering/data/Human_Faces_ds")
    parser.add_argument("--model",       default="facenet")
    parser.add_argument("--pca-cmp",     type=int,   default=100)
    parser.add_argument("--eps",         type=float, default=60.0)
    parser.add_argument("--min-s",       type=int,   default=5)
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--n-waterfalls",type=int,   default=5)
    parser.add_argument("--cache",       default="artifacts/embeddings/embeddings.pkl")
    parser.add_argument("--xai-out",     default="plots/shap_explanations")
    parser.add_argument("--batch-size",  type=int, default=512)
    args = parser.parse_args()

    cfg = SimpleNamespace(
        DATA_DIR       = args.data_dir,
        MODEL_NAME     = args.model,
        IMG_SIZE         = (224,224),
        MEAN             = [0.485,0.456,0.406],
        STD              = [0.229,0.224,0.225],
        CACHE            = args.cache,
        PCA_COMPONENTS   = args.pca_cmp,
        PCA_SAVE         = "artifacts/models/pca_model.pkl",
        CLUSTER_METHOD   = "dbscan",
        EPS              = args.eps,
        MIN_SAMPLES      = args.min_s,
        K                = None,
        SURROGATE        = "svm",
        TEST_SIZE        = args.test_size,
        N_WATERFALLS     = args.n_waterfalls,
        XAI_OUT          = args.xai_out,
        BATCH_SIZE       = args.batch_size
    )

    runner = PipelineRunner(cfg)
    result = runner.run()
    print(result)
