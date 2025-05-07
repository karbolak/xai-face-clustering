import torch
from xai_face_clustering.features.cnn_embeddings import extract_embeddings

def test_embeddings_shape():
    dummy_input = torch.randn(10, 3, 224, 224)  # 10 dummy images
    embeddings = extract_embeddings(dummy_input, model_name="resnet50")
    assert embeddings.shape[0] == 10
    assert embeddings.shape[1] >= 512  # ResNet should give 2048D
