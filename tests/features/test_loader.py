import pytest
from scripts.xai_face_clustering.data.loader import load_images

def test_load_images_shape():
    images, labels, filenames = load_images("xai_face_clustering/data/Human_Faces_ds")
    
    assert len(images) == len(labels) == len(filenames)
    assert images.shape[1:] == (3, 224, 224)
