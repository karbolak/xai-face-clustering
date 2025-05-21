import os
from sklearn.model_selection import train_test_split

def ensure_dir(path):
    """
    Create `path` if it doesn’t exist.
    """
    os.makedirs(path, exist_ok=True)
    return path

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Shorthand for train_test_split.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
