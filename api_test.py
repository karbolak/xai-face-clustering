import os
import requests

API_URL = "http://127.0.0.1:8000/predict"

# Folders to test
real_dir = "scripts/xai_face_clustering/data/Human_Faces_Dataset/Real_Images"
ai_dir   = "scripts/xai_face_clustering/data/Human_Faces_Dataset/AI-Generated_Images"

def test_folder(folder, label, num=5):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files = files[:num]
    for fpath in files:
        with open(fpath, 'rb') as f:
            files_dict = {'file': (os.path.basename(fpath), f, 'image/jpeg')}
            response = requests.post(API_URL, files=files_dict)
            res = response.json()
            print(f"\nFile: {os.path.basename(fpath)} | True: {label} | Pred: {res['prediction']}")
            print(f"Debug: {res.get('debug')}")

print("Testing REAL images:")
test_folder(real_dir, "Real", num=5)

print("\nTesting AI images:")
test_folder(ai_dir, "AI", num=5)