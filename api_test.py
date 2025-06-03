import os
import random
import requests
from collections import Counter
import base64

API_URL = "http://127.0.0.1:8000/predict"
SAVE_SHAP = False   # Set True if you want to save SHAP plots for each image

random.seed(42)

real_dir = "scripts/xai_face_clustering/data/Human_Faces_ds/fake"
ai_dir   = "scripts/xai_face_clustering/data/Human_Faces_ds/real"
shap_dir = "api_test_shap_plots"

def test_folder(folder, true_label, num=50):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(files) < num:
        print(f"Warning: Requested {num} files but only found {len(files)} in {folder}")
        num = len(files)
    sampled_files = random.sample(files, num)
    results = []
    for fpath in sampled_files:
        with open(fpath, 'rb') as f:
            files_dict = {'file': (os.path.basename(fpath), f, 'image/jpeg')}
            try:
                response = requests.post(API_URL, files=files_dict)
                res = response.json()
            except Exception as e:
                print(f"Request failed for {fpath}: {e}")
                continue
            pred = res.get('prediction', 'UNKNOWN')
            print(f"\nFile: {os.path.basename(fpath)} | True: {true_label} | Pred: {pred}")
            print(f"Debug: {res.get('debug')}")
            if SAVE_SHAP and 'shap_plot' in res and res['shap_plot']:
                os.makedirs(shap_dir, exist_ok=True)
                shap_b64 = res['shap_plot']
                out_name = f"{true_label}_{os.path.splitext(os.path.basename(fpath))[0]}_shap.png"
                with open(os.path.join(shap_dir, out_name), "wb") as out_f:
                    out_f.write(base64.b64decode(shap_b64))
            results.append((true_label, pred))
    return results

real_results = test_folder(real_dir, "Real")
ai_results   = test_folder(ai_dir, "AI")

all_results = real_results + ai_results
conf_matrix = Counter(all_results)
total_real = sum(1 for x in all_results if x[0] == "Real")
total_ai   = sum(1 for x in all_results if x[0] == "AI")
correct_real = conf_matrix[("Real", "Real")]
incorrect_real = conf_matrix[("Real", "AI")]
correct_ai = conf_matrix[("AI", "AI")]
incorrect_ai = conf_matrix[("AI", "Real")]

accuracy = (correct_real + correct_ai) / len(all_results) if all_results else 0
real_precision = correct_real / (correct_real + incorrect_ai) if (correct_real + incorrect_ai) > 0 else 0
ai_precision = correct_ai / (correct_ai + incorrect_real) if (correct_ai + incorrect_real) > 0 else 0

print("\n=== FINAL STATISTICS ===")
print(f"Total tested: {len(all_results)} (Real: {total_real}, AI: {total_ai})")
print(f"Correct Real predictions: {correct_real}/{total_real}")
print(f"Incorrect Real predictions (predicted AI): {incorrect_real}/{total_real}")
print(f"Correct AI predictions: {correct_ai}/{total_ai}")
print(f"Incorrect AI predictions (predicted Real): {incorrect_ai}/{total_ai}")
print(f"Overall accuracy: {accuracy:.2f}")
print(f"Precision for Real: {real_precision:.2f}")
print(f"Precision for AI: {ai_precision:.2f}")
