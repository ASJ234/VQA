import os
import zipfile
from huggingface_hub import hf_hub_download

DATA_DIR = "pmc-vqa_data"
os.makedirs(DATA_DIR, exist_ok=True)

REPO = "RadGenome/PMC-VQA"

FILES = {
    "train_2.csv": f"{DATA_DIR}/train_2.csv",
    "test_2.csv": f"{DATA_DIR}/test_2.csv",
}

print("Downloading CSVs from Hugging Face...")
for remote, local in FILES.items():
    hf_hub_download(repo_id=REPO, filename=remote, repo_type="dataset",
                    local_dir=DATA_DIR)
    print(f"  {local} ready")

print("Downloading images_2.zip (2.2 GB, may take a while)...")
images_zip = hf_hub_download(repo_id=REPO, filename="images_2.zip",
                             repo_type="dataset")

print("Extracting images...")
with zipfile.ZipFile(images_zip, 'r') as z:
    z.extractall(os.path.join(DATA_DIR, 'images_2'))

stale_figures = os.path.join(DATA_DIR, 'figures')
target_figures = os.path.join(DATA_DIR, 'images_2', 'figures')
if os.path.isdir(stale_figures) and not os.path.isdir(target_figures):
    print("Moving figures/ into images_2/figures ...")
    os.rename(stale_figures, target_figures)

print("Done!")
