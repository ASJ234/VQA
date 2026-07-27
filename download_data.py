import os
import csv
import zipfile
import urllib.request
from huggingface_hub import hf_hub_download

DATA_DIR = "pmc-vqa_data"
os.makedirs(DATA_DIR, exist_ok=True)

REPO = "RadGenome/PMC-VQA"

print("Downloading CSVs from Hugging Face...")
train_path = hf_hub_download(repo_id=REPO, filename="train2.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=REPO, filename="test2.csv", repo_type="dataset")

print("Converting CSVs to expected format...")
for src, dst, split_label in [
    (train_path, f"{DATA_DIR}/train_2.csv", "train"),
    (test_path, f"{DATA_DIR}/test_2.csv", "test"),
]:
    with open(src, 'r') as fin, open(dst, 'w', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=[
            'index', 'Figure_path', 'Caption', 'Question',
            'Choice A', 'Choice B', 'Choice C', 'Choice D',
            'Answer', 'split'
        ])
        writer.writeheader()
        for i, row in enumerate(reader):
            writer.writerow({
                'index': i,
                'Figure_path': row['Figure_path'],
                'Caption': '',
                'Question': row['Question'],
                'Choice A': row['Choice A'],
                'Choice B': row['Choice B'],
                'Choice C': row['Choice C'],
                'Choice D': row['Choice D'],
                'Answer': row['Answer_label'],
                'split': split_label,
            })
    print(f"  {dst} written")

print("Downloading images2.zip (this may take a while)...")
images_zip = hf_hub_download(repo_id=REPO, filename="images2.zip", repo_type="dataset")

print("Extracting images...")
with zipfile.ZipFile(images_zip, 'r') as z:
    z.extractall(DATA_DIR)

print("Done!")
