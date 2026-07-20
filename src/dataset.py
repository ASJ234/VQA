import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PMCVQADataset(Dataset):
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def __init__(self, csv_path, image_dir, tokenizer, max_length=128,
                 image_size=224, split=None):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is None or row.get('split', '').strip() == split:
                    self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def _clean_choice(self, text):
        text = text.strip()
        if ': ' in text:
            text = text.split(': ', 1)[-1]
        return text

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_path = os.path.join(self.image_dir, sample['Figure_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        question = sample['Question'].strip()
        q_enc = self.tokenizer(
            question, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt',
        )

        choices = [
            self._clean_choice(sample['Choice A']),
            self._clean_choice(sample['Choice B']),
            self._clean_choice(sample['Choice C']),
            self._clean_choice(sample['Choice D']),
        ]
        c_enc = self.tokenizer(
            choices, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt',
        )

        label = self.label_map[sample['Answer'].strip()]

        return {
            'image': image,
            'question_input_ids': q_enc['input_ids'].squeeze(0),
            'question_attention_mask': q_enc['attention_mask'].squeeze(0),
            'choices_input_ids': c_enc['input_ids'],
            'choices_attention_mask': c_enc['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long),
            'figure_path': sample['Figure_path'],
        }


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out
