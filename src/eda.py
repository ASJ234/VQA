import os
import csv
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

DATA_DIR = 'pmc-vqa_data'
IMAGE_DIR = f'{DATA_DIR}/images_2/figures'
OUTPUT_DIR = 'outputs/eda'


def load_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def label_distribution(rows, name):
    counter = Counter(r['Answer'].strip() for r in rows)
    total = sum(counter.values())
    print(f"\n  {name} ({total:,} total):")
    for lbl in ['A', 'B', 'C', 'D']:
        c = counter.get(lbl, 0)
        print(f"    {lbl}: {c:>6d} ({c/total*100:5.2f}%)")
    return counter


def question_type_analysis(rows):
    patterns = ['What', 'Which', 'How', 'Where', 'Why', 'When',
                'Is', 'Does', 'Can', 'Are', 'Name', 'Describe']
    counter = Counter()
    for r in rows:
        q = r['Question'].strip()
        for pat in patterns:
            if q.startswith(pat):
                counter[pat] += 1
                break
        else:
            counter['Other'] += 1
    return dict(counter)


def image_stats(paths, sample_size=5000):
    sampled = np.random.choice(paths, min(sample_size, len(paths)), replace=False)
    widths, heights, ratios, sizes = [], [], [], []
    for fname in tqdm(sampled, desc='Images'):
        fp = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(fp):
            continue
        try:
            with Image.open(fp) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
                ratios.append(w / h if h else 1)
                sizes.append(os.path.getsize(fp))
        except Exception:
            pass
    return widths, heights, ratios, sizes


def eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train = load_csv(f'{DATA_DIR}/train_2.csv')
    test = load_csv(f'{DATA_DIR}/test_2.csv')

    print("=" * 60)
    print("PMC-VQA v2 — Exploratory Data Analysis")
    print("=" * 60)

    # 1 — Sizes
    print(f"\n── 1. Dataset Sizes ──")
    print(f"  Train: {len(train):>8,}")
    print(f"  Test:  {len(test):>8,}")

    # 2 — Label distribution
    print(f"\n── 2. Label Distribution ──")
    train_labels = label_distribution(train, 'Train')
    test_labels = label_distribution(test, 'Test')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (counter, title, total) in zip(
            axes,
            [(train_labels, 'Train', len(train)),
             (test_labels, 'Test', len(test))]):
        vals = [counter.get(l, 0) for l in ['A', 'B', 'C', 'D']]
        bars = ax.bar(['A', 'B', 'C', 'D'], vals,
                       color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'],
                       edgecolor='white')
        ax.set_title(f'{title} (n={total:,})', fontweight='bold')
        ax.set_ylabel('Count')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f'{v:,} ({v/total*100:.1f}%)', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/label_distribution.png', dpi=150)
    plt.close()
    print(f"  Plot saved: {OUTPUT_DIR}/label_distribution.png")

    # 3 — Question types
    print(f"\n── 3. Question Types ──")
    qt = question_type_analysis(train)
    for k, v in sorted(qt.items(), key=lambda x: -x[1]):
        print(f"  {k:>12s}: {v:>6d} ({v/len(train)*100:5.2f}%)")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.cm.tab10(range(len(qt)))
    ax.bar(qt.keys(), qt.values(), color=colors, edgecolor='white')
    ax.set_title('Question Types (Train)', fontweight='bold')
    ax.set_ylabel('Count')
    for i, (k, v) in enumerate(qt.items()):
        ax.text(i, v + max(qt.values()) * 0.01, str(v), ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/question_types.png', dpi=150)
    plt.close()
    print(f"  Plot saved: {OUTPUT_DIR}/question_types.png")

    # 4 — Image statistics
    print(f"\n── 4. Image Statistics ──")
    all_paths = list(set(r['Figure_path'] for r in train) |
                     set(r['Figure_path'] for r in test))
    print(f"  Total unique images: {len(all_paths):,}")
    widths, heights, ratios, f_sizes = image_stats(all_paths, 8000)

    print(f"  Width:    mean={np.mean(widths):.0f}, min={min(widths)}, max={max(widths)}")
    print(f"  Height:   mean={np.mean(heights):.0f}, min={min(heights)}, max={max(heights)}")
    print(f"  Aspect:   mean={np.mean(ratios):.2f}")
    print(f"  File sz:  mean={np.mean(f_sizes)/1024:.0f}KB")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(widths, bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].set_title('Width', fontweight='bold')
    axes[0, 1].hist(heights, bins=50, color='coral', edgecolor='white')
    axes[0, 1].set_title('Height', fontweight='bold')
    axes[1, 0].hist(ratios, bins=50, color='seagreen', edgecolor='white')
    axes[1, 0].set_title('Aspect Ratio', fontweight='bold')
    axes[1, 1].hist(np.array(f_sizes) / 1024, bins=50, color='purple', edgecolor='white')
    axes[1, 1].set_title('File Size (KB)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/image_statistics.png', dpi=150)
    plt.close()
    print(f"  Plot saved: {OUTPUT_DIR}/image_statistics.png")

    # 5 — Text lengths
    print(f"\n── 5. Text Lengths (words) ──")
    q_lens = [len(r['Question'].strip().split()) for r in train]
    c_lens = []
    for r in train[:100000]:
        for l in ['A', 'B', 'C', 'D']:
            c_lens.append(len(r[f'Choice {l}'].strip().split()))
    cap_lens = [len(r['Caption'].strip().split()) for r in train]

    print(f"  Question: mean={np.mean(q_lens):.1f}, median={np.median(q_lens):.0f}, max={max(q_lens)}")
    print(f"  Choice:   mean={np.mean(c_lens):.1f}, median={np.median(c_lens):.0f}, max={max(c_lens)}")
    print(f"  Caption:  mean={np.mean(cap_lens):.1f}, median={np.median(cap_lens):.0f}, max={max(cap_lens)}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (data, title) in zip(axes,
            [(q_lens, 'Question'), (c_lens, 'Choice'),
             (np.clip(cap_lens, 0, 200), 'Caption (clipped at 200)')]):
        ax.hist(data, bins=50, color='steelblue', edgecolor='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Words')
        ax.axvline(np.median(data), color='red', ls='--', label=f'median={np.median(data):.0f}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/text_lengths.png', dpi=150)
    plt.close()
    print(f"  Plot saved: {OUTPUT_DIR}/text_lengths.png")

    # 6 — Questions per image
    print(f"\n── 6. Questions per Image ──")
    qpi = Counter()
    for r in train:
        qpi[r['Figure_path']] += 1
    for r in test:
        qpi[r['Figure_path']] += 1
    qpi_vals = list(qpi.values())
    print(f"  Mean: {np.mean(qpi_vals):.2f}, Median: {np.median(qpi_vals):.0f}, Max: {max(qpi_vals)}")
    print(f"  1 QA:  {sum(1 for x in qpi_vals if x == 1):,} ({sum(1 for x in qpi_vals if x == 1)/len(qpi_vals)*100:.1f}%)")
    print(f"  2+ QA: {sum(1 for x in qpi_vals if x >= 2):,} ({sum(1 for x in qpi_vals if x >= 2)/len(qpi_vals)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_show = min(20, max(qpi_vals))
    dist = Counter(qpi_vals)
    ax.bar(range(1, max_show + 1), [dist.get(i, 0) for i in range(1, max_show + 1)],
           color='steelblue', edgecolor='white')
    ax.set_title('Questions per Image', fontweight='bold')
    ax.set_xlabel('# Questions')
    ax.set_ylabel('# Images')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/questions_per_image.png', dpi=150)
    plt.close()
    print(f"  Plot saved: {OUTPUT_DIR}/questions_per_image.png")

    # 7 — Train/Test image overlap
    print(f"\n── 7. Train/Test Image Overlap ──")
    train_imgs = set(r['Figure_path'] for r in train)
    test_imgs = set(r['Figure_path'] for r in test)
    overlap = train_imgs & test_imgs
    print(f"  Train: {len(train_imgs):,} | Test: {len(test_imgs):,} | Overlap: {len(overlap):,}")
    print(f"  Train-only: {len(train_imgs - test_imgs):,} | Test-only: {len(test_imgs - train_imgs):,}")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    vals = [len(train_imgs - test_imgs), len(test_imgs - train_imgs), len(overlap)]
    bars = ax.barh(['Train-only', 'Test-only', 'Overlap'], vals,
                    color=['steelblue', 'coral', 'seagreen'])
    ax.set_title('Image Overlap', fontweight='bold')
    ax.set_xlabel('Count')
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:,}', va='center')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/train_test_overlap.png', dpi=150)
    plt.close()
    print(f"  Plot saved: {OUTPUT_DIR}/train_test_overlap.png")

    # Summary
    summary = {
        'train_samples': len(train),
        'test_samples': len(test),
        'unique_images': len(all_paths),
        'train_label_dist': {k: v for k, v in sorted(train_labels.items())},
        'test_label_dist': {k: v for k, v in sorted(test_labels.items())},
        'question_types': qt,
        'image_width_mean': float(np.mean(widths)),
        'image_height_mean': float(np.mean(heights)),
        'question_word_mean': float(np.mean(q_lens)),
        'caption_word_mean': float(np.mean(cap_lens)),
        'questions_per_image_mean': float(np.mean(qpi_vals)),
        'train_test_overlap': len(overlap),
    }
    with open(f'{OUTPUT_DIR}/eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {OUTPUT_DIR}/eda_summary.json")
    print("EDA complete.")


if __name__ == '__main__':
    eda()
