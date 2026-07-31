import math
import re
from collections import Counter

import numpy as np


_TOKEN_RE = re.compile(r'[a-z0-9]+')


def tokenize(text):
    return _TOKEN_RE.findall(str(text).lower())


def compute_accuracy(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    return float((labels == preds).mean())


def compute_f1(labels, preds, num_classes=4):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    per_class = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        per_class.append(float(f1))
    return float(np.mean(per_class)), per_class


def compute_bleu(pred_texts, ref_texts, max_n=4, smooth=True):
    preds = [tokenize(t) for t in pred_texts]
    refs = [[tokenize(r)] for r in ref_texts]

    counts = [0.0] * max_n
    clipped = [0.0] * max_n
    total_pred_len = 0.0
    total_ref_len = 0.0

    for pred, ref in zip(preds, refs):
        total_pred_len += len(pred)
        total_ref_len += min((len(r) for r in ref), default=0)
        for n in range(1, max_n + 1):
            pred_count = Counter(_ngrams(pred, n))
            ref_counts = [Counter(_ngrams(r, n)) for r in ref]
            counts[n - 1] += sum(pred_count.values())
            for ng, c in pred_count.items():
                max_ref = max((rc[ng] for rc in ref_counts), default=0)
                clipped[n - 1] += min(c, max_ref)

    precisions = []
    for n in range(1, max_n + 1):
        p = clipped[n - 1] / max(counts[n - 1], 1e-12)
        if smooth:
            p = (clipped[n - 1] + 1.0) / (counts[n - 1] + 1.0)
        precisions.append(p)

    if total_pred_len >= total_ref_len:
        bp = 1.0
    elif total_pred_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - total_ref_len / total_pred_len)

    scores = {}
    for n in range(1, max_n + 1):
        log_prec = sum(math.log(precisions[i]) for i in range(n)) / n
        scores[f'bleu_{n}'] = float(bp * math.exp(log_prec))
    return scores


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 0))]


def _ensure_wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('test')
        return wn
    except LookupError:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet as wn
        wn.synsets('test')
        return wn


def _word_wup(wn, word1, word2):
    if word1 == word2:
        return 1.0
    syns1 = wn.synsets(word1, pos=wn.NOUN) or wn.synsets(word1)
    syns2 = wn.synsets(word2, pos=wn.NOUN) or wn.synsets(word2)
    best = 0.0
    for s1 in syns1:
        for s2 in syns2:
            sim = s1.wup_similarity(s2)
            if sim is not None and sim > best:
                best = sim
    return best


def _wups_sentence(wn, pred_tokens, ref_tokens, threshold):
    if pred_tokens == ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    total = 0.0
    for rt in ref_tokens:
        best = 0.0
        for pt in pred_tokens:
            s = _word_wup(wn, pt, rt)
            if s > best:
                best = s
        total += best
    avg = total / len(ref_tokens)
    if avg < threshold:
        avg = avg * threshold
    return avg


def compute_wups(pred_texts, ref_texts, threshold=0.0):
    wn = _ensure_wordnet()
    scores = []
    for pt, rt in zip(pred_texts, ref_texts):
        scores.append(_wups_sentence(
            wn, tokenize(pt), tokenize(rt), threshold))
    return float(np.mean(scores)) if scores else 0.0


def compute_all_metrics(labels, preds, pred_texts, ref_texts):
    macro_f1, per_class_f1 = compute_f1(labels, preds)
    bleu = compute_bleu(pred_texts, ref_texts)
    return {
        'accuracy': compute_accuracy(labels, preds),
        'f1_macro': macro_f1,
        'f1': {chr(65 + c): v for c, v in enumerate(per_class_f1)},
        'wups_0.0': compute_wups(pred_texts, ref_texts, threshold=0.0),
        'wups_0.9': compute_wups(pred_texts, ref_texts, threshold=0.9),
        'bleu': bleu,
    }
