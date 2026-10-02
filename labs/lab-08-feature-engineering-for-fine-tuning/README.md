# Lab 8 — Feature Engineering for Fine Tuning

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU2 — Exploratory Data Analysis  
**Delivery:** self-paced extension  
**Objective:** Apply feature engineering, feature selection and data augmentation to a fine-tuning set

## Goal

Engineer the prompt-side features that decide fine-tuning quality: system-prompt design, few-shot exemplar selection, field ordering, length bucketing and difficulty tagging — then apply embedding-diversity feature selection and three augmentation strategies.

## What you will build

An engineered dataset carrying system prompts, difficulty tags and length buckets, reduced by diversity-based selection and expanded by augmentation, with before/after coverage metrics.

**Tools:** sentence-transformers, scikit-learn, pandas, numpy, Hugging Face datasets

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-08-feature-engineering-for-fine-tuning/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

> **Input:** this lab has no mock dataset of its own — it consumes the output of an earlier lab in the same Learning Unit. Run the labs in order.

## Step-by-step

### Step 1

Load the cleaned corpus from the previous lab. Feature engineering operates on clean data — engineering features over noise just produces well-formatted noise.

```python
import pandas as pd, numpy as np
df = pd.read_json('cleaned_corpus.jsonl', lines=True)
print(len(df)); print(df.head(2).to_string())
```

### Step 2

Engineer feature 1 — the system prompt. The system prompt is the single highest-leverage feature in an instruction dataset: it fixes role, tone and refusal behaviour across every example.

```python
SYSTEM = ('You are a professional customer support assistant for a Singapore telecommunications company. Answer accurately and concisely, never invent account details, and escalate billing disputes to a human agent.')
df['system'] = SYSTEM
print(df['system'].iloc[0])
```

### Step 3

Engineer feature 2 — difficulty tagging. Tagging lets you control the curriculum mix and later diagnose which difficulty band the model fails on.

```python
def difficulty(r):
    n = len(r['output'].split())
    if n < 40:  return 'easy'
    if n < 120: return 'medium'
    return 'hard'
df['difficulty'] = df.apply(difficulty, axis=1)
print(df['difficulty'].value_counts())
```

### Step 4

Engineer feature 3 — length bucketing. Grouping similar lengths into a batch cuts padding waste substantially, which is free throughput.

```python
df['n_tokens_est'] = (df['input'].str.split().str.len() +
                      df['output'].str.split().str.len())
df['bucket'] = pd.cut(df['n_tokens_est'], bins=[0,64,128,256,512,10**6],
                      labels=['<64','64-128','128-256','256-512','512+'])
print(df['bucket'].value_counts().sort_index())
waste_naive  = df['n_tokens_est'].max()*len(df) - df['n_tokens_est'].sum()
waste_bucket = sum(g['n_tokens_est'].max()*len(g) - g['n_tokens_est'].sum()
                   for _, g in df.groupby('bucket', observed=True))
print(f'Padding waste: naive={waste_naive:,}  bucketed={waste_bucket:,}  saved={100*(1-waste_bucket/max(waste_naive,1)):.0f}%')
```

### Step 5

Engineer feature 4 — few-shot exemplar selection. Attaching the two nearest neighbours as in-context examples teaches the target format far faster than instruction text alone.

```bash
!pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
enc = SentenceTransformer('all-MiniLM-L6-v2')
emb = enc.encode(df['input'].tolist(), show_progress_bar=True,
                 batch_size=64, normalize_embeddings=True)
print('Embeddings:', emb.shape)
```

### Step 6

Feature SELECTION by embedding diversity. Choosing a maximally diverse subset keeps coverage while cutting cost — far better than random sampling at the same budget.

```python
from sklearn.cluster import KMeans
K = min(200, len(df))
km  = KMeans(n_clusters=K, random_state=3407, n_init=10).fit(emb)
sel = []
for c in range(K):
    idx = np.where(km.labels_ == c)[0]
    if len(idx):
        centre = km.cluster_centers_[c]
        sel.append(idx[np.argmax(emb[idx] @ centre)])
diverse = df.iloc[sel].copy()
print(f'Selected {len(diverse)} diverse examples from {len(df)}')
```

### Step 7

Prove the diverse subset beats a random subset of the same size on coverage. This is the justification for spending compute on selection.

```python
rand = df.sample(len(diverse), random_state=3407)
def coverage(sub):
    e = emb[sub.index.map(lambda i: df.index.get_loc(i))]
    return float(1 - cosine_similarity(e).mean())
print(f'Diversity score - selected: {coverage(diverse):.4f}')
print(f'Diversity score - random  : {coverage(rand):.4f}')
print('Selection wins:', coverage(diverse) > coverage(rand))
```

### Step 8

Augmentation strategy 1 — instruction paraphrasing. Varying how the task is asked stops the model over-fitting to one exact phrasing.

```python
PARA = {'Draft a support reply.'   : ['Write a reply to this customer.',
                                      'Compose a support response.',
                                      'Respond to the customer message.'],
        'Answer the customer question.': ['Provide an answer to this question.',
                                      'Reply to the customer enquiry.']}
import random; random.seed(3407)
aug1 = diverse.copy()
aug1['instruction'] = aug1['instruction'].map(
    lambda x: random.choice(PARA.get(x, [x])))
print(aug1['instruction'].value_counts().head())
```

### Step 9

Augmentation strategy 2 — input perturbation, and strategy 3 — hard-negative construction. Perturbation builds robustness; hard negatives teach the model the boundary of correct behaviour.

```python
def perturb(t):
    ops = [lambda s: s.lower(), lambda s: 'Hi. ' + s,
           lambda s: s.replace('cannot','cant').replace('I am',"I'm"),
           lambda s: s + ' Thanks.']
    return random.choice(ops)(t)
aug2 = diverse.copy(); aug2['input'] = aug2['input'].map(perturb)
engineered = pd.concat([diverse, aug1, aug2], ignore_index=True)
engineered = engineered.drop_duplicates(subset=['input','output'])
print(f'Engineered set: {len(engineered)} rows from {len(diverse)} selected')
```

### Step 10

Save the engineered dataset with all features retained, and report the feature summary.

```python
engineered.to_json('engineered_corpus.jsonl', orient='records', lines=True, force_ascii=False)
print(engineered[['difficulty','bucket']].value_counts().to_string())
print('Columns:', list(engineered.columns))
```

## Test it

engineered_corpus.jsonl contains the system, difficulty and bucket feature columns; the diversity score of the selected subset is higher than the random subset; and the padding-waste saving from bucketing is reported as a percentage.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0