# Lab 2 — Create a Versioned Data Repository for Fine Tuning

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU1 — Data Engineering  
**Delivery:** self-paced extension  
**Objective:** Build an immutable, versioned, reproducible data repository for a fine-tuning corpus

## Goal

Take a raw customer-support export and turn it into a proper fine-tuning data repository: a content-hashed, versioned, split dataset with a dataset card, so any future training run can be traced back to the exact bytes it was trained on.

## What you will build

A versioned dataset repository at data_repo/v1/ containing train/validation/test splits, a manifest.json with a SHA-256 content hash and row counts, and a DATASET_CARD.md.

**Tools:** Hugging Face datasets, pandas, hashlib, JSON Lines

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-02-create-a-versioned-data-repository-for-fine-tuni/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

**Mock data provided:**

- `data/support_tickets_raw.csv`

## Step-by-step

### Step 1

Load the provided mock support-ticket export. This is the raw material — deliberately messy, as a real export always is.

```python
import pandas as pd
df = pd.read_csv('data/support_tickets_raw.csv')
print(df.shape)
print(df.head(3).to_string())
print(df.isna().sum())
```

### Step 2

Define the repository layout. Separating raw, interim and processed makes every stage re-runnable and lets you prove which transformation produced which file.

```python
import os
for d in ['data_repo/v1/raw','data_repo/v1/interim','data_repo/v1/processed']:
    os.makedirs(d, exist_ok=True)
df.to_csv('data_repo/v1/raw/support_tickets_raw.csv', index=False)
```

### Step 3

Normalise the schema to the three fields a fine-tuning set needs — instruction, input and output. Fixing the schema now means every downstream lab reads the same columns.

```python
records = (df.rename(columns={'customer_message':'input',
                              'agent_reply'     :'output',
                              'task'            :'instruction'})
             [['instruction','input','output']]
             .dropna(subset=['input','output']))
print('Usable records:', len(records))
```

### Step 4

Split into train / validation / test with a FIXED seed. The seed is part of the dataset version — an unseeded split cannot be reproduced and silently invalidates every later comparison.

```python
from sklearn.model_selection import train_test_split
SEED = 3407
train, temp = train_test_split(records, test_size=0.20, random_state=SEED)
val,   test = train_test_split(temp,    test_size=0.50, random_state=SEED)
print(f'train={len(train)}  val={len(val)}  test={len(test)}')
```

### Step 5

Write each split as JSON Lines. JSONL is the fine-tuning interchange format: one JSON object per line, streamable, appendable and diff-friendly.

```python
for name, part in [('train',train),('validation',val),('test',test)]:
    part.to_json(f'data_repo/v1/processed/{name}.jsonl',
                 orient='records', lines=True, force_ascii=False)
!ls -la data_repo/v1/processed/
```

### Step 6

Content-hash every split. The SHA-256 is the dataset's true version identifier — if the hash changes, it is a different dataset, whatever the folder is called.

```python
import hashlib, json
def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
manifest = {'version':'v1','seed':SEED,'splits':{}}
for name in ['train','validation','test']:
    p = f'data_repo/v1/processed/{name}.jsonl'
    manifest['splits'][name] = {'rows': sum(1 for _ in open(p)), 'sha256': sha256(p)}
json.dump(manifest, open('data_repo/v1/manifest.json','w'), indent=2)
print(json.dumps(manifest, indent=2))
```

### Step 7

Write the dataset card. This is the governance artifact: what the data is, where it came from, what it may be used for and what is known to be wrong with it.

```python
card = '''# Dataset Card - Support Assistant v1

## Source
Mock customer-support ticket export (synthetic, no real PII).

## Schema
instruction (str), input (str), output (str)

## Splits
train / validation / test, 80/10/10, seed 3407

## Intended use
Supervised fine tuning of a small instruction model for support reply drafting.

## Known limitations
English only; single product domain; replies reflect one company style.
'''
open('data_repo/v1/DATASET_CARD.md','w').write(card)
```

### Step 8

Reload the repository through the datasets library to confirm it is consumable by the training stack — the real test of a data repository is that a trainer can read it unchanged.

```python
from datasets import load_dataset
ds = load_dataset('json', data_files={
    'train'     : 'data_repo/v1/processed/train.jsonl',
    'validation': 'data_repo/v1/processed/validation.jsonl',
    'test'      : 'data_repo/v1/processed/test.jsonl'})
print(ds)
print(ds['train'][0])
```

## Test it

data_repo/v1/ contains three JSONL splits whose row counts sum to the number of usable records, manifest.json records a SHA-256 for each split, DATASET_CARD.md exists, and load_dataset() returns a DatasetDict with all three splits.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0