# Lab 9 — Build an ETL Pipeline for Fine-Tuning Datasets

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU2 — Exploratory Data Analysis  
**Delivery:** self-paced extension  
**Objective:** Develop an extract-transform-load pipeline that produces a validated fine-tuning dataset

## Goal

Assemble the previous labs' work into one orchestrated, re-runnable ETL pipeline with validation gates between stages, structured logging, failure handling and an output contract — the form a data pipeline must take before it can run in production.

## What you will build

A single run_pipeline.py that goes from raw sources to a versioned, validated, split dataset, with a quality gate that fails the run rather than shipping a bad dataset.

**Tools:** Hugging Face datasets, pandas, Python logging, JSON Schema

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-09-build-an-etl-pipeline-for-fine-tuning-datasets/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

**Mock data provided:**

- `data/source_a_faq.csv`
- `data/source_b_tickets.jsonl`

## Step-by-step

### Step 1

Define the output CONTRACT first. A pipeline without a declared contract cannot be validated, and every downstream consumer breaks silently when the shape drifts.

```python
SCHEMA = {
  'required_columns': ['instruction','input','output','source'],
  'min_rows'        : 100,
  'max_dupe_rate'   : 0.02,
  'min_output_chars': 20,
  'max_pii_hits'    : 0,
}
import json; print(json.dumps(SCHEMA, indent=2))
```

### Step 2

Set up structured logging. Every stage logs rows in, rows out and duration — without this you cannot tell which stage dropped your data.

```python
import logging, time
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s')
log = logging.getLogger('etl')
STATS = []
def stage(name):
    def deco(fn):
        def wrap(df, *a, **k):
            t0, n0 = time.time(), len(df)
            out = fn(df, *a, **k)
            rec = {'stage':name,'in':n0,'out':len(out),
                   'dropped':n0-len(out),'secs':round(time.time()-t0,2)}
            STATS.append(rec); log.info(rec)
            return out
        return wrap
    return deco
```

### Step 3

EXTRACT — load every source and tag provenance. Extraction must be side-effect free so the pipeline can be re-run safely.

```python
import pandas as pd
from datasets import load_dataset
def extract():
    frames = []
    a = pd.read_csv('data/source_a_faq.csv')
    frames.append(pd.DataFrame({'instruction':'Answer the customer question.',
                   'input':a['question'],'output':a['answer'],'source':'faq_csv'}))
    b = pd.read_json('data/source_b_tickets.jsonl', lines=True)
    frames.append(pd.DataFrame({'instruction':b.get('task','Draft a support reply.'),
                   'input':b['customer_message'],'output':b['agent_reply'],
                   'source':'tickets_jsonl'}))
    df = pd.concat(frames, ignore_index=True)
    log.info(f'EXTRACT: {len(df)} rows from {df["source"].nunique()} sources')
    return df
raw = extract()
```

### Step 4

TRANSFORM stage 1 — normalise and clean. Each transform stage is decorated so its row counts are captured automatically.

```python
import re, unicodedata
@stage('normalise')
def t_normalise(df):
    for c in ['instruction','input','output']:
        df[c] = df[c].astype(str).map(
            lambda t: re.sub(r'\s+',' ', unicodedata.normalize('NFKC', t)).strip())
    return df
@stage('drop_invalid')
def t_valid(df):
    return df[(df['output'].str.len() >= SCHEMA['min_output_chars']) &
              (df['input'].str.len() >= 5)]
df = t_valid(t_normalise(raw))
```

### Step 5

TRANSFORM stage 2 — redact PII and de-duplicate. These are the two stages whose omission causes the most damage downstream.

```python
PII = {'EMAIL': r'[\w.+-]+@[\w-]+\.[\w.]+',
       'PHONE': r'\b(?:\+65[ -]?)?[689]\d{3}[ -]?\d{4}\b',
       'NRIC' : r'\b[STFG]\d{7}[A-Z]\b'}
@stage('redact_pii')
def t_pii(df):
    for tag, p in PII.items():
        for c in ['input','output']:
            df[c] = df[c].str.replace(p, f'[{tag}]', regex=True)
    return df
@stage('dedupe')
def t_dedupe(df):
    return df.drop_duplicates(subset=['input','output'])
df = t_dedupe(t_pii(df))
```

### Step 6

The QUALITY GATE. This is what separates a pipeline from a script: it raises and stops the run rather than loading a dataset that violates the contract.

```python
def quality_gate(df):
    errs = []
    missing = set(SCHEMA['required_columns']) - set(df.columns)
    if missing:                 errs.append(f'missing columns: {missing}')
    if len(df) < SCHEMA['min_rows']:
        errs.append(f'too few rows: {len(df)} < {SCHEMA["min_rows"]}')
    dupe = df.duplicated(subset=['input','output']).mean()
    if dupe > SCHEMA['max_dupe_rate']:
        errs.append(f'dupe rate {dupe:.3f} > {SCHEMA["max_dupe_rate"]}')
    hits = sum(df[c].str.contains(p, regex=True).sum()
               for p in PII.values() for c in ['input','output'])
    if hits > SCHEMA['max_pii_hits']:
        errs.append(f'{hits} PII patterns survived redaction')
    if errs:
        raise ValueError('QUALITY GATE FAILED:\n  - ' + '\n  - '.join(errs))
    log.info('QUALITY GATE PASSED')
    return True
quality_gate(df)
```

### Step 7

Deliberately break the gate to prove it works. A gate you have never seen fire is a gate you cannot trust.

```python
bad = df.copy()
bad.loc[bad.index[:5], 'output'] = 'contact me at test@example.com'
try:
    quality_gate(bad)
except ValueError as e:
    print('Gate correctly rejected the bad dataset:')
    print(e)
```

### Step 8

LOAD — split, version and write with a content hash. The load stage produces the immutable artifact the training job consumes.

```python
import hashlib, os
from sklearn.model_selection import train_test_split
def load_stage(df, version='v2'):
    out = f'data_repo/{version}/processed'; os.makedirs(out, exist_ok=True)
    tr, tmp = train_test_split(df, test_size=0.2, random_state=3407)
    va, te  = train_test_split(tmp, test_size=0.5, random_state=3407)
    manifest = {'version':version,'seed':3407,'stages':STATS,'splits':{}}
    for name, part in [('train',tr),('validation',va),('test',te)]:
        p = f'{out}/{name}.jsonl'
        part.to_json(p, orient='records', lines=True, force_ascii=False)
        manifest['splits'][name] = {
            'rows': len(part),
            'sha256': hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]}
    json.dump(manifest, open(f'data_repo/{version}/manifest.json','w'), indent=2)
    return manifest
manifest = load_stage(df)
print(json.dumps(manifest, indent=2))
```

### Step 9

Print the end-to-end stage report. This table is the operational view of the pipeline and the first thing you read when a run produces the wrong row count.

```python
rep = pd.DataFrame(STATS)
rep['retained_%'] = (100*rep['out']/rep['in']).round(1)
print(rep.to_string(index=False))
print(f"\nEnd-to-end: {rep['in'].iloc[0]} -> {rep['out'].iloc[-1]} rows "
      f"({100*rep['out'].iloc[-1]/rep['in'].iloc[0]:.1f}% retained)")
```

## Test it

The pipeline runs end to end, the quality gate passes on the clean data and raises ValueError on the deliberately poisoned copy, data_repo/v2/manifest.json records per-stage statistics and a hash per split, and the stage report shows retention at every stage.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0