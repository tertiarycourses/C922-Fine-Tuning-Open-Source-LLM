# Lab 7 — Clean and Sanitize a Fine-Tuning Dataset

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU2 — Exploratory Data Analysis  
**Delivery:** in class  
**Objective:** Clean, sanitize and prepare a raw corpus so the fine-tune learns signal rather than noise

## Goal

Take a deliberately dirty corpus and run the full sanitisation gauntlet — PII redaction, exact and near-duplicate removal, truncation and encoding repair, prompt-injection screening and quality scoring — measuring how many rows each stage removes and why.

## What you will build

A cleaned dataset plus a cleaning report that records, stage by stage, how many rows were removed and for what reason — the audit trail a data pipeline must produce.

**Tools:** pandas, regex, datasketch MinHash/LSH, ftfy, Hugging Face datasets

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-07-clean-and-sanitize-a-fine-tuning-dataset/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

**Mock data provided:**

- `data/dirty_corpus.jsonl`

## Step-by-step

### Step 1

Load the dirty corpus and take a first look at the damage. Never clean blind — measure what is wrong before you decide what to remove.

```python
import pandas as pd, re, json
df = pd.read_json('data/dirty_corpus.jsonl', lines=True)
print('Rows:', len(df))
print(df.isna().sum())
print('Empty outputs :', (df['output'].str.strip()=='').sum())
print('Exact dupes   :', df.duplicated(subset=['input','output']).sum())
report = {'initial_rows': len(df)}
```

### Step 2

Repair mojibake and normalise unicode. Encoding damage turns into learned garbage tokens if it reaches the trainer.

```bash
!pip install -q ftfy
import ftfy, unicodedata
def fix_text(t):
    t = ftfy.fix_text(str(t))
    t = unicodedata.normalize('NFKC', t)
    return re.sub(r'\s+', ' ', t).strip()
for col in ['instruction','input','output']:
    df[col] = df[col].map(fix_text)
print(df['input'].head(3).tolist())
```

### Step 3

Redact PII. Emails, phone numbers, NRIC/FIN and card numbers must never enter a training set — a fine-tuned model memorises and will regurgitate them.

```python
PII = {
 'EMAIL' : r'[\w.+-]+@[\w-]+\.[\w.]+',
 'PHONE' : r'\b(?:\+65[ -]?)?[689]\d{3}[ -]?\d{4}\b',
 'NRIC'  : r'\b[STFG]\d{7}[A-Z]\b',
 'CARD'  : r'\b(?:\d{4}[ -]?){3}\d{4}\b',
}
counts = {k:0 for k in PII}
def redact(t):
    for tag, pat in PII.items():
        t, n = re.subn(pat, f'[{tag}]', t)
        counts[tag] += n
    return t
for col in ['instruction','input','output']:
    df[col] = df[col].map(redact)
print('PII redacted:', counts)
report['pii_redacted'] = counts
```

### Step 4

Drop structurally invalid rows — empty fields, and outputs so short they carry no signal.

```python
before = len(df)
df = df[(df['output'].str.len() >= 20) & (df['input'].str.len() >= 5)]
df = df[df['instruction'].str.strip() != '']
report['dropped_too_short'] = before - len(df)
print(f'Dropped {before-len(df)} structurally invalid rows -> {len(df)} remain')
```

### Step 5

Remove exact duplicates. Duplicated examples are implicit over-weighting: the model sees them more often and over-fits to them.

```python
before = len(df)
df = df.drop_duplicates(subset=['input','output'])
report['dropped_exact_dupes'] = before - len(df)
print(f'Dropped {before-len(df)} exact duplicates -> {len(df)} remain')
```

### Step 6

Remove NEAR duplicates with MinHash + LSH. Paraphrased copies survive exact hashing and are the main cause of train/test leakage.

```bash
!pip install -q datasketch
from datasketch import MinHash, MinHashLSH
def mh(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for tok in set(text.lower().split()):
        m.update(tok.encode('utf8'))
    return m
lsh, keep, seen = MinHashLSH(threshold=0.85, num_perm=128), [], 0
for i, row in df.reset_index(drop=True).iterrows():
    sig = mh(row['input'] + ' ' + row['output'])
    if lsh.query(sig):
        seen += 1
        continue
    lsh.insert(str(i), sig); keep.append(i)
df = df.reset_index(drop=True).iloc[keep]
report['dropped_near_dupes'] = seen
print(f'Dropped {seen} near-duplicates (Jaccard >= 0.85) -> {len(df)} remain')
```

### Step 7

Screen for prompt-injection and jailbreak payloads. A poisoned example teaches the model to obey the injection — this is a supply-chain attack on your fine-tune.

```python
INJECTION = [r'ignore (all )?previous instructions', r'disregard (the )?above',
             r'you are now (in )?(DAN|developer mode)', r'system prompt\s*[:=]',
             r'reveal your (system )?(prompt|instructions)']
pat = re.compile('|'.join(INJECTION), re.I)
flag = df.apply(lambda r: bool(pat.search(r['input'] + ' ' + r['output'])), axis=1)
report['dropped_injection'] = int(flag.sum())
print('Injection payloads found:', flag.sum())
print(df[flag]['input'].head(3).tolist())
df = df[~flag]
```

### Step 8

Score the remaining examples for quality and keep the top band. Quality beats quantity: LoRA has few parameters and will faithfully learn whatever you give it.

```python
def quality(r):
    s, out = 0, r['output']
    if 50 <= len(out) <= 2000:            s += 2
    if out[0].isupper():                   s += 1
    if out.rstrip()[-1] in '.!?':          s += 1
    if not re.search(r'(.)\1{4,}', out):   s += 1
    if len(set(out.split())) / max(len(out.split()),1) > 0.4: s += 1
    return s
df['quality'] = df.apply(quality, axis=1)
print(df['quality'].value_counts().sort_index())
before = len(df)
df = df[df['quality'] >= 4]
report['dropped_low_quality'] = before - len(df)
```

### Step 9

Write the cleaned dataset and the cleaning report. The report is what makes the pipeline auditable — every removed row is accounted for.

```python
report['final_rows'] = len(df)
report['retention_pct'] = round(100*len(df)/report['initial_rows'], 1)
df.drop(columns=['quality']).to_json('cleaned_corpus.jsonl',
        orient='records', lines=True, force_ascii=False)
json.dump(report, open('cleaning_report.json','w'), indent=2)
print(json.dumps(report, indent=2))
```

## Test it

cleaned_corpus.jsonl exists, cleaning_report.json accounts for every removed row by stage, no email/phone/NRIC pattern survives a re-scan of the cleaned file, and retention_pct is reported.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0