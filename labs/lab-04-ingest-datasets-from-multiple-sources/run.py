#!/usr/bin/env python3
"""
Lab 4 — Ingest Datasets from Multiple Sources
Fine Tuning Open Source LLM (C922)
LU1 — Data Engineering

Objective: Implement a dataset ingestion solution that loads, streams and merges heterogeneous sources

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: The merged corpus has exactly the four canonical columns, Counter(corpus['source']) shows rows from all three batch sources, the streaming demo prints three rows without a full download, and ingested_train.jsonl / ingested_test.jsonl are written.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load source A — a local CSV. The `datasets` library reads CSV
# directly into an Arrow-backed Dataset, which is memory-mapped rather than
# held in RAM.
# --------------------------------------------------------------------------
from datasets import load_dataset, Dataset, concatenate_datasets
src_a = load_dataset('csv', data_files='data/source_a_faq.csv')['train']
print(src_a)
print(src_a[0])

# --------------------------------------------------------------------------
# STEP 2. Load source B — local JSON Lines with a different column naming
# convention. Real pipelines always meet inconsistent schemas; the ingestion
# layer is where they are reconciled.
# --------------------------------------------------------------------------
src_b = load_dataset('json', data_files='data/source_b_tickets.jsonl')['train']
print(src_b.column_names)
print(src_b[0])

# --------------------------------------------------------------------------
# STEP 3. Load source C — a public dataset from the Hugging Face Hub, taking
# only a slice. Slicing keeps the lab fast and demonstrates the split syntax
# you use to sample a large corpus.
# --------------------------------------------------------------------------
src_c = load_dataset('tatsu-lab/alpaca', split='train[:200]')
print(src_c)
print(src_c[0])

# --------------------------------------------------------------------------
# STEP 4. Load source D in STREAMING mode. A streaming dataset yields rows
# lazily over the network and never materialises the full corpus, which is
# how you ingest data larger than your disk.
# --------------------------------------------------------------------------
stream = load_dataset('tatsu-lab/alpaca', split='train', streaming=True)
import itertools
for row in itertools.islice(stream, 3):
    print({k: str(v)[:60] for k, v in row.items()})
print('Streaming works - nothing was downloaded in full.')

# --------------------------------------------------------------------------
# STEP 5. Define one canonical target schema and a normaliser per source.
# Every source is mapped into instruction / input / output, and tagged with
# its origin so provenance survives the merge.
# --------------------------------------------------------------------------
TARGET = ['instruction','input','output','source']
def norm_a(r): return {'instruction':'Answer the customer question.',
                       'input': r['question'], 'output': r['answer'], 'source':'faq_csv'}
def norm_b(r): return {'instruction': r.get('task') or 'Draft a support reply.',
                       'input': r['customer_message'], 'output': r['agent_reply'],
                       'source':'tickets_jsonl'}
def norm_c(r): return {'instruction': r['instruction'],
                       'input': r.get('input',''), 'output': r['output'], 'source':'alpaca_hub'}

# --------------------------------------------------------------------------
# STEP 6. Apply each normaliser with .map(), removing the original columns.
# batched=False keeps the mapping simple to read; remove_columns is what
# guarantees a clean uniform schema.
# --------------------------------------------------------------------------
a = src_a.map(norm_a, remove_columns=src_a.column_names)
b = src_b.map(norm_b, remove_columns=src_b.column_names)
c = src_c.map(norm_c, remove_columns=src_c.column_names)
for name, d in [('A',a),('B',b),('C',c)]:
    print(name, d.column_names, len(d))

# --------------------------------------------------------------------------
# STEP 7. Concatenate into one corpus and shuffle with a fixed seed.
# Shuffling matters: an unshuffled concatenation trains on all of source A,
# then all of B, which destabilises the loss.
# --------------------------------------------------------------------------
corpus = concatenate_datasets([a,b,c]).shuffle(seed=3407)
print(corpus)
from collections import Counter
print(Counter(corpus['source']))

# --------------------------------------------------------------------------
# STEP 8. Validate the ingested corpus: no empty outputs, no missing
# instructions, and every row carries provenance. An ingestion stage that
# does not validate is an ingestion stage that ships bad rows.
# --------------------------------------------------------------------------
def valid(r):
    return bool(r['instruction'].strip()) and bool(r['output'].strip()) and bool(r['source'])
before = len(corpus)
corpus = corpus.filter(valid)
print(f'Dropped {before-len(corpus)} invalid rows; {len(corpus)} remain.')

# --------------------------------------------------------------------------
# STEP 9. Split and save the unified corpus so the transformation lab can
# pick it up unchanged.
# --------------------------------------------------------------------------
split = corpus.train_test_split(test_size=0.1, seed=3407)
split['train'].to_json('ingested_train.jsonl', orient='records', lines=True, force_ascii=False)
split['test'].to_json('ingested_test.jsonl',  orient='records', lines=True, force_ascii=False)
print(split)


print('\nLab 4 complete — The merged corpus has exactly the four canonical columns, Counter(corpus[\'source\']) shows rows from all thre')
