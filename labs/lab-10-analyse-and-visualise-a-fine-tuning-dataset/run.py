#!/usr/bin/env python3
"""
Lab 10 — Analyse and Visualise a Fine-Tuning Dataset
Fine Tuning Open Source LLM (C922)
LU2 — Exploratory Data Analysis

Objective: Analyse and visualise dataset properties to drive fine-tuning decisions

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: profile_lengths.png, profile_balance.png and profile_umap.png are produced; the token-length percentile table yields a recommended max_seq_length; the eight clusters are each labelled with their top terms; and dataset_profile.json carries the numeric recommendations.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load the prepared dataset and the tokenizer. Token counts must
# come from the ACTUAL tokenizer — word counts are off by 30-50% for most
# models.
# --------------------------------------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from transformers import AutoTokenizer
df  = pd.read_json('data_repo/v2/processed/train.jsonl', lines=True)
tok = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
print(len(df), df.columns.tolist())

# --------------------------------------------------------------------------
# STEP 2. Compute true token lengths for prompt, response and total. These
# three distributions drive three different decisions.
# --------------------------------------------------------------------------
df['tok_in']  = df['input'].map(lambda t: len(tok(t)['input_ids']))
df['tok_out'] = df['output'].map(lambda t: len(tok(t)['input_ids']))
df['tok_all'] = df['tok_in'] + df['tok_out']
print(df[['tok_in','tok_out','tok_all']].describe().round(1).to_string())

# --------------------------------------------------------------------------
# STEP 3. Plot the length distributions and read off the percentiles that
# set max_seq_length. Choose the 99th percentile so you truncate almost
# nothing but do not pay for a huge padded context.
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1,3, figsize=(15,4))
for a, col, t in zip(ax, ['tok_in','tok_out','tok_all'],
                     ['Prompt','Response','Total']):
    a.hist(df[col], bins=40, color='#1F6FEB', alpha=0.8)
    a.axvline(df[col].quantile(0.95), color='#F59E0B', ls='--', label='p95')
    a.axvline(df[col].quantile(0.99), color='#DC2626', ls='--', label='p99')
    a.set_title(f'{t} tokens'); a.set_xlabel('tokens'); a.legend()
plt.tight_layout(); plt.savefig('profile_lengths.png', dpi=120)
p99 = int(df['tok_all'].quantile(0.99))
rec = int(np.ceil(p99/256)*256)
print(f'p99 total = {p99} tokens -> recommended max_seq_length = {rec}')

# --------------------------------------------------------------------------
# STEP 4. Quantify the cost of each max_seq_length choice. Attention cost is
# quadratic, so doubling the sequence length roughly quadruples the
# attention compute.
# --------------------------------------------------------------------------
for L in [512, 1024, 2048, 4096]:
    kept = (df['tok_all'] <= L).mean()
    print(f'max_seq_length={L:>5}: keeps {kept:6.1%} of examples, '
          f'relative attention cost {(L/512)**2:>5.1f}x')

# --------------------------------------------------------------------------
# STEP 5. Analyse task and source balance. An unbalanced mix means the model
# over-fits the majority task and under-performs on the rest.
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1,2, figsize=(13,4))
df['source'].value_counts().plot.barh(ax=ax[0], color='#10B981')
ax[0].set_title('Examples per source')
df['instruction'].value_counts().head(10).plot.barh(ax=ax[1], color='#7C3AED')
ax[1].set_title('Top 10 instruction types')
plt.tight_layout(); plt.savefig('profile_balance.png', dpi=120)
top = df['source'].value_counts(normalize=True).iloc[0]
print(f'Largest source is {top:.1%} of the data — '
      f"{'REBALANCE' if top > 0.6 else 'acceptable'}")

# --------------------------------------------------------------------------
# STEP 6. Measure vocabulary coverage and the type-token ratio. A very low
# TTR signals templated, repetitive data that will teach the model to be
# repetitive.
# --------------------------------------------------------------------------
words = ' '.join(df['output']).lower().split()
vocab = pd.Series(words).value_counts()
print(f'Total tokens      : {len(words):,}')
print(f'Unique vocabulary : {len(vocab):,}')
print(f'Type-token ratio  : {len(vocab)/len(words):.4f}')
print(f'Hapax legomena    : {(vocab==1).sum():,} ({(vocab==1).mean():.1%})')
print(vocab.head(15).to_string())

# --------------------------------------------------------------------------
# STEP 7. Embed every example and project to 2D with UMAP. The map is the
# fastest way to see topic clusters, outliers and — most usefully — the
# empty regions.
# --------------------------------------------------------------------------
os.system('pip install -q umap-learn sentence-transformers')
from sentence_transformers import SentenceTransformer
import umap
enc = SentenceTransformer('all-MiniLM-L6-v2')
emb = enc.encode(df['input'].tolist(), batch_size=64,
                 normalize_embeddings=True, show_progress_bar=True)
xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=3407).fit_transform(emb)
print('Projected:', xy.shape)

# --------------------------------------------------------------------------
# STEP 8. Plot the semantic map coloured by source. Dense blobs are
# redundancy; sparse regions are the coverage gaps you should fill with
# synthetic data.
# --------------------------------------------------------------------------
plt.figure(figsize=(9,7))
for src, colour in zip(df['source'].unique(), ['#1F6FEB','#10B981','#7C3AED','#F59E0B']):
    m = (df['source'] == src).values
    plt.scatter(xy[m,0], xy[m,1], s=8, alpha=0.6, label=src, c=colour)
plt.legend(); plt.title('Semantic map of the fine-tuning corpus (UMAP)')
plt.savefig('profile_umap.png', dpi=120, bbox_inches='tight')

# --------------------------------------------------------------------------
# STEP 9. Cluster the embedding space and name each cluster by its most
# distinctive words, to turn the picture into an actionable topic inventory.
# --------------------------------------------------------------------------
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
df['cluster'] = KMeans(n_clusters=8, random_state=3407, n_init=10).fit_predict(emb)
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X = tfidf.fit_transform(df['input']); terms = np.array(tfidf.get_feature_names_out())
for c in range(8):
    m = (df['cluster'] == c).values
    top = terms[np.asarray(X[m].mean(axis=0)).ravel().argsort()[-6:][::-1]]
    print(f'Cluster {c} (n={m.sum():>4}): {", ".join(top)}')

# --------------------------------------------------------------------------
# STEP 10. Write the profile report with concrete, numeric recommendations —
# the deliverable that the training lab consumes.
# --------------------------------------------------------------------------
import json
profile = {
 'rows': len(df),
 'recommended_max_seq_length': rec,
 'p95_tokens': int(df['tok_all'].quantile(0.95)),
 'p99_tokens': p99,
 'largest_source_share': round(float(top),3),
 'type_token_ratio': round(len(vocab)/len(words),4),
 'clusters': int(df['cluster'].nunique()),
 'recommendations': [
   f'Set max_seq_length={rec} (p99 is {p99}).',
   'Rebalance sources if any exceeds 60% of the corpus.',
   'Generate synthetic data for the sparse UMAP regions.',
   'Investigate any cluster with fewer than 20 examples as a coverage gap.'],
}
json.dump(profile, open('dataset_profile.json','w'), indent=2)
print(json.dumps(profile, indent=2))


print('\nLab 10 complete — profile_lengths.png, profile_balance.png and profile_umap.png are produced; the token-length percentile table ')
