# Lab 3 — Generate Synthetic Training Data

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU1 — Data Engineering  
**Delivery:** self-paced extension  
**Objective:** Generate synthetic fine-tuning examples that preserve the statistical properties of real data

## Goal

Real data is scarce and privacy-restricted, so you generate synthetic examples using three different methods — template expansion, self-instruct bootstrapping and programmatic augmentation — then prove statistically that the synthetic set matches the real distribution.

## What you will build

A synthetic dataset of several hundred examples produced by three methods, plus a distribution comparison report showing length, label-balance and vocabulary overlap against the real seed set.

**Tools:** Hugging Face datasets, pandas, numpy, scipy, matplotlib

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-03-generate-synthetic-training-data/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

**Mock data provided:**

- `data/seed_examples.jsonl`

## Step-by-step

### Step 1

Load the small real seed set. This is the ground truth whose statistical properties the synthetic data must reproduce.

```python
import pandas as pd, numpy as np
seed_df = pd.read_json('data/seed_examples.jsonl', lines=True)
print('Seed examples:', len(seed_df))
print(seed_df.head(3).to_string())
```

### Step 2

Method 1 — template expansion. Combine slot values into a grammar of templates. This is fast, fully controllable and guarantees coverage of every category you enumerate.

```python
import itertools, random
random.seed(3407)
TEMPLATES = [
  'My {product} stopped working after the {event}. What should I do?',
  'I was charged twice for my {product} {plan} plan. Can you fix this?',
  'How do I {action} my {product} account?',
]
SLOTS = {'product':['router','mobile plan','fibre broadband','SIM card'],
         'event'  :['firmware update','power outage','plan upgrade'],
         'plan'   :['basic','premium','enterprise'],
         'action' :['cancel','upgrade','transfer','suspend']}
def expand(t):
    out = t
    for k, vals in SLOTS.items():
        out = out.replace('{'+k+'}', random.choice(vals))
    return out
template_data = [expand(random.choice(TEMPLATES)) for _ in range(150)]
print(template_data[:3])
```

### Step 3

Method 2 — self-instruct bootstrapping. Sample real examples as few-shot exemplars and ask a teacher model to write NEW instructions in the same style. This produces far more diversity than templates, at the cost of needing a generator model.

```python
from unsloth import FastLanguageModel
import torch
gen, gen_tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(gen)
def self_instruct(exemplars, n=3):
    shots  = '\n'.join(f'- {e}' for e in exemplars)
    prompt = (f'Here are real customer support questions:\n{shots}\n\n'
              f'Write {n} NEW questions in the same style and domain. One per line.')
    msgs   = [{'role':'user','content':prompt}]
    ids    = gen_tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors='pt').to('cuda')
    out    = gen.generate(ids, max_new_tokens=160, temperature=0.9, do_sample=True)
    text   = gen_tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return [l.strip('-* ').strip() for l in text.split('\n') if len(l.strip()) > 15]
bootstrap = []
for _ in range(20):
    bootstrap += self_instruct(seed_df['input'].sample(3).tolist())
print(f'Bootstrapped {len(bootstrap)} examples'); print(bootstrap[:3])
```

### Step 4

Method 3 — programmatic augmentation. Perturb real examples with controlled noise so the model learns to be robust to the typos and phrasing variation it will meet in production.

```python
def augment(text):
    ops = [lambda t: t.lower(),
           lambda t: t.replace('cannot','cant').replace('do not','dont'),
           lambda t: 'Hi, ' + t,
           lambda t: t + ' Please advise urgently.']
    return random.choice(ops)(text)
augmented = [augment(t) for t in seed_df['input'].tolist() for _ in range(2)]
print(f'Augmented {len(augmented)} examples')
```

### Step 5

De-duplicate the pooled synthetic set against itself AND against the real data. Near-copies of the seed set add no information and leak the evaluation examples into training.

```python
synthetic = list(dict.fromkeys(template_data + bootstrap + augmented))
real_set  = set(seed_df['input'].str.lower().str.strip())
synthetic = [s for s in synthetic if s.lower().strip() not in real_set]
print(f'Unique synthetic examples after dedup: {len(synthetic)}')
```

### Step 6

Compare the distributions. Matching mean length, length variance and vocabulary overlap is the evidence that the synthetic data preserves the real statistical properties.

```python
from scipy import stats
real_len = seed_df['input'].str.split().str.len()
syn_len  = pd.Series(synthetic).str.split().str.len()
print(f'Real  : mean={real_len.mean():.1f}  std={real_len.std():.1f}')
print(f'Synth : mean={syn_len.mean():.1f}  std={syn_len.std():.1f}')
ks = stats.ks_2samp(real_len, syn_len)
print(f'KS statistic={ks.statistic:.3f}  p={ks.pvalue:.3f}')
print('Distributions match (p>0.05):', ks.pvalue > 0.05)
```

### Step 7

Measure vocabulary overlap. Too little overlap means the synthetic data is off-domain; near-total overlap means it is merely a paraphrase of the seed and adds no new coverage.

```python
rv = set(' '.join(seed_df['input']).lower().split())
sv = set(' '.join(synthetic).lower().split())
print(f'Vocabulary overlap : {len(rv & sv)/len(rv):.1%} of real vocabulary covered')
print(f'New vocabulary     : {len(sv - rv)} tokens not in the seed set')
```

### Step 8

Plot the two length distributions and save the comparison report. A visual check catches bimodality and truncation that a single summary statistic hides.

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.hist(real_len, bins=20, alpha=0.6, label='Real', density=True)
plt.hist(syn_len,  bins=20, alpha=0.6, label='Synthetic', density=True)
plt.xlabel('Length (words)'); plt.ylabel('Density'); plt.legend()
plt.title('Real vs Synthetic length distribution')
plt.savefig('synthetic_distribution.png', dpi=120, bbox_inches='tight')
pd.DataFrame({'input':synthetic}).to_json('synthetic_examples.jsonl',
                                          orient='records', lines=True)
```

## Test it

synthetic_examples.jsonl contains several hundred de-duplicated examples produced by all three methods, the KS test p-value is reported, vocabulary overlap is between roughly 40% and 95%, and synthetic_distribution.png shows the two length histograms overlapping.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0