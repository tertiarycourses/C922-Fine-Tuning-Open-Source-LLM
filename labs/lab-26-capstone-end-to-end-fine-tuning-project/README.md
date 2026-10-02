# Lab 26 — Capstone — End-to-End Fine-Tuning Project

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU4 — Machine Learning Implementation and Operations  
**Delivery:** self-paced extension  
**Objective:** Deliver a complete fine-tuning solution from data to deployment with governance

## Goal

Put the whole course together on a new domain: build the dataset, fine-tune with a justified configuration, benchmark against the base model, audit for bias, and deliver the model card and deployment artifacts as one reviewable package.

## What you will build

A complete, reviewable fine-tuning project: versioned dataset, trained adapter, benchmark report, fairness audit, model card and deployment export.

**Tools:** The full course stack — datasets, Unsloth, PEFT, TRL, evaluate, matplotlib

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-26-capstone-end-to-end-fine-tuning-project/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

**Mock data provided:**

- `data/capstone_raw.jsonl`

## Step-by-step

### Step 1

Choose the capstone domain and write the problem statement. Pick the domain closest to your own work so the result is useful beyond the classroom.

```python
PROJECT = {
 'domain'    : 'Clinic appointment and enquiry assistant',   # or your own domain
 'need'      : 'Draft replies to patient enquiries in the clinic house style.',
 'baseline'  : 'Staff write every reply manually.',
 'success'   : 'ROUGE-L gain over base > 10%, no invented clinical advice.',
 'constraint': 'Single 16GB GPU; no patient data leaves the country.',
}
import json; print(json.dumps(PROJECT, indent=2))
```

### Step 2

STAGE 1 — build the dataset. Reuse the LU1 and LU2 pipeline: ingest, generate synthetic data where real data is thin, clean, and version it.

```python
import pandas as pd
raw = pd.read_json('data/capstone_raw.jsonl', lines=True)
print(f'Raw records: {len(raw)}')
print(raw.head(3).to_string())
print('Now apply Lab 7 sanitisation and Lab 9 ETL to this corpus.')
```

### Step 3

Run the ETL with the quality gate from Lab 9, and stop the project here if the gate fails. A failed gate is a finding, not an obstacle to work around.

```python
import re, hashlib, os
PII = {'EMAIL': r'[\w.+-]+@[\w-]+\.[\w.]+',
       'PHONE': r'\b(?:\+65[ -]?)?[689]\d{3}[ -]?\d{4}\b',
       'NRIC' : r'\b[STFG]\d{7}[A-Z]\b'}
df = raw.copy()
for tag, p in PII.items():
    for c in ['input','output']:
        df[c] = df[c].astype(str).str.replace(p, f'[{tag}]', regex=True)
df = df.drop_duplicates(subset=['input','output'])
df = df[df['output'].str.len() >= 20]
assert len(df) >= 100, 'QUALITY GATE FAILED: fewer than 100 usable rows'
os.makedirs('capstone/data', exist_ok=True)
split = df.sample(frac=1, random_state=3407)
n = int(0.85*len(split))
split[:n].to_json('capstone/data/train.jsonl', orient='records', lines=True, force_ascii=False)
split[n:].to_json('capstone/data/test.jsonl',  orient='records', lines=True, force_ascii=False)
print(f'Gate PASSED. train={n} test={len(split)-n}')
```

### Step 4

STAGE 2 — profile the dataset (Lab 10) and choose max_seq_length from the measured percentiles rather than a guess.

```python
from unsloth import FastLanguageModel
import numpy as np
model, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=2048, load_in_4bit=True)
tr = pd.read_json('capstone/data/train.jsonl', lines=True)
lens = [len(tok(f"{r['input']} {r['output']}")['input_ids']) for _, r in tr.iterrows()]
p99 = int(np.percentile(lens, 99))
MAXLEN = int(np.ceil(p99/256)*256)
print(f'p50={int(np.percentile(lens,50))} p95={int(np.percentile(lens,95))} p99={p99}')
print(f'Chosen max_seq_length = {MAXLEN}')
```

### Step 5

STAGE 3 — fine-tune with the configuration justified in Lab 19. Loading it from the file rather than retyping it is what keeps the project traceable.

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth import is_bfloat16_supported
hp = json.load(open('best_hyperparameters.json'))
print('Using justified configuration:', json.dumps(hp, indent=2))
model = FastLanguageModel.get_peft_model(model,
    r=hp['r'], lora_alpha=hp['lora_alpha'], lora_dropout=0, bias='none',
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    use_gradient_checkpointing='unsloth', random_state=3407)
ds = load_dataset('json', data_files='capstone/data/train.jsonl')['train']
def fmt(b):
    return {'text': [tok.apply_chat_template(
        [{'role':'user','content':i},{'role':'assistant','content':o}], tokenize=False)
        for i, o in zip(b['input'], b['output'])]}
ds = ds.map(fmt, batched=True)
print(ds[0]['text'][:300])
```

### Step 6

Train, monitoring with the Lab 18 diagnostics so a divergent or overfitting run is caught while it happens rather than afterwards.

```python
import torch
args = SFTConfig(output_dir='capstone/out',
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    num_train_epochs=hp['num_train_epochs'], learning_rate=hp['learning_rate'],
    warmup_steps=5, logging_steps=5, optim='adamw_8bit',
    lr_scheduler_type='linear', weight_decay=0.01, max_grad_norm=1.0, seed=3407,
    bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
    max_seq_length=MAXLEN, dataset_text_field='text', report_to='none')
torch.cuda.reset_peak_memory_stats()
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=args)
stats = trainer.train()
print(f"Runtime {stats.metrics['train_runtime']/60:.1f} min, "
      f"loss {stats.metrics['train_loss']:.4f}, "
      f"peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

### Step 7

STAGE 4 — benchmark against the base model on the held-out set (Lab 21), with a fixed decoder and the catastrophic-forgetting check.

```bash
!pip install -q evaluate rouge_score
import evaluate
rouge = evaluate.load('rouge')
test = pd.read_json('capstone/data/test.jsonl', lines=True)
FastLanguageModel.for_inference(model)
def gen(p):
    ids = tok.apply_chat_template([{'role':'user','content':p}],
            add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = model.generate(ids, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
preds = [gen(r['input']) for _, r in test.iterrows()]
sc = rouge.compute(predictions=preds, references=test['output'].tolist())
print({k: round(v,4) for k,v in sc.items()})
```

### Step 8

STAGE 5 — audit for bias (Lab 22) and record the result, whatever it says. An audit you only publish when it passes is not an audit.

```python
from scipy import stats as st
NAMES = ['Wei Ming','Hui Ling','Ahmad Rizal','Siti Nurul','Rajesh','Priya']
probe = [f'Draft a reply to {n}, who is asking to reschedule an appointment.' for n in NAMES]
outs  = [gen(p) for p in probe]
lens  = [len(o.split()) for o in outs]
for n, l, o in zip(NAMES, lens, outs):
    print(f'{n:<12} {l:>4} words | {o[:80]}')
spread = (max(lens)-min(lens))/max(np.mean(lens),1)
print(f'\nRelative length spread across names: {spread:.1%} '
      f"-> {'INVESTIGATE' if spread > 0.30 else 'acceptable'}")
```

### Step 9

STAGE 6 — export and write the project deliverable. The package, not the notebook, is the deliverable.

```python
model.save_pretrained('capstone/adapter'); tok.save_pretrained('capstone/adapter')
model.save_pretrained_merged('capstone/merged_16bit', tok, save_method='merged_16bit')
deliverable = {**PROJECT,
 'dataset'      : {'train': int(n), 'test': int(len(split)-n), 'max_seq_length': MAXLEN},
 'method'       : f"QLoRA via Unsloth, r={hp['r']}, alpha={hp['lora_alpha']}, lr={hp['learning_rate']}",
 'training'     : {'minutes': round(stats.metrics['train_runtime']/60,2),
                   'final_loss': round(stats.metrics['train_loss'],4),
                   'peak_vram_gb': round(torch.cuda.max_memory_allocated()/1024**3,2)},
 'benchmark'    : {k: round(v,4) for k,v in sc.items()},
 'fairness'     : {'name_length_spread': round(float(spread),4)},
 'artifacts'    : ['capstone/data/','capstone/adapter/','capstone/merged_16bit/',
                   'MODEL_CARD.md','benchmark_report.json','fairness_report.json'],
}
json.dump(deliverable, open('capstone/PROJECT_REPORT.json','w'), indent=2)
print(json.dumps(deliverable, indent=2))
```

### Step 10

Present your project. Explain each decision you made and what evidence supports it — this is exactly the reasoning a real fine-tuning project review asks for.

```python
print('Be ready to explain and defend:')
for q in ['Why fine tuning rather than prompting or RAG for this problem?',
          'How did you build the dataset, and what did the quality gate reject?',
          'Why this rank, alpha, learning rate and target-module set?',
          'How did you know training was stable and had converged?',
          'What is the measured gain over the base model, and on what held-out data?',
          'What bias did you test for, and what did you find?',
          'Which deployment target did you choose, and why?',
          'What are the known limitations recorded in the model card?']:
    print('  -', q)
```

## Test it

capstone/PROJECT_REPORT.json records the dataset sizes, the justified method, the training metrics, the benchmark scores and the fairness spread; capstone/adapter/ and capstone/merged_16bit/ both exist; and the quality-gate assertion passed.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0