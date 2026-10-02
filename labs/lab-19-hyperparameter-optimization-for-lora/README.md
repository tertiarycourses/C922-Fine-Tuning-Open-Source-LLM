# Lab 19 — Hyperparameter Optimization for LoRA

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU3 — Modelling  
**Delivery:** self-paced extension  
**Objective:** Perform hyperparameter optimization and select a configuration on measured evidence

## Goal

Run a systematic hyperparameter search in the correct priority order — learning rate, then rank and alpha, then target modules, then epochs and effective batch size — measuring validation loss for each configuration and selecting the winner on evidence.

## What you will build

A hyperparameter sweep table with validation loss per configuration, sensitivity plots, and a selected configuration saved as best_hyperparameters.json.

**Tools:** Unsloth, TRL SFTTrainer, pandas, matplotlib, numpy

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-19-hyperparameter-optimization-for-lora/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

> **Input:** this lab has no mock dataset of its own — it consumes the output of an earlier lab in the same Learning Unit. Run the labs in order.

## Step-by-step

### Step 1

Build a reusable one-configuration trial function. Every trial must be identical apart from the parameter under test, or the comparison means nothing.

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch, gc, time, pandas as pd, numpy as np, json
DS = load_dataset('json', data_files='formatted_train.jsonl')['train']\
     .train_test_split(test_size=0.15, seed=3407)
def trial(lr=2e-4, r=16, alpha=None, epochs=2, bs=2, ga=4,
          modules='all', rslora=False, tag=''):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    alpha = alpha if alpha is not None else 2*r
    TM = {'all'      : ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
          'attn_only': ['q_proj','v_proj'],
          'attn_full': ['q_proj','k_proj','v_proj','o_proj']}[modules]
    m, t = FastLanguageModel.from_pretrained(
        'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
    m = FastLanguageModel.get_peft_model(m, r=r, lora_alpha=alpha, lora_dropout=0,
            target_modules=TM, use_gradient_checkpointing='unsloth',
            random_state=3407, use_rslora=rslora)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    a = SFTConfig(output_dir=f'hpo/{tag}', per_device_train_batch_size=bs,
        gradient_accumulation_steps=ga, num_train_epochs=epochs, learning_rate=lr,
        warmup_steps=5, logging_steps=10, eval_strategy='epoch',
        optim='adamw_8bit', lr_scheduler_type='linear', seed=3407,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        max_seq_length=1024, dataset_text_field='text', report_to='none')
    tr = SFTTrainer(model=m, tokenizer=t, train_dataset=DS['train'],
                    eval_dataset=DS['test'], args=a)
    t0 = time.time(); st = tr.train(); ev = tr.evaluate()
    res = {'tag':tag,'lr':lr,'r':r,'alpha':alpha,'epochs':epochs,
           'eff_batch':bs*ga,'modules':modules,'rslora':rslora,
           'trainable':trainable,'train_loss':st.metrics['train_loss'],
           'eval_loss':ev['eval_loss'],'minutes':round((time.time()-t0)/60,2),
           'peak_gb':round(torch.cuda.max_memory_allocated()/1024**3,2)}
    del m, tr; gc.collect(); torch.cuda.empty_cache()
    print(res); return res
```

### Step 2

PRIORITY 1 — LEARNING RATE. This is by far the most sensitive hyperparameter; tune it first and hold everything else fixed.

```python
results = []
for lr in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]:
    results.append(trial(lr=lr, tag=f'lr{lr}'))
lr_df = pd.DataFrame(results)
print(lr_df[['lr','train_loss','eval_loss','minutes']].to_string(index=False))
BEST_LR = lr_df.loc[lr_df['eval_loss'].idxmin(),'lr']
print('Best learning rate:', BEST_LR)
```

### Step 3

PRIORITY 2 — RANK. Rank sets adapter capacity. Watch for the point where extra rank stops buying validation loss — the Databricks study found r=8 to r=16 already saturating.

```python
rank_res = [trial(lr=BEST_LR, r=r, tag=f'r{r}') for r in [4, 8, 16, 32, 64]]
rank_df = pd.DataFrame(rank_res)
print(rank_df[['r','alpha','trainable','eval_loss','peak_gb','minutes']].to_string(index=False))
BEST_R = rank_df.loc[rank_df['eval_loss'].idxmin(),'r']
print('Best rank:', BEST_R)
```

### Step 4

PRIORITY 3 — ALPHA at the chosen rank. alpha/r is the update strength; a common default is alpha = 2r, but verify it on your own data.

```python
alpha_res = [trial(lr=BEST_LR, r=int(BEST_R), alpha=a, tag=f'a{a}')
             for a in [int(BEST_R), int(2*BEST_R), int(4*BEST_R)]]
alpha_df = pd.DataFrame(alpha_res)
alpha_df['scaling'] = alpha_df['alpha']/alpha_df['r']
print(alpha_df[['alpha','scaling','eval_loss']].to_string(index=False))
BEST_ALPHA = alpha_df.loc[alpha_df['eval_loss'].idxmin(),'alpha']
```

### Step 5

PRIORITY 4 — TARGET MODULES. This is the finding that most often decides output quality: attention-only underfits, all-linear wins.

```python
mod_res = [trial(lr=BEST_LR, r=int(BEST_R), alpha=int(BEST_ALPHA), modules=m, tag=m)
           for m in ['attn_only','attn_full','all']]
mod_df = pd.DataFrame(mod_res)
print(mod_df[['modules','trainable','eval_loss','minutes']].to_string(index=False))
BEST_MOD = mod_df.loc[mod_df['eval_loss'].idxmin(),'modules']
print('Best target modules:', BEST_MOD)
```

### Step 6

PRIORITY 5 — EPOCHS. More epochs on a small dataset buys overfitting, not quality. Find the point where validation loss stops improving.

```python
ep_res = [trial(lr=BEST_LR, r=int(BEST_R), alpha=int(BEST_ALPHA),
                modules=BEST_MOD, epochs=e, tag=f'e{e}') for e in [1, 2, 3, 5]]
ep_df = pd.DataFrame(ep_res)
print(ep_df[['epochs','train_loss','eval_loss','minutes']].to_string(index=False))
BEST_EP = ep_df.loc[ep_df['eval_loss'].idxmin(),'epochs']
```

### Step 7

PRIORITY 6 — EFFECTIVE BATCH SIZE via gradient accumulation. Accumulation buys a large effective batch on small VRAM, trading wall-clock time for stability.

```python
bs_res = [trial(lr=BEST_LR, r=int(BEST_R), alpha=int(BEST_ALPHA), modules=BEST_MOD,
                epochs=int(BEST_EP), bs=b, ga=g, tag=f'b{b}x{g}')
          for b, g in [(1,4),(2,4),(2,8),(4,4)]]
bs_df = pd.DataFrame(bs_res)
print(bs_df[['eff_batch','eval_loss','peak_gb','minutes']].to_string(index=False))
```

### Step 8

Plot the sensitivity of each hyperparameter. The steepest curve is the parameter that most deserves your remaining search budget.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2, figsize=(13,7))
for a, d, x, t in [(ax[0,0], lr_df,   'lr',       'Learning rate'),
                   (ax[0,1], rank_df, 'r',        'Rank r'),
                   (ax[1,0], ep_df,   'epochs',   'Epochs'),
                   (ax[1,1], bs_df,   'eff_batch','Effective batch')]:
    a.plot(d[x], d['eval_loss'], 'o-', color='#1F6FEB')
    a.set_xlabel(t); a.set_ylabel('validation loss'); a.grid(alpha=.3)
    if x == 'lr': a.set_xscale('log')
plt.suptitle('Hyperparameter sensitivity (lower is better)')
plt.tight_layout(); plt.savefig('hpo_sensitivity.png', dpi=120)
```

### Step 9

Save the winning configuration and the full sweep. This file is loaded by the deployment lab, so the production model uses the configuration you actually justified.

```python
best = {'learning_rate': float(BEST_LR), 'r': int(BEST_R),
        'lora_alpha': int(BEST_ALPHA), 'target_modules': BEST_MOD,
        'num_train_epochs': int(BEST_EP),
        'effective_batch_size': int(bs_df.loc[bs_df['eval_loss'].idxmin(),'eff_batch']),
        'selection_metric': 'validation loss',
        'best_eval_loss': float(min(r['eval_loss'] for r in
            results + rank_res + alpha_res + mod_res + ep_res + bs_res))}
json.dump(best, open('best_hyperparameters.json','w'), indent=2)
all_df = pd.DataFrame(results + rank_res + alpha_res + mod_res + ep_res + bs_res)
all_df.to_csv('hpo_sweep.csv', index=False)
print(json.dumps(best, indent=2))
```

## Test it

hpo_sweep.csv records every trial with its validation loss; hpo_sensitivity.png shows all four sensitivity curves; the target-module comparison shows 'all' beating 'attn_only'; and best_hyperparameters.json holds the selected configuration.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0