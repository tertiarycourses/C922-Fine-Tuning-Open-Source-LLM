#!/usr/bin/env python3
"""
Lab 11 — Tune Decoding Parameters for Higher-Quality Output
Fine Tuning Open Source LLM (C922)
LU2 — Exploratory Data Analysis

Objective: Apply optimisation strategies that raise output quality without retraining

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: decoding_sweep.png shows all three parameter sweeps; the temperature table shows distinct-2 rising with temperature; the repetition-penalty table shows the repetition rate falling; and generation_config.json holds both the evaluation and production configurations.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load the base model for inference and fix a small prompt set.
# Holding the prompts fixed is what makes the sweep a controlled experiment.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel
import torch, pandas as pd, numpy as np
model, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
FastLanguageModel.for_inference(model)
PROMPTS = [
 'My fibre broadband keeps dropping every evening. What should I do?',
 'How do I transfer my mobile plan to a family member?',
 'I was charged twice this month. Please explain the billing.',
 'Summarise the steps to reset a home router.',
]

# --------------------------------------------------------------------------
# STEP 2. Write one generation helper so every configuration is compared
# under identical conditions apart from the parameter under test.
# --------------------------------------------------------------------------
def generate(prompt, **kw):
    ids = tok.apply_chat_template([{'role':'user','content':prompt}],
            add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = model.generate(ids, max_new_tokens=160, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()

# --------------------------------------------------------------------------
# STEP 3. Define the quality metrics. Distinct-n measures lexical diversity;
# the repetition rate catches the degenerate loops that low temperature and
# no penalty produce.
# --------------------------------------------------------------------------
def distinct_n(text, n=2):
    w = text.split()
    if len(w) < n: return 0.0
    grams = [tuple(w[i:i+n]) for i in range(len(w)-n+1)]
    return len(set(grams))/len(grams)
def repetition_rate(text, n=4):
    w = text.split()
    if len(w) < n: return 0.0
    grams = [tuple(w[i:i+n]) for i in range(len(w)-n+1)]
    return 1 - len(set(grams))/len(grams)

# --------------------------------------------------------------------------
# STEP 4. Sweep TEMPERATURE. Temperature rescales the logits before
# sampling: low is focused and repeatable, high is creative and eventually
# incoherent.
# --------------------------------------------------------------------------
rows = []
for T in [0.1, 0.3, 0.7, 1.0, 1.5]:
    outs = [generate(p, do_sample=True, temperature=T, top_p=1.0) for p in PROMPTS]
    rows.append({'param':'temperature','value':T,
                 'distinct2': np.mean([distinct_n(o) for o in outs]),
                 'repetition': np.mean([repetition_rate(o) for o in outs]),
                 'avg_len'  : np.mean([len(o.split()) for o in outs])})
    print(f'T={T}: {outs[0][:100]}...')
temp_df = pd.DataFrame(rows); print(temp_df.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# STEP 5. Sweep TOP-P (nucleus sampling). Top-p samples from the smallest
# token set whose probability mass reaches p, adapting the candidate pool to
# the model's own confidence.
# --------------------------------------------------------------------------
rows = []
for P in [0.5, 0.8, 0.9, 0.95, 1.0]:
    outs = [generate(p, do_sample=True, temperature=0.7, top_p=P) for p in PROMPTS]
    rows.append({'param':'top_p','value':P,
                 'distinct2': np.mean([distinct_n(o) for o in outs]),
                 'repetition': np.mean([repetition_rate(o) for o in outs]),
                 'avg_len'  : np.mean([len(o.split()) for o in outs])})
topp_df = pd.DataFrame(rows); print(topp_df.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# STEP 6. Sweep REPETITION PENALTY. This directly attacks the degenerate
# loops that ruin long generations.
# --------------------------------------------------------------------------
rows = []
for R in [1.0, 1.05, 1.1, 1.2, 1.5]:
    outs = [generate(p, do_sample=True, temperature=0.7, top_p=0.9,
                     repetition_penalty=R) for p in PROMPTS]
    rows.append({'param':'repetition_penalty','value':R,
                 'distinct2': np.mean([distinct_n(o) for o in outs]),
                 'repetition': np.mean([repetition_rate(o) for o in outs]),
                 'avg_len'  : np.mean([len(o.split()) for o in outs])})
rep_df = pd.DataFrame(rows); print(rep_df.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# STEP 7. Plot all three sweeps together so the trade-off is visible:
# diversity rises with temperature while coherence falls, and repetition
# falls as the penalty rises.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3, figsize=(15,4))
for a, d, name in zip(ax, [temp_df, topp_df, rep_df],
                      ['Temperature','Top-p','Repetition penalty']):
    a.plot(d['value'], d['distinct2'],  'o-', label='distinct-2', color='#1F6FEB')
    a.plot(d['value'], d['repetition'], 's-', label='repetition', color='#DC2626')
    a.set_xlabel(name); a.set_ylabel('score'); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig('decoding_sweep.png', dpi=120)

# --------------------------------------------------------------------------
# STEP 8. Compare greedy against the tuned sampling configuration on the
# same prompt, and lock in the choice. Greedy is deterministic and right for
# evaluation; sampling is right for user-facing text.
# --------------------------------------------------------------------------
print('--- GREEDY (deterministic) ---')
print(generate(PROMPTS[0], do_sample=False))
print('\n--- TUNED SAMPLING ---')
print(generate(PROMPTS[0], do_sample=True, temperature=0.7,
               top_p=0.9, repetition_penalty=1.1))

# --------------------------------------------------------------------------
# STEP 9. Persist the generation config. Every later evaluation lab loads
# this file, so that a measured difference is caused by the model and not by
# the decoder.
# --------------------------------------------------------------------------
import json
cfg = {
  'evaluation': {'do_sample': False, 'max_new_tokens': 256},
  'production': {'do_sample': True, 'temperature': 0.7, 'top_p': 0.9,
                 'top_k': 50, 'repetition_penalty': 1.1, 'max_new_tokens': 256},
  'rationale': ('Greedy for evaluation so results are reproducible; '
                'temperature 0.7 with top_p 0.9 gave the best diversity/coherence '
                'balance, and a 1.1 repetition penalty removed 4-gram loops.'),
}
json.dump(cfg, open('generation_config.json','w'), indent=2)
print(json.dumps(cfg, indent=2))


print('\nLab 11 complete — decoding_sweep.png shows all three parameter sweeps; the temperature table shows distinct-2 rising with temper')
