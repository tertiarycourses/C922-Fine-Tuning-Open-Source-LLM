#!/usr/bin/env python3
"""
Lab 21 — Benchmark the Fine-Tuned Model Against State of the Art
Fine Tuning Open Source LLM (C922)
LU4 — Machine Learning Implementation and Operations

Objective: Analyse a fine-tuned model against state-of-the-art benchmarks

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: benchmark_report.json compares base and fine-tuned on ROUGE, BLEU, BERTScore and perplexity under one fixed decoding config; the catastrophic-forgetting check reports a percentage change on general text; and benchmark_comparison.png is produced.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load the held-out test set — data the model has NEVER seen.
# Benchmarking on training data produces a number that is meaningless and
# flattering.
# --------------------------------------------------------------------------
import pandas as pd, json, torch, numpy as np
test = pd.read_json('data_repo/v2/processed/test.jsonl', lines=True)
print(f'Held-out test examples: {len(test)}')
gen_cfg = json.load(open('generation_config.json'))['evaluation']
print('Fixed decoding config for ALL systems:', gen_cfg)

# --------------------------------------------------------------------------
# STEP 2. Load the base model — the control. A fine-tune that does not beat
# its own base model is a regression, however good its absolute score looks.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel
base, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
FastLanguageModel.for_inference(base)
def answer(m, prompt):
    ids = tok.apply_chat_template([{'role':'user','content':prompt}],
            add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = m.generate(ids, pad_token_id=tok.eos_token_id, **gen_cfg)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
base_out = [answer(base, r['input']) for _, r in test.iterrows()]
print(base_out[0][:200])

# --------------------------------------------------------------------------
# STEP 3. Load the fine-tuned adapter and generate on exactly the same
# prompts with exactly the same decoding config.
# --------------------------------------------------------------------------
from peft import PeftModel
ft, tok2 = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
ft = PeftModel.from_pretrained(ft, 'unsloth_adapter')
FastLanguageModel.for_inference(ft)
ft_out = [answer(ft, r['input']) for _, r in test.iterrows()]
print(ft_out[0][:200])

# --------------------------------------------------------------------------
# STEP 4. Score both systems on reference-based metrics. Several metrics are
# required because each one is blind to a different failure.
# --------------------------------------------------------------------------
os.system('pip install -q evaluate rouge_score sacrebleu bert_score')
import evaluate
rouge, bleu, bs = evaluate.load('rouge'), evaluate.load('sacrebleu'), evaluate.load('bertscore')
refs = test['output'].tolist()
def score(preds, name):
    r = rouge.compute(predictions=preds, references=refs)
    b = bleu.compute(predictions=preds, references=[[x] for x in refs])
    e = bs.compute(predictions=preds, references=refs, lang='en')
    return {'system':name, 'rougeL':round(r['rougeL'],4),
            'rouge1':round(r['rouge1'],4), 'bleu':round(b['score'],2),
            'bertscore_f1':round(float(np.mean(e['f1'])),4),
            'avg_len':round(float(np.mean([len(p.split()) for p in preds])),1)}
rows = [score(base_out,'base'), score(ft_out,'fine-tuned')]
print(pd.DataFrame(rows).to_string(index=False))

# --------------------------------------------------------------------------
# STEP 5. Add PERPLEXITY on the held-out references — a direct measure of
# how well each model models the target distribution.
# --------------------------------------------------------------------------
def ppl(m, texts):
    ls = []
    for t in texts[:40]:
        ids = tok(t, return_tensors='pt', truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            ls.append(float(m(**ids, labels=ids['input_ids']).loss))
    return float(np.exp(np.mean(ls)))
print(f'Base       held-out PPL: {ppl(base, refs):8.2f}')
print(f'Fine-tuned held-out PPL: {ppl(ft,   refs):8.2f}')

# --------------------------------------------------------------------------
# STEP 6. Run an LLM-AS-A-JUDGE evaluation for the qualities no n-gram
# metric captures: correctness, helpfulness and format compliance.
# --------------------------------------------------------------------------
JUDGE = ('Rate the ASSISTANT reply from 1 to 5 on each of: correctness, helpfulness, '
         'and house-style format (professional, concise, no invented account facts). '
         'Answer with ONLY three integers separated by spaces.')
import re
def judge(question, reply):
    ids = tok.apply_chat_template(
        [{'role':'system','content':JUDGE},
         {'role':'user','content':f'QUESTION: {question}\n\nASSISTANT: {reply}'}],
        add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = base.generate(ids, max_new_tokens=12, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    nums = [int(x) for x in re.findall(r'[1-5]', txt)][:3]
    return nums + [0]*(3-len(nums))
for name, outs in [('base', base_out), ('fine-tuned', ft_out)]:
    sc = np.array([judge(q, a) for q, a in zip(test['input'][:20], outs[:20])])
    print(f'{name:<12} correctness={sc[:,0].mean():.2f}  '
          f'helpfulness={sc[:,1].mean():.2f}  format={sc[:,2].mean():.2f}')

# --------------------------------------------------------------------------
# STEP 7. Check for CATASTROPHIC FORGETTING on a GENERAL task the fine-tune
# never targeted. This is the check most teams skip, and it is where
# regressions hide.
# --------------------------------------------------------------------------
GENERAL = ['What is the capital of France?',
           'Write a Python function that reverses a string.',
           'Explain photosynthesis in two sentences.',
           'What is 17 multiplied by 24?']
for q in GENERAL:
    print(f'\nQ: {q}')
    print(f'  base : {answer(base, q)[:130]}')
    print(f'  tuned: {answer(ft,   q)[:130]}')

# --------------------------------------------------------------------------
# STEP 8. Quantify the forgetting rather than eyeballing it: measure the
# perplexity shift on general text.
# --------------------------------------------------------------------------
GEN_TEXT = ['Paris is the capital and most populous city of France.',
            'Photosynthesis converts light energy into chemical energy in plants.',
            'A for loop iterates over a sequence of items in Python.']
pb, pf = ppl(base, GEN_TEXT), ppl(ft, GEN_TEXT)
print(f'General-text PPL  base={pb:.2f}  fine-tuned={pf:.2f}')
delta = 100*(pf-pb)/pb
print(f'Change: {delta:+.1f}%  -> '
      f"{'ACCEPTABLE' if delta < 15 else 'SIGNIFICANT FORGETTING - add replay data'}")

# --------------------------------------------------------------------------
# STEP 9. Plot the benchmark comparison and write the report with an
# explicit recommendation.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
df = pd.DataFrame(rows).set_index('system')
fig, ax = plt.subplots(1,3, figsize=(14,4))
for a, m in zip(ax, ['rougeL','bleu','bertscore_f1']):
    df[m].plot.bar(ax=a, color=['#5B6372','#1F6FEB'], rot=0)
    a.set_title(m); a.grid(alpha=.3, axis='y')
plt.suptitle('Fine-tuned vs base on the held-out set')
plt.tight_layout(); plt.savefig('benchmark_comparison.png', dpi=120)
gain = 100*(df.loc['fine-tuned','rougeL']-df.loc['base','rougeL'])/df.loc['base','rougeL']
report = {'metrics': rows, 'rougeL_gain_pct': round(float(gain),1),
          'general_ppl_change_pct': round(float(delta),1),
          'decoding': gen_cfg,
          'recommendation': ('Promote the fine-tune' if gain > 5 and delta < 15 else
                             'Do not promote - insufficient gain or significant forgetting')}
json.dump(report, open('benchmark_report.json','w'), indent=2)
print(json.dumps(report, indent=2))


print('\nLab 21 complete — benchmark_report.json compares base and fine-tuned on ROUGE, BLEU, BERTScore and perplexity under one fixed de')
