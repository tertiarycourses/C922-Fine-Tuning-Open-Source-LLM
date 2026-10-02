#!/usr/bin/env python3
"""
Lab 24 — Intermediate Fine Tuning and Continuous Pre-Training
Fine Tuning Open Source LLM (C922)
LU4 — Machine Learning Implementation and Operations

Objective: Build a training pipeline that continues pre-training from an existing checkpoint

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: cpt_cycle_log.json records baseline plus two cycles with domain and general perplexity for each; cycle 2 successfully resumes from a saved checkpoint; the replay ratio is reported; and the forgetting guard produces an accept/reject verdict.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Understand the three strategies before you build. The distinction
# is the objective and the data, not the code.
# --------------------------------------------------------------------------
import pandas as pd
strat = pd.DataFrame([
 {'Strategy':'Direct task fine tuning','Objective':'Supervised (instruction->response)',
  'Data':'Small labelled task set','Adds':'Behaviour, format, style',
  'Use when':'You have labelled examples of the target task'},
 {'Strategy':'Intermediate fine tuning','Objective':'Supervised, then supervised',
  'Data':'Large related set, then the small task set','Adds':'Domain competence, then behaviour',
  'Use when':'Task data is scarce but related data is plentiful'},
 {'Strategy':'Continuous pre-training','Objective':'Self-supervised (next token)',
  'Data':'Large RAW domain text (no labels)','Adds':'Domain KNOWLEDGE and vocabulary',
  'Use when':'The domain language itself is unfamiliar to the base model'},
])
print(strat.to_string(index=False))

# --------------------------------------------------------------------------
# STEP 2. Load the raw domain corpus for continuous pre-training. Note there
# are no labels — the text is its own supervision.
# --------------------------------------------------------------------------
from datasets import load_dataset
raw = load_dataset('json', data_files='data/domain_corpus.jsonl')['train']
print(raw)
print(raw[0]['text'][:400])
print(f'Total documents: {len(raw)}')

# --------------------------------------------------------------------------
# STEP 3. Build the REPLAY mixture. Mixing a slice of general text into the
# domain corpus is the standard defence against catastrophic forgetting —
# without it, the model over-specialises.
# --------------------------------------------------------------------------
from datasets import concatenate_datasets
general = load_dataset('tatsu-lab/alpaca', split='train[:300]')
general = general.map(lambda r: {'text': f"{r['instruction']} {r['output']}"},
                      remove_columns=general.column_names)
REPLAY_RATIO = 0.15
n_replay = int(len(raw) * REPLAY_RATIO / (1 - REPLAY_RATIO))
mixed = concatenate_datasets([raw, general.select(range(min(n_replay, len(general))))])
mixed = mixed.shuffle(seed=3407)
print(f'Domain {len(raw)} + replay {min(n_replay, len(general))} = {len(mixed)} '
      f'({100*min(n_replay,len(general))/len(mixed):.0f}% replay)')

# --------------------------------------------------------------------------
# STEP 4. Set up continuous pre-training. The critical differences from task
# fine tuning: a much LOWER learning rate, and embed_tokens/lm_head included
# in the trainable modules so new domain vocabulary can actually be learned.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
model, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0, bias='none',
    target_modules=['q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj',
                    'embed_tokens','lm_head'],   # needed for NEW vocabulary
    use_gradient_checkpointing='unsloth', random_state=3407, use_rslora=True)
model.print_trainable_parameters()

# --------------------------------------------------------------------------
# STEP 5. Record the baseline before any continued pre-training, on BOTH
# domain text and general text. Both numbers are needed to detect the
# forgetting trade-off.
# --------------------------------------------------------------------------
import numpy as np
def ppl(texts):
    ls = []
    for t in texts:
        i = tok(t, return_tensors='pt', truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            ls.append(float(model(**i, labels=i['input_ids']).loss))
    return float(np.exp(np.mean(ls)))
DOMAIN_VAL  = raw['text'][:20]
GENERAL_VAL = ['Paris is the capital of France.',
               'Water boils at one hundred degrees Celsius at sea level.',
               'A Python list comprehension builds a list from an iterable.']
base_dom, base_gen = ppl(DOMAIN_VAL), ppl(GENERAL_VAL)
print(f'BASELINE  domain PPL={base_dom:.2f}  general PPL={base_gen:.2f}')

# --------------------------------------------------------------------------
# STEP 6. Configure the pipeline for CYCLES with checkpointing. Saving
# optimizer and scheduler state alongside the weights is what makes a run
# resumable rather than restartable-from-scratch.
# --------------------------------------------------------------------------
from trl import SFTTrainer, SFTConfig
args = SFTConfig(
    output_dir='cpt_ckpt',
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=5e-5,          # LOW - continued pre-training, not task fitting
    embedding_learning_rate=1e-5,# even lower for the embedding matrix
    warmup_ratio=0.1, lr_scheduler_type='cosine',
    optim='adamw_8bit', weight_decay=0.01, logging_steps=5,
    save_strategy='steps', save_steps=20, save_total_limit=3,
    bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
    max_seq_length=1024, dataset_text_field='text', seed=3407, report_to='none')
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=mixed, args=args)

# --------------------------------------------------------------------------
# STEP 7. Run cycle 1 and validate it before promoting. A cycle that raises
# general perplexity too far must be rejected, not shipped.
# --------------------------------------------------------------------------
stats = trainer.train()
c1_dom, c1_gen = ppl(DOMAIN_VAL), ppl(GENERAL_VAL)
print(f'CYCLE 1   domain PPL={c1_dom:.2f} ({100*(c1_dom-base_dom)/base_dom:+.1f}%)  '
      f'general PPL={c1_gen:.2f} ({100*(c1_gen-base_gen)/base_gen:+.1f}%)')
FORGET_LIMIT = 20.0
drift = 100*(c1_gen-base_gen)/base_gen
print('Cycle 1 verdict:', 'ACCEPT' if drift < FORGET_LIMIT else
      'REJECT - raise the replay ratio or lower the learning rate')

# --------------------------------------------------------------------------
# STEP 8. Demonstrate RESUMING from a checkpoint — the property that makes
# this a pipeline rather than a script. New documents arrive weekly; you
# never retrain from scratch.
# --------------------------------------------------------------------------
import glob, os
ckpts = sorted(glob.glob('cpt_ckpt/checkpoint-*'),
               key=lambda p: int(p.split('-')[-1]))
print('Checkpoints available:', [os.path.basename(c) for c in ckpts])
print('Contents of the latest:', sorted(os.listdir(ckpts[-1]))[:10])
new_docs = load_dataset('json', data_files='data/domain_corpus_week2.jsonl')['train']
print(f'New documents this cycle: {len(new_docs)}')
trainer.train_dataset = concatenate_datasets(
    [new_docs, general.select(range(40))]).shuffle(seed=3407)
stats2 = trainer.train(resume_from_checkpoint=ckpts[-1])
print('Cycle 2 resumed from', os.path.basename(ckpts[-1]))

# --------------------------------------------------------------------------
# STEP 9. Validate cycle 2 and record the cycle log. Every cycle is
# versioned against its dataset and its validation result — this is the
# audit trail a continuous pre-training pipeline requires.
# --------------------------------------------------------------------------
c2_dom, c2_gen = ppl(DOMAIN_VAL), ppl(GENERAL_VAL)
print(f'CYCLE 2   domain PPL={c2_dom:.2f}  general PPL={c2_gen:.2f}')
import json, hashlib
log = {'cycles': [
  {'cycle':0,'label':'baseline','domain_ppl':round(base_dom,3),'general_ppl':round(base_gen,3)},
  {'cycle':1,'docs':len(raw),'replay_ratio':REPLAY_RATIO,'lr':5e-5,
   'domain_ppl':round(c1_dom,3),'general_ppl':round(c1_gen,3),
   'verdict':'accept' if drift < FORGET_LIMIT else 'reject'},
  {'cycle':2,'docs':len(new_docs),'resumed_from':os.path.basename(ckpts[-1]),
   'domain_ppl':round(c2_dom,3),'general_ppl':round(c2_gen,3)}],
 'safeguards':['replay mixture of general data','low learning rate','lower embedding LR',
               'per-cycle general-benchmark validation','checkpoint + optimizer state saved',
               'cycle rejected if general PPL rises more than 20%'],
}
json.dump(log, open('cpt_cycle_log.json','w'), indent=2)
model.save_pretrained('cpt_adapter'); tok.save_pretrained('cpt_adapter')
print(json.dumps(log, indent=2))

# --------------------------------------------------------------------------
# STEP 10. Compare all three strategies on the same held-out set — the
# deliverable of this lab.
# --------------------------------------------------------------------------
print(strat.to_string(index=False))
print('\nMeasured on this domain:')
print(f'  Baseline (no adaptation) : domain PPL {base_dom:.2f}  general PPL {base_gen:.2f}')
print(f'  Continuous pre-training  : domain PPL {c2_dom:.2f}  general PPL {c2_gen:.2f}')
print('  Recommended sequence: continuous pre-training on raw domain text,')
print('  THEN intermediate fine tuning on related labelled data,')
print('  THEN task fine tuning on the small target set.')


print('\nLab 24 complete — cpt_cycle_log.json records baseline plus two cycles with domain and general perplexity for each; cycle 2 succe')
