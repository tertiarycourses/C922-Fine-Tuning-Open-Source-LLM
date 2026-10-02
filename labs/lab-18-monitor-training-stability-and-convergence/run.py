#!/usr/bin/env python3
"""
Lab 18 — Monitor Training Stability and Convergence
Fine Tuning Open Source LLM (C922)
LU3 — Modelling

Objective: Monitor a generative training run for stability and convergence, and correct failures

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: monitor_healthy.png, monitor_diverged.png, monitor_overfit.png and monitor_stalled.png are all produced; diagnose() correctly labels the healthy run HEALTHY and each broken run with its specific failure; and training_troubleshooting.csv maps every symptom to a fix.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Build a monitoring callback that records everything a diagnosis
# needs. What you do not log, you cannot debug after the run.
# --------------------------------------------------------------------------
from transformers import TrainerCallback
import torch, numpy as np
class Monitor(TrainerCallback):
    def __init__(self): self.log = []
    def on_log(self, args, state, control, logs=None, **kw):
        if not logs: return
        self.log.append({
            'step' : state.global_step,
            'loss' : logs.get('loss'),
            'eval_loss': logs.get('eval_loss'),
            'lr'   : logs.get('learning_rate'),
            'gnorm': logs.get('grad_norm'),
            'vram' : torch.cuda.max_memory_allocated()/1024**3,
        })
mon = Monitor()

# --------------------------------------------------------------------------
# STEP 2. Run a HEALTHY baseline training so you know what 'good' looks like
# before you try to recognise 'bad'.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
def build(lr=2e-4, epochs=2, r=16):
    m, t = FastLanguageModel.from_pretrained(
        'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
    m = FastLanguageModel.get_peft_model(m, r=r, lora_alpha=2*r, lora_dropout=0,
            target_modules=['q_proj','k_proj','v_proj','o_proj',
                            'gate_proj','up_proj','down_proj'],
            use_gradient_checkpointing='unsloth', random_state=3407)
    d = load_dataset('json', data_files='formatted_train.jsonl')['train']
    d = d.train_test_split(test_size=0.15, seed=3407)
    a = SFTConfig(output_dir='mon_out', per_device_train_batch_size=2,
        gradient_accumulation_steps=4, num_train_epochs=epochs, learning_rate=lr,
        warmup_steps=5, logging_steps=2, eval_strategy='steps', eval_steps=10,
        optim='adamw_8bit', lr_scheduler_type='linear', seed=3407,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        max_seq_length=1024, dataset_text_field='text', report_to='none')
    return SFTTrainer(model=m, tokenizer=t, train_dataset=d['train'],
                      eval_dataset=d['test'], args=a)
healthy = build(); healthy.add_callback(mon); healthy.train()

# --------------------------------------------------------------------------
# STEP 3. Plot the four-panel dashboard. A healthy run shows loss falling
# smoothly, validation tracking training, a bounded gradient norm and the
# learning rate following its schedule.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt, pandas as pd
def dashboard(log, title, fname):
    d = pd.DataFrame(log)
    fig, ax = plt.subplots(2,2, figsize=(13,7)); fig.suptitle(title)
    t = d.dropna(subset=['loss'])
    ax[0,0].plot(t['step'], t['loss'], color='#1F6FEB', label='train')
    e = d.dropna(subset=['eval_loss'])
    if len(e): ax[0,0].plot(e['step'], e['eval_loss'], 'o-', color='#F59E0B', label='val')
    ax[0,0].set_title('Loss'); ax[0,0].legend(); ax[0,0].grid(alpha=.3)
    g = d.dropna(subset=['gnorm'])
    ax[0,1].plot(g['step'], g['gnorm'], color='#DC2626')
    ax[0,1].set_title('Gradient norm'); ax[0,1].grid(alpha=.3)
    l = d.dropna(subset=['lr'])
    ax[1,0].plot(l['step'], l['lr'], color='#10B981')
    ax[1,0].set_title('Learning rate'); ax[1,0].grid(alpha=.3)
    ax[1,1].plot(d['step'], d['vram'], color='#7C3AED')
    ax[1,1].set_title('Peak VRAM (GB)'); ax[1,1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(fname, dpi=120)
    return d
healthy_df = dashboard(mon.log, 'Healthy run', 'monitor_healthy.png')
print(healthy_df.dropna(subset=['loss'])[['step','loss','gnorm']].tail().to_string(index=False))

# --------------------------------------------------------------------------
# STEP 4. Define the automated convergence checks. These are the rules that
# turn a chart into a decision.
# --------------------------------------------------------------------------
def diagnose(d):
    out = []
    tr = d.dropna(subset=['loss'])
    ev = d.dropna(subset=['eval_loss'])
    if tr['loss'].isna().any() or np.isinf(tr['loss']).any():
        out.append('DIVERGED: NaN/Inf loss -> lower the learning rate, use bf16, clip gradients')
    if len(tr) > 4 and tr['loss'].iloc[-1] > tr['loss'].iloc[0]:
        out.append('DIVERGED: loss ended above where it started -> lower the learning rate')
    if len(tr) > 4 and abs(tr['loss'].iloc[-1] - tr['loss'].iloc[0]) < 0.02:
        out.append('STALLED: loss essentially flat -> raise the LR, check the adapter is attached')
    if len(ev) > 2 and ev['eval_loss'].iloc[-1] > ev['eval_loss'].min() * 1.05:
        out.append('OVERFITTING: validation loss rising -> fewer epochs, more data, add dropout')
    g = d.dropna(subset=['gnorm'])
    if len(g) and g['gnorm'].max() > 10:
        out.append(f"UNSTABLE: gradient norm peaked at {g['gnorm'].max():.1f} -> clip to 1.0")
    return out or ['HEALTHY: loss decreasing, validation tracking, gradients bounded']
for line in diagnose(healthy_df): print(' ', line)

# --------------------------------------------------------------------------
# STEP 5. FAILURE 1 — DIVERGENCE. Set the learning rate 50x too high and
# watch the loss explode. The signature is a loss that spikes and may become
# NaN, with a gradient norm that blows up.
# --------------------------------------------------------------------------
mon_div = Monitor()
div = build(lr=1e-2); div.add_callback(mon_div)
try:
    div.train()
except Exception as e:
    print('Training crashed:', type(e).__name__, e)
div_df = dashboard(mon_div.log, 'Diverged run (lr=1e-2)', 'monitor_diverged.png')
for line in diagnose(div_df): print(' ', line)

# --------------------------------------------------------------------------
# STEP 6. Apply the fix for divergence — lower the learning rate and clip
# the gradient norm — and confirm the loss recovers.
# --------------------------------------------------------------------------
mon_fix = Monitor()
fix = build(lr=2e-4); fix.args.max_grad_norm = 1.0
fix.add_callback(mon_fix); fix.train()
fix_df = pd.DataFrame(mon_fix.log)
print('After the fix:')
for line in diagnose(fix_df): print(' ', line)

# --------------------------------------------------------------------------
# STEP 7. FAILURE 2 — OVERFITTING. Train far too many epochs on a small set.
# The signature is training loss still falling while validation loss turns
# upward.
# --------------------------------------------------------------------------
mon_of = Monitor()
of = build(epochs=12); of.add_callback(mon_of); of.train()
of_df = dashboard(mon_of.log, 'Overfitted run (12 epochs)', 'monitor_overfit.png')
ev = of_df.dropna(subset=['eval_loss'])
if len(ev):
    best = ev.loc[ev['eval_loss'].idxmin()]
    print(f"Best validation loss {best['eval_loss']:.4f} at step {int(best['step'])}; "
          f"final {ev['eval_loss'].iloc[-1]:.4f}")
for line in diagnose(of_df): print(' ', line)

# --------------------------------------------------------------------------
# STEP 8. Fix overfitting with early stopping on the validation metric —
# stop at the best checkpoint rather than the last one.
# --------------------------------------------------------------------------
from transformers import EarlyStoppingCallback
mon_es = Monitor()
es = build(epochs=12)
es.args.load_best_model_at_end = True
es.args.metric_for_best_model  = 'eval_loss'
es.args.greater_is_better      = False
es.args.save_strategy          = 'steps'
es.args.save_steps             = 10
es.add_callback(mon_es)
es.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
es.train()
print('Early stopping restored the best checkpoint instead of the last.')

# --------------------------------------------------------------------------
# STEP 9. FAILURE 3 — A STALLED RUN, and the troubleshooting table that maps
# each signature to its fix. This table is the deliverable of this lab.
# --------------------------------------------------------------------------
mon_st = Monitor()
st = build(lr=1e-7); st.add_callback(mon_st); st.train()
st_df = dashboard(mon_st.log, 'Stalled run (lr=1e-7)', 'monitor_stalled.png')
for line in diagnose(st_df): print(' ', line)
guide = pd.DataFrame([
 {'Symptom':'Loss -> NaN/Inf','Cause':'LR too high or fp16 overflow','Fix':'Lower LR 10x, use bf16, clip grad to 1.0'},
 {'Symptom':'Loss spikes then recovers','Cause':'Bad batch or LR slightly high','Fix':'Clip gradients, add warmup steps'},
 {'Symptom':'Loss flat','Cause':'LR too low / adapter not attached / labels masked','Fix':'Raise LR, check print_trainable_parameters'},
 {'Symptom':'Val loss rising, train falling','Cause':'Overfitting','Fix':'Fewer epochs, early stopping, more data, dropout'},
 {'Symptom':'Gradient norm > 10','Cause':'Unstable updates','Fix':'max_grad_norm=1.0, lower LR'},
 {'Symptom':'CUDA OOM','Cause':'Batch or sequence too large','Fix':'Lower batch, raise grad accumulation, gradient checkpointing'},
])
print(guide.to_string(index=False)); guide.to_csv('training_troubleshooting.csv', index=False)


print('\nLab 18 complete — monitor_healthy.png, monitor_diverged.png, monitor_overfit.png and monitor_stalled.png are all produced; diagn')
