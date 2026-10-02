# Lab 14 — Build LoRA from First Principles

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU3 — Modelling  
**Delivery:** self-paced extension  
**Objective:** Implement the LoRA low-rank decomposition and verify its parameter and memory savings

## Goal

Implement LoRA yourself in about thirty lines of PyTorch — the low-rank update h = Wx + (alpha/r)BAx — then prove the three properties that make it work: zero-initialised B means an identity start, the parameter count collapses with r, and the merged weight is mathematically identical to the adapted forward pass.

## What you will build

A working LoRALinear layer, a parameter-count table across ranks, and a numerical proof that merging BA into W reproduces the adapted output exactly.

**Tools:** PyTorch, numpy, matplotlib

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-14-build-lora-from-first-principles/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

> **Input:** this lab has no mock dataset of its own — it consumes the output of an earlier lab in the same Learning Unit. Run the labs in order.

## Step-by-step

### Step 1

Start from a plain linear layer — the thing LoRA wraps. Every attention and MLP projection in a transformer is one of these.

```python
import torch, torch.nn as nn, numpy as np
torch.manual_seed(3407)
d_in, d_out = 512, 512
base = nn.Linear(d_in, d_out, bias=False)
print(f'Base layer parameters: {base.weight.numel():,}')
```

### Step 2

Implement LoRA. A is d_in x r, B is r x d_out, and only these two are trained — W stays frozen. The scaling factor is alpha/r.

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False              # W is FROZEN
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r                 # the alpha/r factor
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))   # A ~ Kaiming
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.T @ self.B.T * self.scaling
lora = LoRALinear(base, r=8, alpha=16)
print(lora)
```

### Step 3

Property 1 — IDENTITY AT INITIALISATION. Because B starts at zeros, BA = 0, so the adapted model begins exactly equal to the base model. Training is a strictly additive perturbation.

```python
x = torch.randn(4, d_in)
with torch.no_grad():
    diff = (lora(x) - base(x)).abs().max()
print(f'Max |LoRA(x) - base(x)| at init = {diff:.10f}')
assert diff < 1e-6, 'B must be zero-initialised'
print('Confirmed: LoRA starts as an exact no-op.')
```

### Step 4

Property 2 — PARAMETER COLLAPSE. The trainable count is r(d_in + d_out) instead of d_in x d_out, which is what makes fine tuning cheap.

```python
import pandas as pd
rows = []
for r in [1,2,4,8,16,32,64,128]:
    lp = r * (d_in + d_out)
    rows.append({'rank r': r, 'LoRA params': lp,
                 'Full params': d_in*d_out,
                 '% of full': round(100*lp/(d_in*d_out), 2)})
tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))
```

### Step 5

Plot the collapse so the trade-off between rank and cost is visible: capacity rises linearly with r while full fine tuning stays flat and enormous.

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(tbl['rank r'], tbl['LoRA params'], 'o-', color='#1F6FEB', label='LoRA')
plt.axhline(d_in*d_out, color='#DC2626', ls='--', label='Full fine tune')
plt.xscale('log', base=2); plt.yscale('log')
plt.xlabel('rank r'); plt.ylabel('trainable parameters'); plt.legend(); plt.grid(alpha=.3)
plt.title('LoRA parameter count vs rank')
plt.savefig('lora_param_scaling.png', dpi=120, bbox_inches='tight')
```

### Step 6

Train the adapter briefly on a synthetic target so B becomes non-zero and the adapter actually changes the output.

```python
target = nn.Linear(d_in, d_out, bias=False)      # the function we want to reach
opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-3)
for step in range(300):
    xb = torch.randn(32, d_in)
    loss = nn.functional.mse_loss(lora(xb), target(xb))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0: print(f'step {step:>3}  loss {loss.item():.4f}')
print(f'B is now non-zero: {lora.B.abs().max().item():.4f}')
```

### Step 7

Property 3 — MERGING IS EXACT. Fold the update into W: W' = W + (alpha/r)BA. The merged layer must reproduce the adapted output bit for bit, which is why merged LoRA adds ZERO inference latency.

```python
with torch.no_grad():
    delta = (lora.B @ lora.A) * lora.scaling      # (d_out, d_in)
    merged = nn.Linear(d_in, d_out, bias=False)
    merged.weight.copy_(base.weight + delta)
    xt = torch.randn(16, d_in)
    err = (lora(xt) - merged(xt)).abs().max()
print(f'Max |adapted - merged| = {err:.10f}')
assert err < 1e-4
print('Confirmed: merging is mathematically exact -> zero added latency.')
```

### Step 8

Show the effect of ALPHA. alpha/r scales the update, so raising alpha at a fixed rank strengthens the adaptation without adding a single parameter.

```python
for alpha in [4, 8, 16, 32, 64]:
    s = alpha / 8
    with torch.no_grad():
        out = base(xt) + (xt @ lora.A.T @ lora.B.T) * s
    shift = (out - base(xt)).abs().mean().item()
    print(f'alpha={alpha:>3} (scaling={s:>4.1f}): mean shift from base = {shift:.4f}')
```

### Step 9

Compare standard scaling against rank-stabilised LoRA. rsLoRA divides by sqrt(r), which keeps the update magnitude stable as the rank rises instead of shrinking it.

```python
rows = []
for r in [8,16,32,64,128]:
    rows.append({'r': r, 'alpha/r (standard)': round(16/r, 4),
                 'alpha/sqrt(r) (rsLoRA)': round(16/np.sqrt(r), 4)})
print(pd.DataFrame(rows).to_string(index=False))
print('Standard scaling shrinks the update as r grows; rsLoRA keeps it steady.')
```

## Test it

The identity-at-init assertion passes (difference < 1e-6), the parameter table shows LoRA at well under 5% of full parameters for r<=16, the merge assertion passes (error < 1e-4), and lora_param_scaling.png is produced.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0