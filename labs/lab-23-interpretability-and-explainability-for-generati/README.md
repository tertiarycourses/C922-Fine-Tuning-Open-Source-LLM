# Lab 23 — Interpretability and Explainability for Generative Models

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU4 — Machine Learning Implementation and Operations  
**Delivery:** self-paced extension  
**Objective:** Apply interpretability and explainability methods to a generative model

## Goal

Open the model up with four complementary methods — attention visualisation, gradient-based token attribution, a layer probing classifier and behavioural ablation — and understand what each one can and cannot tell a regulator.

## What you will build

Four working explainability analyses with plots, plus a written note on the limits of each method — the kind of evidence a compliance officer actually asks for.

**Tools:** PyTorch, transformers, matplotlib, seaborn, scikit-learn

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-23-interpretability-and-explainability-for-generati/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

**Mock data provided:**

- `data/probe_sentiment.jsonl`

## Step-by-step

### Step 1

Load the model with attention outputs enabled. Attention is not returned by default because it costs memory — you must ask for it.

```python
import torch, numpy as np, matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL='unsloth/Llama-3.2-1B-Instruct'
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float32, device_map='auto',
    attn_implementation='eager', output_attentions=True)
model.eval()
print(f'Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}')
```

### Step 2

METHOD 1 — ATTENTION VISUALISATION. Run a forward pass and capture the attention tensors for every layer and head.

```python
TEXT = 'The customer reported that the fibre router has been offline since Monday evening'
ids  = tok(TEXT, return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model(**ids, output_attentions=True)
att  = out.attentions            # tuple: layers x (batch, heads, seq, seq)
toks = [tok.decode([i]) for i in ids['input_ids'][0]]
print(f'{len(att)} layers, shape per layer: {tuple(att[0].shape)}')
print('Tokens:', toks)
```

### Step 3

Plot the attention heatmap for an early, a middle and a late layer. Early layers attend locally; later layers attend to semantically relevant tokens far away.

```python
L = len(att)
fig, axes = plt.subplots(1, 3, figsize=(16,5))
for ax, layer in zip(axes, [0, L//2, L-1]):
    a = att[layer][0].mean(0).float().cpu().numpy()   # mean over heads
    im = ax.imshow(a, cmap='Blues', aspect='auto')
    ax.set_title(f'Layer {layer} (mean over heads)')
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=90, fontsize=7)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig('explain_attention.png', dpi=120)
```

### Step 4

Find the heads that specialise. Some heads track syntax, others track the previous token — naming them is what 'mechanistic interpretability' means in practice.

```python
last = att[-1][0].float().cpu().numpy()
for h in range(min(6, last.shape[0])):
    m = last[h]
    diag_prev = np.mean([m[i, i-1] for i in range(1, len(toks))])
    to_first  = m[1:,0].mean()
    print(f'Head {h}: previous-token={diag_prev:.3f}  attention-sink(BOS)={to_first:.3f}')
print('High previous-token = a positional head; high BOS = an attention sink.')
```

### Step 5

METHOD 2 — GRADIENT-BASED ATTRIBUTION (integrated gradients). Unlike attention this measures CAUSAL influence on the output, which is what a regulator actually asks about.

```python
emb_layer = model.get_input_embeddings()
inp_emb   = emb_layer(ids['input_ids']).detach()
baseline  = torch.zeros_like(inp_emb)
STEPS, total = 32, torch.zeros_like(inp_emb)
for a_ in np.linspace(0, 1, STEPS):
    x = (baseline + a_*(inp_emb - baseline)).requires_grad_(True)
    logit = model(inputs_embeds=x).logits[0,-1].max()
    grad, = torch.autograd.grad(logit, x)
    total += grad
ig = ((inp_emb - baseline) * total / STEPS).sum(-1)[0].abs().detach().cpu().numpy()
ig = ig / ig.sum()
for t, v in sorted(zip(toks, ig), key=lambda x:-x[1])[:8]:
    print(f'{v:7.2%}  {t!r}')
```

### Step 6

Plot attention against integrated gradients side by side. They frequently DISAGREE, and that disagreement is the single most important lesson about attention-as-explanation.

```python
attn_last = att[-1][0].mean(0)[-1].float().cpu().numpy()
attn_last = attn_last/attn_last.sum()
x = np.arange(len(toks)); w = 0.4
plt.figure(figsize=(12,4))
plt.bar(x-w/2, attn_last, w, label='Attention (last layer)', color='#1F6FEB')
plt.bar(x+w/2, ig,        w, label='Integrated gradients',   color='#F59E0B')
plt.xticks(x, toks, rotation=45, ha='right'); plt.legend(); plt.ylabel('normalised score')
plt.title('Attention shows where the model LOOKED; gradients show what MATTERED')
plt.tight_layout(); plt.savefig('explain_attention_vs_ig.png', dpi=120)
corr = np.corrcoef(attn_last, ig)[0,1]
print(f'Correlation between the two: {corr:.3f}')
```

### Step 7

METHOD 3 — PROBING CLASSIFIERS. Train a small classifier on each layer's hidden states to find WHERE in the network a property becomes linearly decodable.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
probe = pd.read_json('data/probe_sentiment.jsonl', lines=True)
def hidden(texts):
    reps = []
    for t in texts:
        i = tok(t, return_tensors='pt', truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            h = model(**i, output_hidden_states=True).hidden_states
        reps.append([layer[0].mean(0).float().cpu().numpy() for layer in h])
    return np.array(reps)       # (n, layers, dim)
H = hidden(probe['text'].tolist()); y = probe['label'].values
print('Hidden states:', H.shape)
```

### Step 8

Score the probe at every layer. The accuracy curve shows exactly where the model forms the concept — usually the middle layers.

```python
accs = []
for l in range(H.shape[1]):
    s = cross_val_score(LogisticRegression(max_iter=2000), H[:,l,:], y, cv=4).mean()
    accs.append(s)
    print(f'Layer {l:>2}: probe accuracy {s:.3f}')
plt.figure(figsize=(9,4))
plt.plot(accs, 'o-', color='#7C3AED')
plt.axhline(0.5, color='grey', ls='--', label='chance')
plt.xlabel('layer'); plt.ylabel('probe accuracy'); plt.legend(); plt.grid(alpha=.3)
plt.title('Where sentiment becomes linearly decodable')
plt.tight_layout(); plt.savefig('explain_probing.png', dpi=120)
print(f'Peak at layer {int(np.argmax(accs))} with accuracy {max(accs):.3f}')
```

### Step 9

METHOD 4 — BEHAVIOURAL ABLATION. Remove a token and measure how the output changes. This is the most convincing explanation for a non-technical stakeholder because it needs no internals.

```python
words = TEXT.split()
def logprob(text):
    i = tok(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        return -float(model(**i, labels=i['input_ids']).loss)
full = logprob(TEXT)
rows = []
for k in range(len(words)):
    ablated = ' '.join(words[:k] + words[k+1:])
    rows.append({'removed': words[k], 'delta': full - logprob(ablated)})
abl = pd.DataFrame(rows).sort_values('delta', ascending=False)
print(abl.head(8).to_string(index=False))
print('A large positive delta = removing that word hurt the model most.')
```

### Step 10

Write the explainability note, including the LIMITS of each method. Overstating what a method proves is itself a compliance failure.

```python
import json
note = {
 'attention_visualisation': {
   'shows':'Which tokens each head attended to, per layer.',
   'limit':'Attention is correlational, NOT causal - a high weight does not prove influence.'},
 'integrated_gradients': {
   'shows':'Causal attribution of the output to each input token.',
   'limit':'Depends on the baseline choice and is expensive over long sequences.'},
 'probing_classifier': {
   'shows':f'Sentiment becomes linearly decodable around layer {int(np.argmax(accs))}.',
   'limit':'Decodable does not mean the model USES it for its prediction.'},
 'behavioural_ablation': {
   'shows':'The measured output change when an input token is removed.',
   'limit':'Ignores interactions between tokens; combinatorial for multi-token effects.'},
 'attention_ig_correlation': round(float(corr),3),
 'guidance':'Use at least two methods; a claim supported by only attention is not defensible.',
}
json.dump(note, open('explainability_note.json','w'), indent=2)
print(json.dumps(note, indent=2))
```

## Test it

explain_attention.png, explain_attention_vs_ig.png and explain_probing.png are produced; the attention-vs-IG correlation is reported; the probing curve identifies a peak layer above chance; the ablation table ranks the tokens; and explainability_note.json records each method's limits.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0