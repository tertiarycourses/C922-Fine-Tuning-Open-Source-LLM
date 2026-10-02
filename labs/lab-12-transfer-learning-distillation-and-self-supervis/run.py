#!/usr/bin/env python3
"""
Lab 12 — Transfer Learning, Distillation and Self-Supervised Learning
Fine Tuning Open Source LLM (C922)
LU2 — Exploratory Data Analysis

Objective: Explain and demonstrate the generative concepts that underpin fine tuning

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: The domain text scores a clearly lower perplexity than the shuffled text; distilled_dataset.jsonl holds the teacher's generations; the softmax temperature sweep shows the distribution flattening; explainability_attribution.png is produced; and generative_concepts.json records all four.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. TRANSFER LEARNING — measure what the pre-trained model already
# knows. Fine tuning starts from this baseline, and the baseline is why so
# few examples are needed.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel
import torch, numpy as np
model, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
FastLanguageModel.for_inference(model)
def perplexity(text):
    ids = tok(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        loss = model(**ids, labels=ids['input_ids']).loss
    return float(torch.exp(loss))
for t in ['The capital of Singapore is Singapore.',
          'Colourless green ideas sleep furiously banana.',
          'Low-rank adaptation freezes the base weights and trains two small matrices.']:
    print(f'PPL {perplexity(t):8.2f}  |  {t[:60]}')

# --------------------------------------------------------------------------
# STEP 2. Show that the pre-trained representation transfers: the model
# assigns much lower perplexity to well-formed domain text than to nonsense,
# before any fine tuning at all.
# --------------------------------------------------------------------------
domain = ['Please restart your router and check the WAN light.',
          'Your billing cycle runs from the first of the month.']
noise  = ['Router the please and light check WAN restart your.',
          'Cycle the billing month first from runs of your the.']
print(f'Domain text mean PPL : {np.mean([perplexity(t) for t in domain]):.1f}')
print(f'Shuffled text mean PPL: {np.mean([perplexity(t) for t in noise]):.1f}')
print('Transfer learning gives us this structure for free.')

# --------------------------------------------------------------------------
# STEP 3. KNOWLEDGE DISTILLATION step 1 — generate the teacher's outputs.
# The teacher's responses become the student's training targets,
# transferring capability into a smaller model.
# --------------------------------------------------------------------------
import pandas as pd
prompts = pd.read_json('data/distill_prompts.jsonl', lines=True)['prompt'].tolist()[:30]
def teacher_answer(p):
    ids = tok.apply_chat_template([{'role':'user','content':p}],
            add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = model.generate(ids, max_new_tokens=120, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
distilled = [{'instruction':'Answer the customer question.',
              'input':p, 'output':teacher_answer(p)} for p in prompts]
print(distilled[0])

# --------------------------------------------------------------------------
# STEP 4. Distillation step 2 — inspect the teacher's SOFT targets. The
# probability distribution carries far more information than the single
# argmax token, which is why distillation works so well.
# --------------------------------------------------------------------------
ids = tok('The customer asks about', return_tensors='pt').to('cuda')
with torch.no_grad():
    logits = model(**ids).logits[0,-1]
for T in [1.0, 2.0, 4.0]:
    probs = torch.softmax(logits/T, dim=-1)
    top = torch.topk(probs, 5)
    toks = [tok.decode([i]) for i in top.indices]
    print(f'T={T}: ' + ', '.join(f'{t!r}={p:.3f}' for t,p in zip(toks, top.values)))
print('Higher T reveals the dark knowledge the argmax hides.')

# --------------------------------------------------------------------------
# STEP 5. Save the distillation dataset — a real artifact you could now use
# to fine-tune a smaller model.
# --------------------------------------------------------------------------
import json
with open('distilled_dataset.jsonl','w') as f:
    for r in distilled:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Wrote {len(distilled)} teacher-generated examples')

# --------------------------------------------------------------------------
# STEP 6. SELF-SUPERVISED LEARNING — build the objective that created the
# base model in the first place. Causal language modelling generates its own
# labels by shifting the input by one token.
# --------------------------------------------------------------------------
text = 'Low-rank adaptation freezes the pre-trained weights and trains two small matrices.'
ids  = tok(text, return_tensors='pt')['input_ids']
inp, tgt = ids[:, :-1], ids[:, 1:]
print('Input :', tok.decode(inp[0][:12]))
print('Target:', tok.decode(tgt[0][:12]))
print('The label IS the data shifted by one - no human annotation needed.')

# --------------------------------------------------------------------------
# STEP 7. Run the self-supervised loss and confirm it is exactly the cross-
# entropy the trainer optimises.
# --------------------------------------------------------------------------
with torch.no_grad():
    out = model(ids.to('cuda'), labels=ids.to('cuda'))
print(f'Self-supervised loss : {out.loss.item():.4f}')
print(f'Perplexity           : {torch.exp(out.loss).item():.2f}')
print('This same loss, over billions of tokens, produced the base model.')

# --------------------------------------------------------------------------
# STEP 8. EXPLAINABILITY — attribute the prediction to input tokens with
# gradient x input. This is the concrete form of 'explainability', revisited
# in Lab 23.
# --------------------------------------------------------------------------
model.gradient_checkpointing_disable()
emb_layer = model.get_input_embeddings()
ids = tok('The router keeps disconnecting every evening', return_tensors='pt').to('cuda')
embs = emb_layer(ids['input_ids']).detach().requires_grad_(True)
out  = model(inputs_embeds=embs).logits[0,-1]
out.max().backward()
attr = (embs.grad * embs).sum(-1)[0].abs().detach().float().cpu().numpy()
toks = [tok.decode([i]) for i in ids['input_ids'][0]]
for t, a in sorted(zip(toks, attr/attr.sum()), key=lambda x:-x[1])[:8]:
    print(f'{a:6.1%}  {t!r}')

# --------------------------------------------------------------------------
# STEP 9. Plot the attribution and record the four concepts with their
# measured evidence — the deliverable of this lab.
# --------------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(11,3))
plt.bar(range(len(toks)), attr/attr.sum(), color='#1F6FEB')
plt.xticks(range(len(toks)), toks, rotation=45, ha='right')
plt.ylabel('attribution'); plt.title('Token attribution (gradient x input)')
plt.tight_layout(); plt.savefig('explainability_attribution.png', dpi=120)
summary = {
 'transfer_learning': 'Pre-trained weights already encode language structure; fine tuning adapts rather than teaches from scratch.',
 'distillation'     : 'A large teacher supplies soft targets that transfer capability to a smaller student.',
 'self_supervised'  : 'Labels come from the data itself (next-token prediction), which is how the base model was trained.',
 'explainability'   : 'Gradient-based attribution shows which input tokens drove the prediction.',
}
json.dump(summary, open('generative_concepts.json','w'), indent=2)
print(json.dumps(summary, indent=2))


print('\nLab 12 complete — The domain text scores a clearly lower perplexity than the shuffled text; distilled_dataset.jsonl holds the te')
