#!/usr/bin/env python3
"""
Lab 13 — Frame the Problem and Select the Adaptation Method
Fine Tuning Open Source LLM (C922)
LU3 — Modelling

Objective: Frame a business problem as a fine-tuning problem and justify the adaptation method

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: All three options produce output on the same ticket, the option comparison table is printed, CLIP scores the correct caption highest, and adaptation_decision.json records the chosen method with an explicit justification against the stated constraints.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. State the business problem precisely. A vague problem statement is
# the root cause of most wasted fine-tuning budget.
# --------------------------------------------------------------------------
PROBLEM = {
 'business_need': 'Draft first-response replies for telco support tickets in the house style.',
 'current_state': 'Agents write every reply by hand; average handling time 6 minutes.',
 'target'       : 'Draft that an agent accepts with minor edits in under 1 minute.',
 'constraint'   : 'Must run on a single 16GB GPU; no customer data leaves Singapore.',
 'success'      : 'Style match >= 80% on a held-out set; no invented account facts.',
}
import json; print(json.dumps(PROBLEM, indent=2))

# --------------------------------------------------------------------------
# STEP 2. Option A — prompt engineering. Always test this first: it costs
# nothing and often resolves instruction-following problems entirely.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel
import torch
model, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(model)
def ask(system, user, **kw):
    ids = tok.apply_chat_template(
        [{'role':'system','content':system},{'role':'user','content':user}],
        add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = model.generate(ids, max_new_tokens=180, do_sample=False,
                         pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
TICKET = 'My fibre has been down since yesterday evening and nobody has called me back.'
print(ask('You are a telco support agent. Be concise and empathetic.', TICKET))

# --------------------------------------------------------------------------
# STEP 3. Option B — retrieval augmented generation. RAG is the right answer
# when the gap is missing FACTS, because fine tuning teaches behaviour, not
# a knowledge base that changes weekly.
# --------------------------------------------------------------------------
from sentence_transformers import SentenceTransformer
import numpy as np, pandas as pd
kb = pd.read_json('data/kb_articles.jsonl', lines=True)
enc = SentenceTransformer('all-MiniLM-L6-v2')
kb_emb = enc.encode(kb['text'].tolist(), normalize_embeddings=True)
def retrieve(q, k=2):
    sims = enc.encode([q], normalize_embeddings=True) @ kb_emb.T
    return kb.iloc[sims[0].argsort()[-k:][::-1]]['text'].tolist()
ctx = retrieve(TICKET)
print('Retrieved:', ctx)
print(ask('Answer using ONLY the provided context.',
          f'Context:\n' + '\n'.join(ctx) + f'\n\nTicket: {TICKET}'))

# --------------------------------------------------------------------------
# STEP 4. Option C — fine tuning. Fine tuning is the right answer when the
# gap is BEHAVIOUR: house style, format, tone and consistency that no prompt
# reliably enforces at scale.
# --------------------------------------------------------------------------
print('Fine tuning is indicated when:')
for r in ['The required style cannot be described in a prompt, only shown.',
          'The same formatting must hold across thousands of calls.',
          'Prompt length (and therefore cost) must be reduced.',
          'Latency matters and long few-shot prompts are too slow.']:
    print('  -', r)

# --------------------------------------------------------------------------
# STEP 5. Score the three options against the stated constraints. The
# decision must be traceable to the problem statement, not to preference.
# --------------------------------------------------------------------------
options = pd.DataFrame([
 {'Option':'Prompt engineering','Cost':'None','Time':'Hours','Fixes':'Instruction following',
  'Latency':'Worse (long prompt)','Fit':'Partial - style drifts'},
 {'Option':'RAG','Cost':'Low','Time':'Days','Fixes':'Missing/changing facts',
  'Latency':'Worse (retrieval)','Fit':'Needed for account facts'},
 {'Option':'LoRA fine tune','Cost':'Low (1 GPU)','Time':'Hours','Fixes':'Style, format, tone',
  'Latency':'Best (short prompt)','Fit':'Direct match to the need'},
 {'Option':'Full fine tune','Cost':'Very high','Time':'Days','Fixes':'Same as LoRA',
  'Latency':'Best','Fit':'Violates the 16GB constraint'},
])
print(options.to_string(index=False))

# --------------------------------------------------------------------------
# STEP 6. MULTI-MODAL EMBEDDINGS — when the input is not text alone. A
# contrastive model places image and text in ONE shared vector space so they
# can be compared directly.
# --------------------------------------------------------------------------
os.system('pip install -q open_clip_torch')
from PIL import Image
import open_clip
clip, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_tok = open_clip.get_tokenizer('ViT-B-32')
clip = clip.eval().cuda()
print('CLIP loaded - image and text share one 512-dimensional space.')

# --------------------------------------------------------------------------
# STEP 7. Embed an image and several captions, then measure their
# similarity. A high score between an image and its true caption is the
# shared space working.
# --------------------------------------------------------------------------
img = preprocess(Image.open('data/router_photo.jpg')).unsqueeze(0).cuda()
captions = ['a photo of a wifi router with a red status light',
            'a photo of a cat sleeping on a sofa',
            'a screenshot of a mobile phone bill']
with torch.no_grad():
    ie = clip.encode_image(img);  ie /= ie.norm(dim=-1, keepdim=True)
    te = clip.encode_text(clip_tok(captions).cuda())
    te /= te.norm(dim=-1, keepdim=True)
    sims = (100.0 * ie @ te.T).softmax(dim=-1)[0]
for c, s in zip(captions, sims):
    print(f'{s:6.1%}  {c}')

# --------------------------------------------------------------------------
# STEP 8. Show why the shared space matters: a text query retrieves the
# right image without any image labels — this is the mechanism a multi-modal
# fine-tune adapts.
# --------------------------------------------------------------------------
print('Because image and text land in the SAME space, you can:')
for u in ['Retrieve images with a text query (and vice versa).',
          'Condition a generator on an image as easily as on a prompt.',
          'Fine-tune the projector so a vision encoder feeds an LLM.',
          'Score text-image alignment with CLIPScore as a training metric.']:
    print('  -', u)

# --------------------------------------------------------------------------
# STEP 9. Write the decision record. This document is the deliverable of
# this lab.
# --------------------------------------------------------------------------
decision = {**PROBLEM,
 'chosen': 'LoRA fine tuning of Llama-3.2, with RAG for account-specific facts',
 'rejected': {'full_fine_tune':'exceeds the single 16GB GPU constraint',
              'prompt_only'   :'style drifts across long conversations'},
 'why': ('The gap is behavioural (house style and format), which fine tuning fixes; '
         'factual account data changes daily and belongs in retrieval, not in weights. '
         'LoRA meets the 16GB constraint and keeps data in-region.'),
 'metrics': ['style match on held-out set','invented-fact rate','agent edit distance'],
}
json.dump(decision, open('adaptation_decision.json','w'), indent=2)
print(json.dumps(decision, indent=2))


print('\nLab 13 complete — All three options produce output on the same ticket, the option comparison table is printed, CLIP scores the c')
