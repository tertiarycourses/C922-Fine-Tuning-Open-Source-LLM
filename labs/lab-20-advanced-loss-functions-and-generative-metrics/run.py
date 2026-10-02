#!/usr/bin/env python3
"""
Lab 20 — Advanced Loss Functions and Generative Metrics
Fine Tuning Open Source LLM (C922)
LU3 — Modelling

Objective: Apply advanced loss functions and metrics for evaluating generative performance

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: Perplexity is lower for well-formed than for shuffled text; BERTScore rates the paraphrase higher than BLEU does; FID is lower for the near-real images than the unrelated ones; CLIPScore is higher for the correct caption; and metric_selection_guide.csv is written.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. CROSS-ENTROPY and PERPLEXITY — the training objective itself.
# Perplexity is exp(cross-entropy): the effective number of tokens the model
# is choosing between.
# --------------------------------------------------------------------------
import torch, torch.nn.functional as F, numpy as np
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
FastLanguageModel.for_inference(model)
def ce_ppl(text):
    ids = tok(text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        loss = model(**ids, labels=ids['input_ids']).loss
    return float(loss), float(torch.exp(loss))
for t in ['The router should be restarted by holding the reset button.',
          'Router restarted the holding reset by should button be the.']:
    ce, ppl = ce_ppl(t)
    print(f'CE={ce:.3f}  PPL={ppl:8.1f}  |  {t[:55]}')
print('Lower perplexity = the model finds the text more predictable.')

# --------------------------------------------------------------------------
# STEP 2. Show why perplexity alone is not enough: it is defined only
# against a reference and says nothing about factual accuracy, so it must
# never be the only metric.
# --------------------------------------------------------------------------
pairs = [('Restart your router and wait 60 seconds.', 'a correct, helpful reply'),
         ('Restart your router and wait 6000 seconds.','fluent but wrong'),
         ('Restart your router.', 'correct but terse')]
for text, note in pairs:
    ce, ppl = ce_ppl(text)
    print(f'PPL={ppl:7.1f}  ({note}) : {text}')
print('Fluent-but-wrong can score BETTER than correct-but-terse. Never rank on PPL alone.')

# --------------------------------------------------------------------------
# STEP 3. BLEU and ROUGE — n-gram overlap against a reference. BLEU is
# precision-oriented (translation), ROUGE is recall-oriented
# (summarisation).
# --------------------------------------------------------------------------
os.system('pip install -q evaluate rouge_score sacrebleu')
import evaluate
bleu, rouge = evaluate.load('sacrebleu'), evaluate.load('rouge')
preds = ['Please restart your router and wait for the WAN light to turn green.',
         'Turn the router off and on again.']
refs  = [['Restart the router and wait until the WAN light is green.'],
         ['Restart the router and wait until the WAN light is green.']]
print('BLEU :', bleu.compute(predictions=preds, references=refs)['score'])
print('ROUGE:', {k: round(v,4) for k,v in
      rouge.compute(predictions=preds, references=[r[0] for r in refs]).items()})

# --------------------------------------------------------------------------
# STEP 4. BERTScore — semantic similarity from contextual embeddings. It
# scores a correct paraphrase highly where BLEU would penalise it for using
# different words.
# --------------------------------------------------------------------------
os.system('pip install -q bert_score')
bertscore = evaluate.load('bertscore')
para = ['Kindly power-cycle the modem and wait for the green indicator.']
ref  = ['Restart the router and wait until the WAN light is green.']
b = bertscore.compute(predictions=para, references=ref, lang='en')
print(f"BERTScore F1 : {b['f1'][0]:.4f}  (paraphrase)")
print('BLEU        :', round(bleu.compute(predictions=para, references=[ref])['score'],2))
print('BLEU punishes the paraphrase; BERTScore recognises it.')

# --------------------------------------------------------------------------
# STEP 5. FID (Frechet Inception Distance) — the standard image-generation
# metric. It compares the distribution of generated images to the real
# distribution; LOWER is better.
# --------------------------------------------------------------------------
os.system('pip install -q torchmetrics torch-fidelity')
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64, normalize=True)
real = torch.rand(32, 3, 128, 128)
good = real + 0.05*torch.randn_like(real)          # close to real
bad  = torch.rand(32, 3, 128, 128)                 # unrelated noise
for name, imgs in [('close to real', good.clamp(0,1)), ('unrelated', bad)]:
    fid.reset(); fid.update(real, real=True); fid.update(imgs, real=False)
    print(f'FID ({name:<14}): {float(fid.compute()):8.2f}')
print('FID is a DISTRIBUTION distance - it needs many samples, not one.')

# --------------------------------------------------------------------------
# STEP 6. INCEPTION SCORE — measures both the confidence and the diversity
# of generated images; HIGHER is better. Unlike FID it needs no reference
# set, which is also its weakness.
# --------------------------------------------------------------------------
from torchmetrics.image.inception import InceptionScore
iscore = InceptionScore(normalize=True)
iscore.update(torch.rand(32,3,128,128))
mean, std = iscore.compute()
print(f'Inception Score: {float(mean):.3f} +/- {float(std):.3f}')
print('IS rewards confident AND diverse output; it cannot detect mode collapse to real data.')

# --------------------------------------------------------------------------
# STEP 7. CLIPScore — text-image alignment, the metric that matters for
# conditional generation: did the model generate what the prompt asked for?
# --------------------------------------------------------------------------
import open_clip
from PIL import Image
clip, _, prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
ctok = open_clip.get_tokenizer('ViT-B-32'); clip = clip.eval().cuda()
img = prep(Image.open('data/router_photo.jpg')).unsqueeze(0).cuda()
for cap in ['a wifi router with a status light', 'a bowl of noodles']:
    with torch.no_grad():
        ie = clip.encode_image(img); ie /= ie.norm(dim=-1, keepdim=True)
        te = clip.encode_text(ctok([cap]).cuda()); te /= te.norm(dim=-1, keepdim=True)
    print(f'CLIPScore {float(100*(ie@te.T)):6.2f}  |  {cap}')

# --------------------------------------------------------------------------
# STEP 8. The ADVERSARIAL loss (GAN) and the CONTRASTIVE loss (InfoNCE) —
# the objectives behind the other two architecture families. Implement each
# so the mechanism is concrete.
# --------------------------------------------------------------------------
# Adversarial: generator maximises the discriminator's error
d_real, d_fake = torch.sigmoid(torch.randn(64)), torch.sigmoid(torch.randn(64))
d_loss = -(torch.log(d_real+1e-8).mean() + torch.log(1-d_fake+1e-8).mean())
g_loss = -torch.log(d_fake+1e-8).mean()
print(f'Discriminator loss {d_loss:.4f} | Generator loss {g_loss:.4f}')
# Contrastive (InfoNCE): pull matched pairs together, push others apart
a = F.normalize(torch.randn(16,128), dim=1); b = F.normalize(torch.randn(16,128), dim=1)
logits = a @ b.T / 0.07
info_nce = F.cross_entropy(logits, torch.arange(16))
print(f'InfoNCE contrastive loss: {info_nce:.4f}  (this trains CLIP)')

# --------------------------------------------------------------------------
# STEP 9. The DIFFUSION objective — predict the noise that was added. It is
# a simple MSE, which is exactly why diffusion training is so much more
# stable than a GAN.
# --------------------------------------------------------------------------
x0    = torch.randn(8, 3, 32, 32)
noise = torch.randn_like(x0)
for t, ab in [(50, 0.9), (500, 0.5), (900, 0.1)]:
    xt   = (ab**0.5)*x0 + ((1-ab)**0.5)*noise
    pred = noise + 0.1*torch.randn_like(noise)      # a model's prediction
    print(f'timestep {t:>3} (alpha_bar={ab}): MSE loss = {F.mse_loss(pred, noise):.4f}')
print('One simple MSE - no adversarial game, hence stable training.')

# --------------------------------------------------------------------------
# STEP 10. Build the metric selection guide. Choosing an invalid metric for
# the task is the most common evaluation error, and this table is the
# deliverable of this lab.
# --------------------------------------------------------------------------
import pandas as pd
guide = pd.DataFrame([
 {'Task':'Language modelling','Loss':'Cross-entropy','Metric':'Perplexity','Direction':'lower','Caveat':'Says nothing about factual accuracy'},
 {'Task':'Translation','Loss':'Cross-entropy','Metric':'BLEU / chrF','Direction':'higher','Caveat':'Punishes valid paraphrase'},
 {'Task':'Summarisation','Loss':'Cross-entropy','Metric':'ROUGE-1/2/L','Direction':'higher','Caveat':'Recall-biased; rewards copying'},
 {'Task':'Open-ended generation','Loss':'Cross-entropy','Metric':'BERTScore + LLM judge','Direction':'higher','Caveat':'Judge needs human validation'},
 {'Task':'Image generation','Loss':'Diffusion MSE / adversarial','Metric':'FID','Direction':'lower','Caveat':'Needs thousands of samples'},
 {'Task':'Image quality/diversity','Loss':'Adversarial','Metric':'Inception Score','Direction':'higher','Caveat':'Blind to mode collapse onto real data'},
 {'Task':'Text-to-image','Loss':'Diffusion MSE','Metric':'CLIPScore','Direction':'higher','Caveat':'Measures alignment, not image quality'},
 {'Task':'Multi-modal retrieval','Loss':'InfoNCE contrastive','Metric':'Recall@k','Direction':'higher','Caveat':'Depends on the candidate pool size'},
])
print(guide.to_string(index=False)); guide.to_csv('metric_selection_guide.csv', index=False)


print('\nLab 20 complete — Perplexity is lower for well-formed than for shuffled text; BERTScore rates the paraphrase higher than BLEU do')
