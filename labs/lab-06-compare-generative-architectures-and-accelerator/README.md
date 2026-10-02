# Lab 6 — Compare Generative Architectures and Accelerator Options

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU1 — Data Engineering  
**Delivery:** self-paced extension  
**Objective:** Compare GAN, diffusion and autoregressive architectures and select accelerator and cloud platform

## Goal

Run a hands-on comparison of the three generative architecture families on the same small task, then measure the real memory and throughput cost of a fine-tune on the available accelerator so you can justify a hardware and cloud-platform choice with numbers rather than opinion.

## What you will build

A comparison table of the three architecture families on generation quality, training stability and compute cost, plus measured throughput and VRAM figures for the lab GPU.

**Tools:** PyTorch, Unsloth, diffusers, matplotlib, nvidia-smi

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-06-compare-generative-architectures-and-accelerator/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

> **Input:** this lab has no mock dataset of its own — it consumes the output of an earlier lab in the same Learning Unit. Run the labs in order.

## Step-by-step

### Step 1

Instrument an autoregressive transformer. This is the family every LLM fine-tune in this course uses: the sequence probability is factorised token by token.

```python
from unsloth import FastLanguageModel
import torch, time
model, tok = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=1024, load_in_4bit=True)
FastLanguageModel.for_inference(model)
print('Parameters:', sum(p.numel() for p in model.parameters())/1e9, 'B')
```

### Step 2

Measure autoregressive generation: one token at a time, each conditioned on all previous tokens. The sequential dependency is why generation latency scales with output length.

```python
prompt = 'Explain low-rank adaptation in one paragraph.'
ids = tok.apply_chat_template([{'role':'user','content':prompt}],
                              add_generation_prompt=True, return_tensors='pt').to('cuda')
torch.cuda.synchronize(); t0 = time.time()
out = model.generate(ids, max_new_tokens=128, do_sample=False)
torch.cuda.synchronize(); dt = time.time()-t0
n = out.shape[-1] - ids.shape[-1]
print(f'{n} tokens in {dt:.2f}s = {n/dt:.1f} tok/s')
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

### Step 3

Contrast with a diffusion model, which generates by iteratively denoising. Note that quality is traded directly against the number of denoising steps.

```python
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    'segmind/tiny-sd', torch_dtype=torch.float16).to('cuda')
for steps in [4, 12, 25]:
    t0 = time.time()
    img = pipe('a clean product photo of a laptop', num_inference_steps=steps).images[0]
    print(f'{steps:>3} denoising steps -> {time.time()-t0:.1f}s')
    img.save(f'diffusion_{steps}steps.png')
```

### Step 4

Reason about the conditional GAN family. A GAN generates in a single forward pass — the fastest of the three at inference — but the generator/discriminator game is the least stable to train.

```python
import torch.nn as nn
class TinyGenerator(nn.Module):
    def __init__(self, z=64, cls=10, out=784):
        super().__init__()
        self.emb = nn.Embedding(cls, cls)
        self.net = nn.Sequential(nn.Linear(z+cls,256), nn.ReLU(),
                                 nn.Linear(256,512),  nn.ReLU(),
                                 nn.Linear(512,out),  nn.Tanh())
    def forward(self, z, y):
        return self.net(torch.cat([z, self.emb(y)], dim=1))
g = TinyGenerator().cuda()
t0 = time.time()
fake = g(torch.randn(64,64).cuda(), torch.randint(0,10,(64,)).cuda())
torch.cuda.synchronize()
print(f'cGAN: 64 samples in ONE forward pass, {time.time()-t0:.4f}s, shape {tuple(fake.shape)}')
```

### Step 5

Measure the VRAM cost of each approach. VRAM, not raw FLOPs, is what decides whether a fine-tune fits on your accelerator.

```python
def vram(): return torch.cuda.max_memory_allocated()/1024**3
torch.cuda.reset_peak_memory_stats()
_ = model.generate(ids, max_new_tokens=64)
print(f'Autoregressive 1B (4-bit) peak: {vram():.2f} GB')
torch.cuda.reset_peak_memory_stats()
_ = pipe('test', num_inference_steps=4)
print(f'Diffusion (tiny-sd fp16) peak : {vram():.2f} GB')
```

### Step 6

Compare accelerators on paper against the measured figures. GPUs are general-purpose and VRAM-rich; TPUs are systolic-array ASICs that win on large dense matmul but need XLA-compatible code.

```python
import pandas as pd
acc = pd.DataFrame([
 {'Accelerator':'NVIDIA T4 (Colab free)','VRAM_GB':16,'Precision':'fp16/int4','Best for':'LoRA/QLoRA on 1B-8B'},
 {'Accelerator':'NVIDIA A100 40GB','VRAM_GB':40,'Precision':'bf16/fp16','Best for':'Full fine-tune up to 7B'},
 {'Accelerator':'NVIDIA H100 80GB','VRAM_GB':80,'Precision':'bf16/fp8','Best for':'Large-scale training'},
 {'Accelerator':'Google TPU v5e','VRAM_GB':16,'Precision':'bf16','Best for':'Large dense matmul via XLA'},
])
print(acc.to_string(index=False))
```

### Step 7

Build the architecture comparison table — the deliverable of this lab.

```python
arch = pd.DataFrame([
 {'Family':'Conditional GAN','Quality':'Sharp, can mode-collapse','Stability':'Low (adversarial game)',
  'Train cost':'Low','Inference':'Fastest (1 pass)','Typical use':'Fast image synthesis'},
 {'Family':'Diffusion','Quality':'Highest fidelity + diversity','Stability':'High (simple denoise loss)',
  'Train cost':'Very high','Inference':'Slow (many steps)','Typical use':'Text-to-image, audio'},
 {'Family':'Autoregressive Transformer','Quality':'State of the art for text','Stability':'High (cross-entropy)',
  'Train cost':'High','Inference':'Sequential','Typical use':'LLMs, code, multi-modal'},
])
print(arch.to_string(index=False))
arch.to_csv('architecture_comparison.csv', index=False)
```

### Step 8

Compare cloud training platforms and record the selection rationale. The choice is driven by accelerator availability, data residency and cost, not by preference.

```python
cloud = pd.DataFrame([
 {'Platform':'Google Colab','Accelerators':'T4/L4/A100','Managed':'No','Best for':'Learning, prototyping'},
 {'Platform':'AWS SageMaker','Accelerators':'A10G/A100/H100','Managed':'Yes','Best for':'Production pipelines'},
 {'Platform':'Google Vertex AI','Accelerators':'A100/H100/TPU','Managed':'Yes','Best for':'TPU + AutoML'},
 {'Platform':'Azure ML','Accelerators':'A100/H100','Managed':'Yes','Best for':'Microsoft estates'},
])
print(cloud.to_string(index=False))
cloud.to_csv('platform_comparison.csv', index=False)
```

## Test it

architecture_comparison.csv and platform_comparison.csv are written, the autoregressive throughput is reported in tokens/second, the diffusion step-count timings show latency rising with steps, and peak VRAM is reported for both model types.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0