# Lab 1 — Set Up the Fine-Tuning Environment on Colab

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU1 — Data Engineering  
**Delivery:** in class  
**Objective:** Prepare a reproducible GPU environment and verify the accelerator, VRAM and library stack

## Goal

Open a free Colab GPU runtime, install the Unsloth + PEFT + TRL stack, and verify the GPU, the available VRAM and every library version. You then record the environment in a reproducibility manifest so the same run can be repeated later.

## What you will build

A verified GPU notebook environment and an env_manifest.json recording the GPU model, VRAM, CUDA version and the exact version of every library used in the course.

**Tools:** Google Colab, NVIDIA T4, PyTorch, Unsloth, PEFT, TRL, bitsandbytes

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-01-set-up-the-fine-tuning-environment-on-colab/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

> **Input:** this lab has no mock dataset of its own — it consumes the output of an earlier lab in the same Learning Unit. Run the labs in order.

## Step-by-step

### Step 1

Open Google Colab and select a GPU runtime: Runtime > Change runtime type > T4 GPU. This is the accelerator every lab in this course assumes.

### Step 2

Confirm the GPU is attached and inspect its model, total VRAM and driver version. If this command fails, no GPU is attached and you must fix the runtime before continuing.

```python
!nvidia-smi
```

### Step 3

Install the fine-tuning stack. Unsloth pulls compatible pinned versions of transformers, peft, trl and bitsandbytes, which avoids the version conflicts that break a hand-assembled install.

```bash
!pip install -q unsloth
!pip install -q --no-deps trl peft accelerate bitsandbytes
```

### Step 4

Verify PyTorch sees the GPU from Python and read the exact compute capability and VRAM. torch.cuda.is_available() must return True.

```python
import torch
print('CUDA available :', torch.cuda.is_available())
print('Device          :', torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f'Total VRAM      : {props.total_memory/1024**3:.1f} GB')
print('Compute cap.    :', f'{props.major}.{props.minor}')
```

### Step 5

Record every library version. Pinning versions is what makes a fine-tune reproducible — an unpinned environment is the most common reason a run cannot be repeated three months later.

```python
import torch, transformers, peft, trl, datasets, platform, json
manifest = {
    'python'      : platform.python_version(),
    'torch'       : torch.__version__,
    'cuda'        : torch.version.cuda,
    'transformers': transformers.__version__,
    'peft'        : peft.__version__,
    'trl'         : trl.__version__,
    'datasets'    : datasets.__version__,
    'gpu'         : torch.cuda.get_device_name(0),
    'vram_gb'     : round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1),
}
json.dump(manifest, open('env_manifest.json','w'), indent=2)
print(json.dumps(manifest, indent=2))
```

### Step 6

Measure the free VRAM baseline before any model is loaded. Every later lab compares its peak memory against this number to show what LoRA and QLoRA actually save.

```python
free, total = torch.cuda.mem_get_info()
print(f'Free VRAM before loading: {free/1024**3:.2f} GB of {total/1024**3:.2f} GB')
```

### Step 7

Load a small model to confirm the whole stack works end to end, then release it. A clean load and unload proves the environment is ready for the labs that follow.

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = 'unsloth/Llama-3.2-1B-Instruct',
    max_seq_length= 2048,
    load_in_4bit  = True,
)
print('Loaded OK:', model.config.model_type)
del model; torch.cuda.empty_cache()
```

## Test it

nvidia-smi shows a T4 (or better), torch.cuda.is_available() is True, env_manifest.json exists with all seven library versions filled in, and the test model loads and unloads without an out-of-memory error.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0