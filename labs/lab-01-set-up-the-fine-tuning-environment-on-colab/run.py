#!/usr/bin/env python3
"""
Lab 1 — Set Up the Fine-Tuning Environment on Colab
Fine Tuning Open Source LLM (C922)
LU1 — Data Engineering

Objective: Prepare a reproducible GPU environment and verify the accelerator, VRAM and library stack

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: nvidia-smi shows a T4 (or better), torch.cuda.is_available() is True, env_manifest.json exists with all seven library versions filled in, and the test model loads and unloads without an out-of-memory error.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Open Google Colab and select a GPU runtime: Runtime > Change
# runtime type > T4 GPU. This is the accelerator every lab in this course
# assumes.
# --------------------------------------------------------------------------
#   (discussion step — no code)

# --------------------------------------------------------------------------
# STEP 2. Confirm the GPU is attached and inspect its model, total VRAM and
# driver version. If this command fails, no GPU is attached and you must fix
# the runtime before continuing.
# --------------------------------------------------------------------------
os.system('nvidia-smi')

# --------------------------------------------------------------------------
# STEP 3. Install the fine-tuning stack. Unsloth pulls compatible pinned
# versions of transformers, peft, trl and bitsandbytes, which avoids the
# version conflicts that break a hand-assembled install.
# --------------------------------------------------------------------------
os.system('pip install -q unsloth')
os.system('pip install -q --no-deps trl peft accelerate bitsandbytes')

# --------------------------------------------------------------------------
# STEP 4. Verify PyTorch sees the GPU from Python and read the exact compute
# capability and VRAM. torch.cuda.is_available() must return True.
# --------------------------------------------------------------------------
import torch
print('CUDA available :', torch.cuda.is_available())
print('Device          :', torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f'Total VRAM      : {props.total_memory/1024**3:.1f} GB')
print('Compute cap.    :', f'{props.major}.{props.minor}')

# --------------------------------------------------------------------------
# STEP 5. Record every library version. Pinning versions is what makes a
# fine-tune reproducible — an unpinned environment is the most common reason
# a run cannot be repeated three months later.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# STEP 6. Measure the free VRAM baseline before any model is loaded. Every
# later lab compares its peak memory against this number to show what LoRA
# and QLoRA actually save.
# --------------------------------------------------------------------------
free, total = torch.cuda.mem_get_info()
print(f'Free VRAM before loading: {free/1024**3:.2f} GB of {total/1024**3:.2f} GB')

# --------------------------------------------------------------------------
# STEP 7. Load a small model to confirm the whole stack works end to end,
# then release it. A clean load and unload proves the environment is ready
# for the labs that follow.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = 'unsloth/Llama-3.2-1B-Instruct',
    max_seq_length= 2048,
    load_in_4bit  = True,
)
print('Loaded OK:', model.config.model_type)
del model; torch.cuda.empty_cache()


print('\nLab 1 complete — nvidia-smi shows a T4 (or better), torch.cuda.is_available() is True, env_manifest.json exists with all seven ')
