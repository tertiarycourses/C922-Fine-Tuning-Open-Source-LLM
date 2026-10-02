# Lab 15 — Fine-Tune with PEFT LoRA

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU3 — Modelling  
**Delivery:** in class  
**Objective:** Fine-tune a pre-trained model with the PEFT LoRA implementation

## Goal

Run a real LoRA fine-tune with the Hugging Face PEFT library on the dataset built in LU1 and LU2: configure LoraConfig, inspect the trainable parameter count, train with SFTTrainer, and compare the base and adapted model on held-out prompts.

## What you will build

A trained LoRA adapter of a few tens of megabytes, a training-loss curve, and a side-by-side before/after comparison on held-out prompts.

**Tools:** Hugging Face PEFT, transformers, TRL SFTTrainer, bitsandbytes, datasets

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-15-fine-tune-with-peft-lora/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

> **Input:** this lab has no mock dataset of its own — it consumes the output of an earlier lab in the same Learning Unit. Run the labs in order.

## Step-by-step

### Step 1

Load the base model in 4-bit and the tokenizer. Loading quantised is what makes this fit in a free Colab session.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
MODEL = 'unsloth/Llama-3.2-1B-Instruct'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                             device_map='auto')
print(f'Base VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
```

### Step 2

Inspect the module names so target_modules is chosen from what actually exists, not from a copied snippet — names differ between model families.

```python
names = {n.split('.')[-1] for n,_ in model.named_modules()
         if 'proj' in n or 'lin' in n}
print('Linear projections available:', sorted(names))
```

### Step 3

Configure LoRA. Target ALL linear layers — attention plus the MLP gate/up/down projections. The Databricks study showed attention-only targeting produces low-quality output.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
cfg = LoraConfig(
    r              = 16,
    lora_alpha     = 32,               # alpha/r = 2.0
    lora_dropout   = 0.05,
    bias           = 'none',
    task_type      = 'CAUSAL_LM',
    target_modules = ['q_proj','k_proj','v_proj','o_proj',
                      'gate_proj','up_proj','down_proj'],
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
```

### Step 4

Confirm the trainable fraction. Under 1% of parameters is the expected figure and the whole point of the method.

```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable : {trainable:,}')
print(f'Total     : {total:,}')
print(f'Fraction  : {100*trainable/total:.3f}%')
```

### Step 5

Load the formatted dataset built in Lab 5 and split off a validation set so overfitting is visible during training rather than after it.

```python
from datasets import load_dataset
ds = load_dataset('json', data_files='formatted_train.jsonl')['train']
ds = ds.train_test_split(test_size=0.1, seed=3407)
print(ds)
print(ds['train'][0]['text'][:300])
```

### Step 6

Configure the trainer. The learning rate of 2e-4 is roughly 10x a full fine-tune, because the adapter is small and starts from zero.

```python
from trl import SFTTrainer, SFTConfig
args = SFTConfig(
    output_dir                  = 'lora_out',
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,      # effective batch = 8
    warmup_steps                = 5,
    num_train_epochs            = 2,
    learning_rate               = 2e-4,
    logging_steps               = 5,
    eval_strategy               = 'steps',
    eval_steps                  = 25,
    optim                       = 'adamw_8bit',
    weight_decay                = 0.01,
    lr_scheduler_type           = 'linear',
    seed                        = 3407,
    bf16                        = torch.cuda.is_bf16_supported(),
    fp16                        = not torch.cuda.is_bf16_supported(),
    max_seq_length              = 1024,
    dataset_text_field          = 'text',
    report_to                   = 'none',
)
```

### Step 7

Capture a baseline generation BEFORE training, on a held-out prompt. Without a before, you cannot claim an after.

```python
PROMPT = 'My fibre broadband has been down since yesterday and no one has called back.'
def gen(m):
    ids = tok.apply_chat_template([{'role':'user','content':PROMPT}],
            add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = m.generate(ids, max_new_tokens=150, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
before = gen(model)
print('BEFORE:\n', before)
```

### Step 8

Train. Watch the loss in the log — it should fall smoothly, and the validation loss should track it.

```python
trainer = SFTTrainer(model=model, args=args,
                     train_dataset=ds['train'], eval_dataset=ds['test'])
torch.cuda.reset_peak_memory_stats()
stats = trainer.train()
print(f"Runtime    : {stats.metrics['train_runtime']/60:.1f} min")
print(f"Final loss : {stats.metrics['train_loss']:.4f}")
print(f'Peak VRAM  : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
```

### Step 9

Plot the training and validation loss curves from the trainer log history.

```python
import matplotlib.pyplot as plt
hist = trainer.state.log_history
tr = [(h['step'],h['loss'])      for h in hist if 'loss'      in h]
ev = [(h['step'],h['eval_loss']) for h in hist if 'eval_loss' in h]
plt.figure(figsize=(8,4))
plt.plot(*zip(*tr), label='train', color='#1F6FEB')
if ev: plt.plot(*zip(*ev), 'o-', label='validation', color='#F59E0B')
plt.xlabel('step'); plt.ylabel('loss'); plt.legend(); plt.grid(alpha=.3)
plt.title('LoRA fine-tuning loss')
plt.savefig('peft_loss_curve.png', dpi=120, bbox_inches='tight')
```

### Step 10

Compare after against before on the same prompt, then save the adapter. Note the adapter file size — tens of megabytes against gigabytes for a full model.

```python
after = gen(model)
print('BEFORE:\n', before)
print('\nAFTER:\n', after)
model.save_pretrained('lora_adapter')
tok.save_pretrained('lora_adapter')
!du -sh lora_adapter
```

## Test it

print_trainable_parameters() reports under 1% trainable, the training loss falls from its starting value, peft_loss_curve.png shows a descending curve, the after-generation differs visibly from the before, and lora_adapter/ is only tens of megabytes.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0