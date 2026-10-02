# Lab 5 — Transform Data into Chat and Instruction Formats

**Course:** Fine Tuning Open Source LLM (C922)  
**Learning Unit:** LU1 — Data Engineering  
**Delivery:** in class  
**Objective:** Apply data transformation techniques that convert raw records into the model's training format

## Goal

Convert the ingested corpus into the exact text the model must see during training. You render the Alpaca template by hand, render the model's native chat template with apply_chat_template(), append the EOS token, and prove why a template mismatch silently ruins a fine-tune.

## What you will build

A formatted training dataset whose 'text' column is exactly what the trainer will tokenise, verified against the model's own chat template and confirmed to terminate with the EOS token.

**Tools:** Hugging Face datasets, transformers tokenizer, Unsloth chat templates

## Environment

- Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- Stack: Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.

## Files in this lab

```
lab-05-transform-data-into-chat-and-instruction-formats/
  README.md      this guide
  run.py         the runnable script for every step
  data/          this lab's mock dataset
```

> **Input:** this lab has no mock dataset of its own — it consumes the output of an earlier lab in the same Learning Unit. Run the labs in order.

## Step-by-step

### Step 1

Load the ingested corpus from the previous lab and the tokenizer for the target model. The tokenizer, not the documentation, is the authority on the correct format.

```python
from datasets import load_dataset
from unsloth import FastLanguageModel
ds = load_dataset('json', data_files='ingested_train.jsonl')['train']
model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=2048, load_in_4bit=True)
print('EOS token:', repr(tokenizer.eos_token))
```

### Step 2

Render the classic Alpaca instruction template. Note the two variants — with and without an input field — because an empty '### Input:' section teaches the model to expect blank inputs.

```python
ALPACA_WITH_INPUT = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}'''

ALPACA_NO_INPUT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}'''
```

### Step 3

Apply the Alpaca template and append the EOS token to EVERY example. Without EOS the model never learns where a response ends and will generate until it hits the token limit.

```python
EOS = tokenizer.eos_token
def to_alpaca(batch):
    texts = []
    for ins, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
        tpl = ALPACA_WITH_INPUT if (inp or '').strip() else ALPACA_NO_INPUT
        texts.append(tpl.format(instruction=ins, input=inp, output=out) + EOS)
    return {'text': texts}
alpaca_ds = ds.map(to_alpaca, batched=True)
print(alpaca_ds[0]['text'][:600])
```

### Step 4

Now render the model's NATIVE chat template instead. apply_chat_template() emits the exact special tokens the base model was trained with — for Llama 3 these are the <|start_header_id|> role markers, which the Alpaca template does not contain.

```python
def to_chatml(batch):
    texts = []
    for ins, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
        user = ins if not (inp or '').strip() else f'{ins}\n\n{inp}'
        msgs = [{'role':'user','content':user},
                {'role':'assistant','content':out}]
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
    return {'text': texts}
chat_ds = ds.map(to_chatml, batched=True)
print(chat_ds[0]['text'][:600])
```

### Step 5

Compare the two renderings side by side. This is the core lesson of the lab: the same record produces very different token sequences, and only one matches what the base model expects.

```python
print('--- ALPACA ---'); print(repr(alpaca_ds[0]['text'][:200]))
print('--- NATIVE CHAT ---'); print(repr(chat_ds[0]['text'][:200]))
print('Alpaca tokens :', len(tokenizer(alpaca_ds[0]['text'])['input_ids']))
print('Chat   tokens :', len(tokenizer(chat_ds[0]['text'])['input_ids']))
```

### Step 6

Verify the special tokens actually made it into the token IDs. Decoding back with skip_special_tokens=False is the only way to see them.

```python
ids = tokenizer(chat_ds[0]['text'])['input_ids']
print(tokenizer.decode(ids[:40], skip_special_tokens=False))
print('Ends with EOS:', ids[-1] == tokenizer.eos_token_id or
      tokenizer.decode(ids[-5:], skip_special_tokens=False))
```

### Step 7

Analyse the token-length distribution of the formatted text. This number sets max_seq_length in the training lab — set it at the 95th-99th percentile.

```python
import numpy as np
lens = [len(tokenizer(t)['input_ids']) for t in chat_ds['text'][:500]]
for p in [50,90,95,99,100]:
    print(f'p{p:<3} = {int(np.percentile(lens,p)):>5} tokens')
print('Recommended max_seq_length:', int(np.percentile(lens,99)//256+1)*256)
```

### Step 8

Drop examples that exceed the chosen sequence length rather than letting the trainer truncate them — a truncated example teaches the model to produce an unfinished answer.

```python
MAX_LEN = 1024
before = len(chat_ds)
chat_ds = chat_ds.filter(lambda r: len(tokenizer(r['text'])['input_ids']) <= MAX_LEN)
print(f'Dropped {before-len(chat_ds)} over-length examples; {len(chat_ds)} remain.')
chat_ds.to_json('formatted_train.jsonl', orient='records', lines=True, force_ascii=False)
```

## Test it

formatted_train.jsonl exists with a 'text' column; printing one row shows the Llama 3 header tokens; the token-length percentile table is produced; and no remaining example exceeds MAX_LEN tokens.

---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W · C922 · v1.0