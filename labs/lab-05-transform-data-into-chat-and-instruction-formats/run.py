#!/usr/bin/env python3
"""
Lab 5 — Transform Data into Chat and Instruction Formats
Fine Tuning Open Source LLM (C922)
LU1 — Data Engineering

Objective: Apply data transformation techniques that convert raw records into the model's training format

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: formatted_train.jsonl exists with a 'text' column; printing one row shows the Llama 3 header tokens; the token-length percentile table is produced; and no remaining example exceeds MAX_LEN tokens.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load the ingested corpus from the previous lab and the tokenizer
# for the target model. The tokenizer, not the documentation, is the
# authority on the correct format.
# --------------------------------------------------------------------------
from datasets import load_dataset
from unsloth import FastLanguageModel
ds = load_dataset('json', data_files='ingested_train.jsonl')['train']
model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/Llama-3.2-1B-Instruct', max_seq_length=2048, load_in_4bit=True)
print('EOS token:', repr(tokenizer.eos_token))

# --------------------------------------------------------------------------
# STEP 2. Render the classic Alpaca instruction template. Note the two
# variants — with and without an input field — because an empty '### Input:'
# section teaches the model to expect blank inputs.
# --------------------------------------------------------------------------
ALPACA_WITH_INPUT = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}'''

ALPACA_NO_INPUT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}'''

# --------------------------------------------------------------------------
# STEP 3. Apply the Alpaca template and append the EOS token to EVERY
# example. Without EOS the model never learns where a response ends and will
# generate until it hits the token limit.
# --------------------------------------------------------------------------
EOS = tokenizer.eos_token
def to_alpaca(batch):
    texts = []
    for ins, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
        tpl = ALPACA_WITH_INPUT if (inp or '').strip() else ALPACA_NO_INPUT
        texts.append(tpl.format(instruction=ins, input=inp, output=out) + EOS)
    return {'text': texts}
alpaca_ds = ds.map(to_alpaca, batched=True)
print(alpaca_ds[0]['text'][:600])

# --------------------------------------------------------------------------
# STEP 4. Now render the model's NATIVE chat template instead.
# apply_chat_template() emits the exact special tokens the base model was
# trained with — for Llama 3 these are the <|start_header_id|> role markers,
# which the Alpaca template does not contain.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# STEP 5. Compare the two renderings side by side. This is the core lesson
# of the lab: the same record produces very different token sequences, and
# only one matches what the base model expects.
# --------------------------------------------------------------------------
print('--- ALPACA ---'); print(repr(alpaca_ds[0]['text'][:200]))
print('--- NATIVE CHAT ---'); print(repr(chat_ds[0]['text'][:200]))
print('Alpaca tokens :', len(tokenizer(alpaca_ds[0]['text'])['input_ids']))
print('Chat   tokens :', len(tokenizer(chat_ds[0]['text'])['input_ids']))

# --------------------------------------------------------------------------
# STEP 6. Verify the special tokens actually made it into the token IDs.
# Decoding back with skip_special_tokens=False is the only way to see them.
# --------------------------------------------------------------------------
ids = tokenizer(chat_ds[0]['text'])['input_ids']
print(tokenizer.decode(ids[:40], skip_special_tokens=False))
print('Ends with EOS:', ids[-1] == tokenizer.eos_token_id or
      tokenizer.decode(ids[-5:], skip_special_tokens=False))

# --------------------------------------------------------------------------
# STEP 7. Analyse the token-length distribution of the formatted text. This
# number sets max_seq_length in the training lab — set it at the 95th-99th
# percentile.
# --------------------------------------------------------------------------
import numpy as np
lens = [len(tokenizer(t)['input_ids']) for t in chat_ds['text'][:500]]
for p in [50,90,95,99,100]:
    print(f'p{p:<3} = {int(np.percentile(lens,p)):>5} tokens')
print('Recommended max_seq_length:', int(np.percentile(lens,99)//256+1)*256)

# --------------------------------------------------------------------------
# STEP 8. Drop examples that exceed the chosen sequence length rather than
# letting the trainer truncate them — a truncated example teaches the model
# to produce an unfinished answer.
# --------------------------------------------------------------------------
MAX_LEN = 1024
before = len(chat_ds)
chat_ds = chat_ds.filter(lambda r: len(tokenizer(r['text'])['input_ids']) <= MAX_LEN)
print(f'Dropped {before-len(chat_ds)} over-length examples; {len(chat_ds)} remain.')
chat_ds.to_json('formatted_train.jsonl', orient='records', lines=True, force_ascii=False)


print('\nLab 5 complete — formatted_train.jsonl exists with a \'text\' column; printing one row shows the Llama 3 header tokens; the tok')
