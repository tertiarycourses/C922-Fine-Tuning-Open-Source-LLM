#!/usr/bin/env python3
"""
Lab 17 — Accelerate Fine Tuning with Unsloth
Fine Tuning Open Source LLM (C922)
LU3 — Modelling

Objective: Use Unsloth to fine-tune faster and with less memory, and verify the speed-up

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: The training completes and reports runtime, peak reserved VRAM and the memory used for training; the response-only masking reports a non-zero count of masked prompt tokens; unsloth_benchmark.csv compares Unsloth against the PEFT run; and unsloth_adapter/ is saved.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load the model through Unsloth. FastLanguageModel patches the
# attention, MLP and cross-entropy kernels with hand-written Triton
# implementations at load time.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch, time
max_seq_length = 1024
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model, tok = FastLanguageModel.from_pretrained(
    model_name     = 'unsloth/Llama-3.2-1B-Instruct',
    max_seq_length = max_seq_length,
    dtype          = None,          # auto-detect bf16 vs fp16
    load_in_4bit   = True,
)
print(f'Load time: {time.time()-t0:.1f}s')
print(f'Load VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')

# --------------------------------------------------------------------------
# STEP 2. Attach LoRA through Unsloth's get_peft_model. Note
# use_gradient_checkpointing='unsloth', which is a custom offloading
# implementation that saves further memory on long sequences.
# --------------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ['q_proj','k_proj','v_proj','o_proj',
                                  'gate_proj','up_proj','down_proj'],
    lora_alpha                 = 32,
    lora_dropout               = 0,          # 0 is optimised in Unsloth
    bias                       = 'none',     # 'none' is optimised
    use_gradient_checkpointing = 'unsloth',  # Unsloth's own implementation
    random_state               = 3407,
    use_rslora                 = False,
    loftq_config               = None,
)
model.print_trainable_parameters()

# --------------------------------------------------------------------------
# STEP 3. Apply the Unsloth chat template. get_chat_template normalises the
# many model-specific formats into one interface, so switching base model
# does not rewrite your data code.
# --------------------------------------------------------------------------
from unsloth.chat_templates import get_chat_template
tok = get_chat_template(tok, chat_template='llama-3.1')
from datasets import load_dataset
ds = load_dataset('json', data_files='data_repo/v2/processed/train.jsonl')['train']
def fmt(batch):
    texts = []
    for ins, inp, out in zip(batch['instruction'], batch['input'], batch['output']):
        user = ins if not (inp or '').strip() else f'{ins}\n\n{inp}'
        texts.append(tok.apply_chat_template(
            [{'role':'user','content':user},{'role':'assistant','content':out}],
            tokenize=False))
    return {'text': texts}
ds = ds.map(fmt, batched=True)
print(ds[0]['text'][:400])

# --------------------------------------------------------------------------
# STEP 4. Configure the trainer with the same hyperparameters as the PEFT
# run, so the benchmark compares kernels rather than settings.
# --------------------------------------------------------------------------
from trl import SFTTrainer, SFTConfig
args = SFTConfig(
    output_dir='unsloth_out', per_device_train_batch_size=2,
    gradient_accumulation_steps=4, warmup_steps=5, num_train_epochs=2,
    learning_rate=2e-4, logging_steps=5, optim='adamw_8bit',
    weight_decay=0.01, lr_scheduler_type='linear', seed=3407,
    bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
    max_seq_length=max_seq_length, dataset_text_field='text', report_to='none')
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=args)

# --------------------------------------------------------------------------
# STEP 5. Train on the RESPONSE ONLY. Masking the prompt tokens out of the
# loss means the model is graded on what it must generate, not on repeating
# the question — usually a clear quality gain.
# --------------------------------------------------------------------------
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = '<|start_header_id|>user<|end_header_id|>\n\n',
    response_part    = '<|start_header_id|>assistant<|end_header_id|>\n\n',
)
sample = trainer.train_dataset[0]
masked = sum(1 for l in sample['labels'] if l == -100)
print(f'{masked}/{len(sample["labels"])} tokens masked out of the loss (the prompt).')

# --------------------------------------------------------------------------
# STEP 6. Record the memory baseline, then train and time it. These are the
# numbers the benchmark table is built from.
# --------------------------------------------------------------------------
gpu = torch.cuda.get_device_properties(0)
start_mem = torch.cuda.max_memory_reserved()/1024**3
print(f'GPU: {gpu.name}  Total: {gpu.total_memory/1024**3:.1f} GB  Reserved now: {start_mem:.2f} GB')
stats = trainer.train()

# --------------------------------------------------------------------------
# STEP 7. Report the Unsloth results in the standard form: runtime, peak
# memory and the memory attributable to training itself.
# --------------------------------------------------------------------------
used      = round(torch.cuda.max_memory_reserved()/1024**3, 3)
used_lora = round(used - start_mem, 3)
pct       = round(used/(gpu.total_memory/1024**3)*100, 2)
print(f"Runtime            : {stats.metrics['train_runtime']/60:.2f} min")
print(f"Final loss         : {stats.metrics['train_loss']:.4f}")
print(f'Peak reserved      : {used} GB ({pct}% of the GPU)')
print(f'Peak for training  : {used_lora} GB')
unsloth_result = {'runtime_min': stats.metrics['train_runtime']/60,
                  'peak_gb': used, 'loss': stats.metrics['train_loss']}

# --------------------------------------------------------------------------
# STEP 8. Build the benchmark table against the standard PEFT run from Lab
# 15. Use YOUR measured numbers, not the marketing figures.
# --------------------------------------------------------------------------
import pandas as pd, json
peft_result = json.load(open('peft_result.json')) if __import__('os').path.exists('peft_result.json') \
              else {'runtime_min': None, 'peak_gb': None, 'loss': None}
bench = pd.DataFrame([
  {'Pipeline':'Standard PEFT + Transformers', **{k: peft_result.get(k) for k in ['runtime_min','peak_gb','loss']}},
  {'Pipeline':'Unsloth', **unsloth_result},
])
print(bench.round(3).to_string(index=False))
json.dump(unsloth_result, open('unsloth_result.json','w'), indent=2)
bench.to_csv('unsloth_benchmark.csv', index=False)

# --------------------------------------------------------------------------
# STEP 9. Run inference through Unsloth's optimised path and save the
# adapter. for_inference switches the model into a fast 2x generation mode.
# --------------------------------------------------------------------------
FastLanguageModel.for_inference(model)
ids = tok.apply_chat_template(
    [{'role':'user','content':'My fibre broadband has been down since yesterday.'}],
    add_generation_prompt=True, return_tensors='pt').to('cuda')
out = model.generate(ids, max_new_tokens=150, use_cache=True)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
model.save_pretrained('unsloth_adapter'); tok.save_pretrained('unsloth_adapter')


print('\nLab 17 complete — The training completes and reports runtime, peak reserved VRAM and the memory used for training; the response-')
