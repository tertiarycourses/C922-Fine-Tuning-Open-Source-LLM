#!/usr/bin/env python3
"""
Lab 16 — Fine-Tune with QLoRA in 4-Bit
Fine Tuning Open Source LLM (C922)
LU3 — Modelling

Objective: Apply QLoRA 4-bit quantisation and measure the memory saving against LoRA

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: quantisation_memory.csv shows 4-bit using substantially less VRAM than 16-bit, the parameter dump shows Params4bit storage with a bf16 adapter, the QLoRA training completes with peak VRAM reported, and qlora_adapter/ is saved.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Measure the model at 16-bit. This is the baseline the quantisation
# savings are measured against.
# --------------------------------------------------------------------------
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
MODEL='unsloth/Llama-3.2-1B-Instruct'
def measure(**kw):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto', **kw)
    v = torch.cuda.max_memory_allocated()/1024**3
    del m; gc.collect(); torch.cuda.empty_cache()
    return v
v16 = measure(torch_dtype=torch.bfloat16)
print(f'16-bit (bf16): {v16:.2f} GB')

# --------------------------------------------------------------------------
# STEP 2. Measure at 8-bit, then at 4-bit NF4. NF4 is an information-
# theoretically optimal datatype for normally-distributed weights, which is
# why it beats plain int4.
# --------------------------------------------------------------------------
v8 = measure(quantization_config=BitsAndBytesConfig(load_in_8bit=True))
print(f'8-bit        : {v8:.2f} GB')
v4 = measure(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16))
print(f'4-bit NF4    : {v4:.2f} GB')

# --------------------------------------------------------------------------
# STEP 3. Add DOUBLE QUANTISATION, which quantises the quantisation
# constants themselves for a further saving of roughly 0.4 bits per
# parameter.
# --------------------------------------------------------------------------
v4d = measure(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True))
print(f'4-bit + double quant: {v4d:.2f} GB')
import pandas as pd
tbl = pd.DataFrame([
  {'Precision':'bf16','VRAM_GB':round(v16,2),'Saving_vs_16bit':'-'},
  {'Precision':'int8','VRAM_GB':round(v8,2), 'Saving_vs_16bit':f'{100*(1-v8/v16):.0f}%'},
  {'Precision':'nf4','VRAM_GB':round(v4,2),  'Saving_vs_16bit':f'{100*(1-v4/v16):.0f}%'},
  {'Precision':'nf4+dq','VRAM_GB':round(v4d,2),'Saving_vs_16bit':f'{100*(1-v4d/v16):.0f}%'},
])
print(tbl.to_string(index=False)); tbl.to_csv('quantisation_memory.csv', index=False)

# --------------------------------------------------------------------------
# STEP 4. Inspect what actually changed inside the model. The frozen base
# weights become 4-bit Params4bit while the compute dtype stays bf16 —
# storage is quantised, arithmetic is not.
# --------------------------------------------------------------------------
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map='auto')
for n, p in list(model.named_parameters())[:6]:
    print(f'{n[:55]:<55} {str(p.dtype):<16} {type(p).__name__}')
print('Weights stored in 4-bit, dequantised to bf16 for each matmul.')

# --------------------------------------------------------------------------
# STEP 5. Attach the LoRA adapter on top of the quantised base. The adapter
# itself stays in bf16 — you never train quantised parameters, only the
# small high-precision adapter.
# --------------------------------------------------------------------------
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                 task_type='CAUSAL_LM',
                 target_modules=['q_proj','k_proj','v_proj','o_proj',
                                 'gate_proj','up_proj','down_proj'])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
print('Adapter dtype:', next(p.dtype for n,p in model.named_parameters()
                             if 'lora_A' in n and p.requires_grad))

# --------------------------------------------------------------------------
# STEP 6. Train with a PAGED optimiser. Paged optimisers spill optimiser
# state to CPU RAM on a memory spike instead of crashing with an out-of-
# memory error.
# --------------------------------------------------------------------------
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
ds = load_dataset('json', data_files='formatted_train.jsonl')['train']
ds = ds.train_test_split(test_size=0.1, seed=3407)
args = SFTConfig(output_dir='qlora_out', per_device_train_batch_size=2,
    gradient_accumulation_steps=4, num_train_epochs=2, learning_rate=2e-4,
    warmup_steps=5, logging_steps=5, optim='paged_adamw_8bit',
    lr_scheduler_type='linear', weight_decay=0.01, seed=3407,
    bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
    max_seq_length=1024, dataset_text_field='text',
    gradient_checkpointing=True, report_to='none')

# --------------------------------------------------------------------------
# STEP 7. Run the QLoRA training and record peak memory. This is the
# headline number: a fine-tune that would need a datacentre GPU at 16-bit
# now runs on a free T4.
# --------------------------------------------------------------------------
torch.cuda.reset_peak_memory_stats()
trainer = SFTTrainer(model=model, args=args,
                     train_dataset=ds['train'], eval_dataset=ds['test'])
stats = trainer.train()
peak = torch.cuda.max_memory_allocated()/1024**3
print(f"QLoRA runtime   : {stats.metrics['train_runtime']/60:.1f} min")
print(f"QLoRA final loss: {stats.metrics['train_loss']:.4f}")
print(f'QLoRA peak VRAM : {peak:.2f} GB')

# --------------------------------------------------------------------------
# STEP 8. Compare quality against the 16-bit LoRA run from the previous lab.
# The QLoRA literature reports quality close to 16-bit LoRA — verify it on
# your own data rather than trusting the claim.
# --------------------------------------------------------------------------
PROMPT = 'My fibre broadband has been down since yesterday and no one has called back.'
ids = tok.apply_chat_template([{'role':'user','content':PROMPT}],
        add_generation_prompt=True, return_tensors='pt').to('cuda')
out = model.generate(ids, max_new_tokens=150, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print('QLoRA output:\n', tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))

# --------------------------------------------------------------------------
# STEP 9. Save the QLoRA adapter and record the decision table for when to
# choose each method.
# --------------------------------------------------------------------------
model.save_pretrained('qlora_adapter'); tok.save_pretrained('qlora_adapter')
guide = pd.DataFrame([
 {'Method':'Full fine tune','VRAM (7B)':'~60+ GB','Quality':'Baseline','Use when':'Large budget, maximum quality'},
 {'Method':'LoRA (bf16)','VRAM (7B)':'~16-20 GB','Quality':'~= full','Use when':'A100/L4 available'},
 {'Method':'QLoRA (nf4)','VRAM (7B)':'~6-8 GB','Quality':'~= LoRA','Use when':'Single consumer GPU / free Colab'},
])
print(guide.to_string(index=False)); guide.to_csv('qlora_decision_guide.csv', index=False)


print('\nLab 16 complete — quantisation_memory.csv shows 4-bit using substantially less VRAM than 16-bit, the parameter dump shows Params')
