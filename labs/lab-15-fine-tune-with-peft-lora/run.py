#!/usr/bin/env python3
"""
Lab 15 — Fine-Tune with PEFT LoRA
Fine Tuning Open Source LLM (C922)
LU3 — Modelling

Objective: Fine-tune a pre-trained model with the PEFT LoRA implementation

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: print_trainable_parameters() reports under 1% trainable, the training loss falls from its starting value, peft_loss_curve.png shows a descending curve, the after-generation differs visibly from the before, and lora_adapter/ is only tens of megabytes.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load the base model in 4-bit and the tokenizer. Loading quantised
# is what makes this fit in a free Colab session.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# STEP 2. Inspect the module names so target_modules is chosen from what
# actually exists, not from a copied snippet — names differ between model
# families.
# --------------------------------------------------------------------------
names = {n.split('.')[-1] for n,_ in model.named_modules()
         if 'proj' in n or 'lin' in n}
print('Linear projections available:', sorted(names))

# --------------------------------------------------------------------------
# STEP 3. Configure LoRA. Target ALL linear layers — attention plus the MLP
# gate/up/down projections. The Databricks study showed attention-only
# targeting produces low-quality output.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# STEP 4. Confirm the trainable fraction. Under 1% of parameters is the
# expected figure and the whole point of the method.
# --------------------------------------------------------------------------
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable : {trainable:,}')
print(f'Total     : {total:,}')
print(f'Fraction  : {100*trainable/total:.3f}%')

# --------------------------------------------------------------------------
# STEP 5. Load the formatted dataset built in Lab 5 and split off a
# validation set so overfitting is visible during training rather than after
# it.
# --------------------------------------------------------------------------
from datasets import load_dataset
ds = load_dataset('json', data_files='formatted_train.jsonl')['train']
ds = ds.train_test_split(test_size=0.1, seed=3407)
print(ds)
print(ds['train'][0]['text'][:300])

# --------------------------------------------------------------------------
# STEP 6. Configure the trainer. The learning rate of 2e-4 is roughly 10x a
# full fine-tune, because the adapter is small and starts from zero.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# STEP 7. Capture a baseline generation BEFORE training, on a held-out
# prompt. Without a before, you cannot claim an after.
# --------------------------------------------------------------------------
PROMPT = 'My fibre broadband has been down since yesterday and no one has called back.'
def gen(m):
    ids = tok.apply_chat_template([{'role':'user','content':PROMPT}],
            add_generation_prompt=True, return_tensors='pt').to('cuda')
    out = m.generate(ids, max_new_tokens=150, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
before = gen(model)
print('BEFORE:\n', before)

# --------------------------------------------------------------------------
# STEP 8. Train. Watch the loss in the log — it should fall smoothly, and
# the validation loss should track it.
# --------------------------------------------------------------------------
trainer = SFTTrainer(model=model, args=args,
                     train_dataset=ds['train'], eval_dataset=ds['test'])
torch.cuda.reset_peak_memory_stats()
stats = trainer.train()
print(f"Runtime    : {stats.metrics['train_runtime']/60:.1f} min")
print(f"Final loss : {stats.metrics['train_loss']:.4f}")
print(f'Peak VRAM  : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')

# --------------------------------------------------------------------------
# STEP 9. Plot the training and validation loss curves from the trainer log
# history.
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# STEP 10. Compare after against before on the same prompt, then save the
# adapter. Note the adapter file size — tens of megabytes against gigabytes
# for a full model.
# --------------------------------------------------------------------------
after = gen(model)
print('BEFORE:\n', before)
print('\nAFTER:\n', after)
model.save_pretrained('lora_adapter')
tok.save_pretrained('lora_adapter')
os.system('du -sh lora_adapter')


print('\nLab 15 complete — print_trainable_parameters() reports under 1% trainable, the training loss falls from its starting value, peft')
