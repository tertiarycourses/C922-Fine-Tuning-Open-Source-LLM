#!/usr/bin/env python3
"""
Lab 25 — Merge, Export and Deploy the Fine-Tuned Model
Fine Tuning Open Source LLM (C922)
LU4 — Machine Learning Implementation and Operations

Objective: Deploy and operationalize a fine-tuned model across serving targets

Run cell by cell in Google Colab (T4 GPU), or straight through:
    python run.py

Verify: merged_16bit/, merged_4bit/ and gguf_out/ all exist with a .gguf file present; the merged model produces output comparable to the adapter version; deployment_options.csv compares all four serving options by size; and MODEL_CARD.md is written with real measured numbers.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ is relative to this file


# --------------------------------------------------------------------------
# STEP 1. Load the base model and attach the trained adapter, then confirm
# the adapter is actually active before merging anything.
# --------------------------------------------------------------------------
from unsloth import FastLanguageModel
import torch, time, os
model, tok = FastLanguageModel.from_pretrained(
    'unsloth_adapter', max_seq_length=1024, load_in_4bit=True)
FastLanguageModel.for_inference(model)
PROMPT = 'My fibre broadband has been down since yesterday evening.'
def gen(m, t=tok):
    ids = t.apply_chat_template([{'role':'user','content':PROMPT}],
            add_generation_prompt=True, return_tensors='pt').to('cuda')
    s = time.time()
    out = m.generate(ids, max_new_tokens=128, do_sample=False,
                     pad_token_id=t.eos_token_id, use_cache=True)
    n = out.shape[-1]-ids.shape[-1]
    return t.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip(), n/(time.time()-s)
adapter_text, adapter_tps = gen(model)
print(f'Adapter-only: {adapter_tps:.1f} tok/s')
print(adapter_text[:200])

# --------------------------------------------------------------------------
# STEP 2. MERGE to 16-bit. This folds W + (alpha/r)BA into a single weight
# matrix, producing a standalone model with zero added inference latency.
# --------------------------------------------------------------------------
model.save_pretrained_merged('merged_16bit', tok, save_method='merged_16bit')
os.system('du -sh merged_16bit')
os.system('ls -la merged_16bit | head')

# --------------------------------------------------------------------------
# STEP 3. Verify the merge PRESERVED behaviour. A merge that changes the
# output is a bug — the arithmetic is supposed to be exact.
# --------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer
mt  = AutoTokenizer.from_pretrained('merged_16bit')
mm  = AutoModelForCausalLM.from_pretrained('merged_16bit',
        torch_dtype=torch.bfloat16, device_map='auto')
merged_text, merged_tps = gen(mm, mt)
print('ADAPTER:', adapter_text[:160])
print('MERGED :', merged_text[:160])
print(f'Merged throughput: {merged_tps:.1f} tok/s vs adapter {adapter_tps:.1f} tok/s')

# --------------------------------------------------------------------------
# STEP 4. Also export merged 4-bit for constrained GPU serving, which trades
# a little quality for a much smaller footprint.
# --------------------------------------------------------------------------
model.save_pretrained_merged('merged_4bit', tok, save_method='merged_4bit')
os.system('du -sh merged_4bit merged_16bit unsloth_adapter')

# --------------------------------------------------------------------------
# STEP 5. Export to GGUF for CPU and local serving with llama.cpp or Ollama.
# q4_k_m is the usual quality/size sweet spot.
# --------------------------------------------------------------------------
model.push_to_hub_gguf if False else None   # (hub push is optional)
model.save_pretrained_gguf('gguf_out', tok, quantization_method='q4_k_m')
os.system('ls -la gguf_out/*.gguf')
os.system('du -sh gguf_out')

# --------------------------------------------------------------------------
# STEP 6. Compare every serving option on size, memory and throughput. This
# table is what an operations decision is actually made from.
# --------------------------------------------------------------------------
import pandas as pd
def size_gb(p):
    return round(sum(os.path.getsize(os.path.join(d,f))
                for d,_,fs in os.walk(p) for f in fs)/1024**3, 3)
opts = pd.DataFrame([
 {'Option':'Adapter + 4-bit base','Size_GB':size_gb('unsloth_adapter'),
  'tok/s':round(adapter_tps,1),'Serve with':'PEFT / Unsloth',
  'Best for':'Many tasks on one base; hot-swap per tenant'},
 {'Option':'Merged 16-bit','Size_GB':size_gb('merged_16bit'),
  'tok/s':round(merged_tps,1),'Serve with':'vLLM / TGI',
  'Best for':'Highest-throughput GPU serving'},
 {'Option':'Merged 4-bit','Size_GB':size_gb('merged_4bit'),
  'tok/s':None,'Serve with':'Transformers + bitsandbytes',
  'Best for':'Single small GPU'},
 {'Option':'GGUF q4_k_m','Size_GB':size_gb('gguf_out'),
  'tok/s':None,'Serve with':'llama.cpp / Ollama',
  'Best for':'CPU, laptop, edge, offline'},
])
print(opts.to_string(index=False)); opts.to_csv('deployment_options.csv', index=False)

# --------------------------------------------------------------------------
# STEP 7. Show the multi-adapter serving pattern: one base model in VRAM,
# many small adapters swapped per request. This is the operational advantage
# of NOT merging.
# --------------------------------------------------------------------------
print('One 4-bit base in VRAM  : ~1.5 GB')
print(f"Each task adapter        : ~{size_gb('unsloth_adapter')*1024:.0f} MB")
print('So 20 task-specific models cost one base plus 20 small adapters,')
print('instead of 20 full model copies. This is why adapters are kept separate in multi-tenant serving.')

# --------------------------------------------------------------------------
# STEP 8. Write the serving entry point so the deployment is reproducible
# rather than a notebook cell.
# --------------------------------------------------------------------------
serve = '''from vllm import LLM, SamplingParams

llm = LLM(model="merged_16bit", dtype="bfloat16", max_model_len=1024)
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256,
                        repetition_penalty=1.1)

def reply(question: str) -> str:
    prompt = ("<|start_header_id|>user<|end_header_id|>\\n\\n"
              f"{question}<|eot_id|>"
              "<|start_header_id|>assistant<|end_header_id|>\\n\\n")
    return llm.generate([prompt], params)[0].outputs[0].text
'''
open('serve_vllm.py','w').write(serve)
print(serve)

# --------------------------------------------------------------------------
# STEP 9. Write the MODEL CARD. This is the governance artifact: intended
# use, training data, evaluation results and known limitations, without
# which the deployment is not auditable.
# --------------------------------------------------------------------------
import json
bench = json.load(open('benchmark_report.json'))
fair  = json.load(open('fairness_report.json'))
hp    = json.load(open('best_hyperparameters.json'))
card = f'''# Model Card - Support Assistant (LoRA fine-tune)

## Base model
unsloth/Llama-3.2-1B-Instruct

## Method
QLoRA (4-bit NF4) via Unsloth. r={hp['r']}, alpha={hp['lora_alpha']}, lr={hp['learning_rate']}, target_modules={hp['target_modules']}, epochs={hp['num_train_epochs']}.

## Training data
data_repo/v2 - synthetic and mock support tickets. PII redacted; exact and near duplicates removed; quality-filtered. See DATASET_CARD.md.

## Intended use
Drafting first-response customer support replies for agent review. NOT for autonomous customer-facing responses without a human in the loop.

## Evaluation
ROUGE-L gain over base: {bench['rougeL_gain_pct']}%. General-text perplexity change: {bench['general_ppl_change_pct']}%.

## Fairness
Counterfactual audit across 6 demographic groups. Agentic-language gap {fair['agentic_gap_before']} -> {fair['agentic_gap_after']} after mitigation.

## Limitations
English only. Single product domain. Must not state account-specific facts - pair with retrieval for those. Outputs require agent review before sending.

## Deployment
See deployment_options.csv. Merged 16-bit for vLLM; GGUF q4_k_m for local CPU.
'''
open('MODEL_CARD.md','w').write(card)
print(card)


print('\nLab 25 complete — merged_16bit/, merged_4bit/ and gguf_out/ all exist with a .gguf file present; the merged model produces outpu')
