# Fine Tuning Open Source LLM — Hands-On Labs

**Course code:** C922  
**Version:** v1.0 · 2 October 2026

26 hands-on labs across 4 Learning Units. In the one-day class, Labs 1, 5, 7, 15, 17 and 21 are run together; the other 20 are self-paced extension work. Each lab has its own folder containing a step-by-step `README.md`, a runnable `run.py`, and a `data/` folder. A lab either ships its own mock dataset in `data/`, or consumes the output of an earlier lab in the same Learning Unit (for example Lab 8 reads `cleaned_corpus.jsonl` produced by Lab 7) — so no lab depends on a live download or private data. Work through the labs in order within a Learning Unit.

## Environment

- **Runtime:** Google Colab (free T4 GPU) — the primary lab environment; no local GPU is required.
- **Alternatives:** Kaggle Notebooks (2x T4) or a local NVIDIA GPU with >= 12 GB VRAM are supported alternatives.
- **Stack:** Python 3.11+, PyTorch, Hugging Face transformers / datasets / peft / trl / accelerate, bitsandbytes, Unsloth, evaluate, matplotlib, pandas, scikit-learn.
- **Models:** unsloth/Llama-3.2-1B-Instruct and unsloth/Llama-3.2-3B-Instruct-bnb-4bit — small enough to fine-tune inside a free Colab session, and gated-model free.
- **Data:** Each lab either ships its own mock dataset in `labs/lab-NN-*/data/`, or reads the output written by an earlier lab in the same Learning Unit. No lab depends on a live download or a private dataset.

```bash
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
```

## Labs

### LU1 — Data Engineering  (25%)

*LO1*

| # | Lab | Delivery | Folder |
|---|---|---|---|
| 1 | Set Up the Fine-Tuning Environment on Colab | In class | [`lab-01-set-up-the-fine-tuning-environment-on-colab/`](lab-01-set-up-the-fine-tuning-environment-on-colab/) |
| 2 | Create a Versioned Data Repository for Fine Tuning | Self-paced | [`lab-02-create-a-versioned-data-repository-for-fine-tuni/`](lab-02-create-a-versioned-data-repository-for-fine-tuni/) |
| 3 | Generate Synthetic Training Data | Self-paced | [`lab-03-generate-synthetic-training-data/`](lab-03-generate-synthetic-training-data/) |
| 4 | Ingest Datasets from Multiple Sources | Self-paced | [`lab-04-ingest-datasets-from-multiple-sources/`](lab-04-ingest-datasets-from-multiple-sources/) |
| 5 | Transform Data into Chat and Instruction Formats | In class | [`lab-05-transform-data-into-chat-and-instruction-formats/`](lab-05-transform-data-into-chat-and-instruction-formats/) |
| 6 | Compare Generative Architectures and Accelerator Options | Self-paced | [`lab-06-compare-generative-architectures-and-accelerator/`](lab-06-compare-generative-architectures-and-accelerator/) |

### LU2 — Exploratory Data Analysis  (25%)

*LO2*

| # | Lab | Delivery | Folder |
|---|---|---|---|
| 7 | Clean and Sanitize a Fine-Tuning Dataset | In class | [`lab-07-clean-and-sanitize-a-fine-tuning-dataset/`](lab-07-clean-and-sanitize-a-fine-tuning-dataset/) |
| 8 | Feature Engineering for Fine Tuning | Self-paced | [`lab-08-feature-engineering-for-fine-tuning/`](lab-08-feature-engineering-for-fine-tuning/) |
| 9 | Build an ETL Pipeline for Fine-Tuning Datasets | Self-paced | [`lab-09-build-an-etl-pipeline-for-fine-tuning-datasets/`](lab-09-build-an-etl-pipeline-for-fine-tuning-datasets/) |
| 10 | Analyse and Visualise a Fine-Tuning Dataset | Self-paced | [`lab-10-analyse-and-visualise-a-fine-tuning-dataset/`](lab-10-analyse-and-visualise-a-fine-tuning-dataset/) |
| 11 | Tune Decoding Parameters for Higher-Quality Output | Self-paced | [`lab-11-tune-decoding-parameters-for-higher-quality-outp/`](lab-11-tune-decoding-parameters-for-higher-quality-outp/) |
| 12 | Transfer Learning, Distillation and Self-Supervised Learning | Self-paced | [`lab-12-transfer-learning-distillation-and-self-supervis/`](lab-12-transfer-learning-distillation-and-self-supervis/) |

### LU3 — Modelling  (30%)

*LO3*

| # | Lab | Delivery | Folder |
|---|---|---|---|
| 13 | Frame the Problem and Select the Adaptation Method | Self-paced | [`lab-13-frame-the-problem-and-select-the-adaptation-meth/`](lab-13-frame-the-problem-and-select-the-adaptation-meth/) |
| 14 | Build LoRA from First Principles | Self-paced | [`lab-14-build-lora-from-first-principles/`](lab-14-build-lora-from-first-principles/) |
| 15 | Fine-Tune with PEFT LoRA | In class | [`lab-15-fine-tune-with-peft-lora/`](lab-15-fine-tune-with-peft-lora/) |
| 16 | Fine-Tune with QLoRA in 4-Bit | Self-paced | [`lab-16-fine-tune-with-qlora-in-4-bit/`](lab-16-fine-tune-with-qlora-in-4-bit/) |
| 17 | Accelerate Fine Tuning with Unsloth | In class | [`lab-17-accelerate-fine-tuning-with-unsloth/`](lab-17-accelerate-fine-tuning-with-unsloth/) |
| 18 | Monitor Training Stability and Convergence | Self-paced | [`lab-18-monitor-training-stability-and-convergence/`](lab-18-monitor-training-stability-and-convergence/) |
| 19 | Hyperparameter Optimization for LoRA | Self-paced | [`lab-19-hyperparameter-optimization-for-lora/`](lab-19-hyperparameter-optimization-for-lora/) |
| 20 | Advanced Loss Functions and Generative Metrics | Self-paced | [`lab-20-advanced-loss-functions-and-generative-metrics/`](lab-20-advanced-loss-functions-and-generative-metrics/) |

### LU4 — Machine Learning Implementation and Operations  (20%)

*LO4*

| # | Lab | Delivery | Folder |
|---|---|---|---|
| 21 | Benchmark the Fine-Tuned Model Against State of the Art | In class | [`lab-21-benchmark-the-fine-tuned-model-against-state-of-/`](lab-21-benchmark-the-fine-tuned-model-against-state-of-/) |
| 22 | Detect and Mitigate Bias in Generative Outputs | Self-paced | [`lab-22-detect-and-mitigate-bias-in-generative-outputs/`](lab-22-detect-and-mitigate-bias-in-generative-outputs/) |
| 23 | Interpretability and Explainability for Generative Models | Self-paced | [`lab-23-interpretability-and-explainability-for-generati/`](lab-23-interpretability-and-explainability-for-generati/) |
| 24 | Intermediate Fine Tuning and Continuous Pre-Training | Self-paced | [`lab-24-intermediate-fine-tuning-and-continuous-pre-trai/`](lab-24-intermediate-fine-tuning-and-continuous-pre-trai/) |
| 25 | Merge, Export and Deploy the Fine-Tuned Model | Self-paced | [`lab-25-merge-export-and-deploy-the-fine-tuned-model/`](lab-25-merge-export-and-deploy-the-fine-tuned-model/) |
| 26 | Capstone — End-to-End Fine-Tuning Project | Self-paced | [`lab-26-capstone-end-to-end-fine-tuning-project/`](lab-26-capstone-end-to-end-fine-tuning-project/) |


---

© Tertiary Infotech Academy Pte Ltd · UEN: 201200696W