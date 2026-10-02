# Fine Tuning Open Source LLM

Fine-tune an open source large language model on a free GPU — build and clean the dataset, adapt the model with LoRA, QLoRA and Unsloth, then evaluate it for quality, bias and benchmark performance.

| Course detail | Information |
|---|---|
| Course code | `C922` |
| Programme | Non-WSQ |
| Duration | 1 day · 7.5 instructional hours (9:30am – 5:30pm) |
| Mode | Physical, instructor-led, hands-on labs on Google Colab (T4 GPU) |
| Registration | **[View course details and register](https://www.tertiarycourses.com.sg/fine-tuning-open-source-llm.html)** |
| Provider | Tertiary Infotech Academy Pte Ltd · UEN 201200696W |

## About the course

Prompting and retrieval cannot teach a model your tone, terminology and output formats — fine tuning can. In this course you take an open source model end to end: choose the adaptation method, build a versioned and sanitised training set in the model's chat template, fine-tune with Hugging Face PEFT LoRA, QLoRA 4-bit quantisation and Unsloth, then benchmark the result against the base model, audit it for bias and export it for deployment. Everything runs on a free Google Colab T4 GPU — no local GPU and no paid service is required.

## Learning outcomes

- **LO1** — Apply transformation methods to ingest and prepare synthetic data using cloud-based model tools.
- **LO2** — Develop optimized data pipelines for feature engineering using various optimization strategies.
- **LO3** — Fine-tune pre-trained multi-modal models using advanced loss metrics and training strategies.
- **LO4** — Analyse ML solutions for bias, explainability, and alignment with performance benchmarks.

## Topics covered

| Learning Unit | Topics |
|---|---|
| **LU1 — Data Engineering** | Create data repositories for fine tuning · Identify and implement a dataset ingestion solution · Identify and implement a data transformation solution |
| **LU2 — Exploratory Data Analysis** | Sanitize and prepare data for modelling · Perform feature engineering for fine tuning · Analyze and visualize data for fine tuning insights |
| **LU3 — Modelling** | Frame business problems as fine-tuning problems · Select the appropriate model and adaptation method · Train and fine-tune models with LoRA, QLoRA and Unsloth · Perform hyperparameter optimization · Evaluate fine-tuned models |
| **LU4 — Machine Learning Implementation and Operations** | Build and benchmark generative solutions against state of the art · Recommend and implement bias mitigation and fairness interventions · Apply interpretability and explainability methods · Deploy and operationalize fine-tuned models |

## Labs

26 hands-on labs, each with a step-by-step `README.md`, a runnable `run.py` and its own mock dataset (or the output of an earlier lab). Six labs are run together in the one-day class; the other twenty are **self-paced extension work** with the same full instructions. See the [labs index](labs/README.md).

**LU1 — Data Engineering**

| # | Lab | Delivery |
|---|---|---|
| 1 | [Set Up the Fine-Tuning Environment on Colab](labs/lab-01-set-up-the-fine-tuning-environment-on-colab/) | In class |
| 2 | [Create a Versioned Data Repository for Fine Tuning](labs/lab-02-create-a-versioned-data-repository-for-fine-tuni/) | Self-paced |
| 3 | [Generate Synthetic Training Data](labs/lab-03-generate-synthetic-training-data/) | Self-paced |
| 4 | [Ingest Datasets from Multiple Sources](labs/lab-04-ingest-datasets-from-multiple-sources/) | Self-paced |
| 5 | [Transform Data into Chat and Instruction Formats](labs/lab-05-transform-data-into-chat-and-instruction-formats/) | In class |
| 6 | [Compare Generative Architectures and Accelerator Options](labs/lab-06-compare-generative-architectures-and-accelerator/) | Self-paced |

**LU2 — Exploratory Data Analysis**

| # | Lab | Delivery |
|---|---|---|
| 7 | [Clean and Sanitize a Fine-Tuning Dataset](labs/lab-07-clean-and-sanitize-a-fine-tuning-dataset/) | In class |
| 8 | [Feature Engineering for Fine Tuning](labs/lab-08-feature-engineering-for-fine-tuning/) | Self-paced |
| 9 | [Build an ETL Pipeline for Fine-Tuning Datasets](labs/lab-09-build-an-etl-pipeline-for-fine-tuning-datasets/) | Self-paced |
| 10 | [Analyse and Visualise a Fine-Tuning Dataset](labs/lab-10-analyse-and-visualise-a-fine-tuning-dataset/) | Self-paced |
| 11 | [Tune Decoding Parameters for Higher-Quality Output](labs/lab-11-tune-decoding-parameters-for-higher-quality-outp/) | Self-paced |
| 12 | [Transfer Learning, Distillation and Self-Supervised Learning](labs/lab-12-transfer-learning-distillation-and-self-supervis/) | Self-paced |

**LU3 — Modelling**

| # | Lab | Delivery |
|---|---|---|
| 13 | [Frame the Problem and Select the Adaptation Method](labs/lab-13-frame-the-problem-and-select-the-adaptation-meth/) | Self-paced |
| 14 | [Build LoRA from First Principles](labs/lab-14-build-lora-from-first-principles/) | Self-paced |
| 15 | [Fine-Tune with PEFT LoRA](labs/lab-15-fine-tune-with-peft-lora/) | In class |
| 16 | [Fine-Tune with QLoRA in 4-Bit](labs/lab-16-fine-tune-with-qlora-in-4-bit/) | Self-paced |
| 17 | [Accelerate Fine Tuning with Unsloth](labs/lab-17-accelerate-fine-tuning-with-unsloth/) | In class |
| 18 | [Monitor Training Stability and Convergence](labs/lab-18-monitor-training-stability-and-convergence/) | Self-paced |
| 19 | [Hyperparameter Optimization for LoRA](labs/lab-19-hyperparameter-optimization-for-lora/) | Self-paced |
| 20 | [Advanced Loss Functions and Generative Metrics](labs/lab-20-advanced-loss-functions-and-generative-metrics/) | Self-paced |

**LU4 — Machine Learning Implementation and Operations**

| # | Lab | Delivery |
|---|---|---|
| 21 | [Benchmark the Fine-Tuned Model Against State of the Art](labs/lab-21-benchmark-the-fine-tuned-model-against-state-of-/) | In class |
| 22 | [Detect and Mitigate Bias in Generative Outputs](labs/lab-22-detect-and-mitigate-bias-in-generative-outputs/) | Self-paced |
| 23 | [Interpretability and Explainability for Generative Models](labs/lab-23-interpretability-and-explainability-for-generati/) | Self-paced |
| 24 | [Intermediate Fine Tuning and Continuous Pre-Training](labs/lab-24-intermediate-fine-tuning-and-continuous-pre-trai/) | Self-paced |
| 25 | [Merge, Export and Deploy the Fine-Tuned Model](labs/lab-25-merge-export-and-deploy-the-fine-tuned-model/) | Self-paced |
| 26 | [Capstone — End-to-End Fine-Tuning Project](labs/lab-26-capstone-end-to-end-fine-tuning-project/) | Self-paced |

## Course package

| Item | Files |
|---|---|
| Slide deck (183 slides) | [PPTX](courseware/Fine%20Tuning%20Open%20Source%20LLM%20%28C922%29-v1.0.pptx) · [PDF](courseware/Fine%20Tuning%20Open%20Source%20LLM%20%28C922%29-v1.0.pdf) |
| Lesson Plan | [DOCX](courseware/LP-Fine%20Tuning%20Open%20Source%20LLM%20%28C922%29.docx) · [PDF](courseware/LP-Fine%20Tuning%20Open%20Source%20LLM%20%28C922%29.pdf) |
| Learner Guide | [DOCX](courseware/LG-Fine%20Tuning%20Open%20Source%20LLM%20%28C922%29.docx) · [PDF](courseware/LG-Fine%20Tuning%20Open%20Source%20LLM%20%28C922%29.pdf) · [Markdown](LG-Fine%20Tuning%20Open%20Source%20LLM%20%28C922%29.md) |
| Hands-on labs | [`labs/`](labs/) |

Open each lab in Google Colab with a T4 GPU runtime and install the stack first:

```bash
pip install unsloth
pip install --no-deps trl peft accelerate bitsandbytes
```

## Distribution

This repository publishes the learner-facing courseware: slides, Lesson Plan, Learner Guide and labs. Source reference material is kept private and is not part of this repository.

---

© 2026 Tertiary Infotech Academy Pte Ltd · UEN 201200696W · [www.tertiarycourses.com.sg](https://www.tertiarycourses.com.sg)
