# DiCo-NLI UR2PhD Starter Assignment

This repository contains my work for the **UR2PhD Starter Assignment** on
**SemEval-2027 Task 2: Directional-Consistent Fine-Grained Natural Language
Inference (DiCo-NLI), Track 1 (English).**

The goal of the assignment was to explore the DiCo-NLI task, implement and
compare two different NLI systems, evaluate them using the official task
metrics, and analyze their classification and directional-consistency errors.

The two systems evaluated were:

- **System A:** Fine-tuned `distilbert-base-uncased`
- **System B:** Prompted `Qwen/Qwen2.5-1.5B-Instruct`

All final results were evaluated on the official **660-item Track 1 development
set** using the official DiCo-NLI scorer.

---

## Task Overview

DiCo-NLI is a fine-grained natural language inference task in which the order
of the two input texts matters. Each ordered text pair is assigned one of four
labels:

- `FORWARD_ENTAILMENT`
- `BACKWARD_ENTAILMENT`
- `EQUIVALENCE`
- `NEGATIVE_OTHER`

An important part of the task is **directional consistency**. Each example is
linked to its reversed counterpart using `instance_id` and `reverse_pair_id`.

For directional relationships, reversing the text pair should also reverse the
label:

```text
FORWARD_ENTAILMENT  <->  BACKWARD_ENTAILMENT
```

while the symmetric labels remain unchanged:

```text
EQUIVALENCE     -> EQUIVALENCE
NEGATIVE_OTHER  -> NEGATIVE_OTHER
```

The notebook begins by exploring the train and development label
distributions and examining examples of each label and their reversed pairs.

---

## System A — Fine-Tuned Encoder

### Initial DeBERTa Attempt

The recommended encoder for the assignment was:

```text
microsoft/deberta-v3-base
```

I initially attempted to fine-tune DeBERTa-v3-base as a four-class
sequence-pair classifier. However, training repeatedly became unstable when
using FP16 mixed precision in the available Google Colab T4 environment,
producing `NaN` losses and preventing a reliable training run.

Running the model in full FP32 precision would have required more GPU memory
and training time than was available.

Following the assignment guidelines, I therefore used:

```text
distilbert-base-uncased
```

for the final System A experiments.

### Training Configuration

DistilBERT was fine-tuned on the official Track 1 training split using:

| Hyperparameter | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Number of labels | 4 |
| Epochs | 4 |
| Learning rate | `3e-5` |
| Batch size | 16 |
| Maximum sequence length | 128 |
| Weight decay | 0.01 |
| Precision | FP16 |
| Seeds | 42, 123 |

The four labels were mapped to numerical IDs in the notebook:

```python
label2id = {
    "FORWARD_ENTAILMENT": 0,
    "BACKWARD_ENTAILMENT": 1,
    "EQUIVALENCE": 2,
    "NEGATIVE_OTHER": 3
}
```

The text pairs were tokenized together using the DistilBERT tokenizer:

```python
def preprocess_function(examples):
    return tokenizer(
        examples["text1"],
        examples["text2"],
        truncation=True,
        max_length=128
    )
```

Two independent runs were performed using random seeds **42** and **123**.
Development predictions from each run were saved separately and evaluated
with the official scorer.

---

## System B — Prompted LLM

System B used:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model was evaluated **without fine-tuning**.

I compared two prompting strategies:

1. **Zero-shot prompting**
2. **6-shot prompting**

The prompt explained the four possible DiCo-NLI labels and instructed the
model to return one label for each text pair.

For the 6-shot experiment, the demonstrations were selected from the official
Track 1 training split. The examples covered all four classes and included
directional relationships to demonstrate how entailment changes when the
order of a pair is reversed.

Six examples were used because this provided coverage of the four labels while
remaining within the assignment's recommended range of 4–8 demonstrations.

### Output Parsing

Because an instruction-tuned language model may generate additional text
instead of returning only a label, I implemented a parser that maps the
generated response to one of the four valid labels.

All **660 development examples were successfully parsed** for both prompting
variants:

```text
Zero-shot: 0 unparsed outputs
6-shot:    0 unparsed outputs
```

A fallback to `NEGATIVE_OTHER` was implemented for any output that could not
be parsed.

---

## Evaluation

All systems were evaluated on the official **660-item Track 1 development
set** using the official DiCo-NLI evaluation script.

The three official metrics were:

- **Weighted F1** — overall classification performance while accounting for
  class frequencies.
- **SoftCons** — measures whether predictions remain logically consistent when
  the order of a text pair is reversed.
- **HardCons** — requires the predictions for the reversed pair to be both
  consistent and correct.

The official scorer output was used directly when reporting the final results.

### Development-Set Results

| System | Weighted F1 | SoftCons | HardCons |
|---|---:|---:|---:|
| Pilot Reference (DeBERTa-v3-base) | 0.7500 | 0.8400 | 0.7900 |
| System A — DistilBERT (Seed 42) | 0.5873 | 0.6137 | 0.5235 |
| System A — DistilBERT (Seed 123) | **0.6119** | **0.6426** | **0.5487** |
| System A — Mean ± Spread | **0.5996 ± 0.0123** | **0.6282 ± 0.0144** | **0.5361 ± 0.0126** |
| System B — Qwen (Zero-Shot) | 0.2088 | 0.0289 | 0.0144 |
| System B — Qwen (6-Shot) | 0.2410 | 0.3213 | 0.1264 |

System A substantially outperformed System B across all three metrics.
Among the DistilBERT runs, **Seed 123 achieved the strongest performance on
all three official metrics**.

For System B, adding six demonstrations improved both classification
performance and directional consistency, although its results remained below
those of the fine-tuned encoder.

---

## Analysis

### Confusion Matrices

The notebook generates normalized confusion matrices for the strongest
configuration of each system:

- **System A:** DistilBERT, Seed 123
- **System B:** Qwen, 6-shot

DistilBERT performed best on `BACKWARD_ENTAILMENT` (**73.7%**) and had the
most difficulty with `NEGATIVE_OTHER` (**39.6%** correct).

Qwen showed substantially greater difficulty with directional entailment. In
the 6-shot experiment, it did not predict `BACKWARD_ENTAILMENT` for any
development example. Instead, many directional examples were classified as
`EQUIVALENCE`.

### Qualitative Error Analysis

I examined 10 incorrect predictions from the strongest System A run
(**Seed 123**).

A recurring error involved determining the correct direction of entailment
between closely related phrases. For example:

```text
Text 1: anti-protest laws
Text 2: anti-protest law

Gold:      FORWARD_ENTAILMENT
Predicted: BACKWARD_ENTAILMENT
```

A similar pattern appeared for pairs involving quantities:

```text
Text 1: Nine children
Text 2: Five children

Gold:      FORWARD_ENTAILMENT
Predicted: BACKWARD_ENTAILMENT
```

High lexical and semantic similarity also sometimes caused DistilBERT to
predict `EQUIVALENCE` when the gold relationship was directional. Examples
included pairs such as **"Pope Francis" / "pope"** and
**"in destroying" / "destroy"**.

Overall, the model generally recognized that the phrases were related but
sometimes struggled to determine the precise direction of that relationship.

### SoftCons Failure Analysis

I also examined five reversed pairs where the strongest System A run failed
SoftCons.

A clear pattern appeared in four of the five examples: DistilBERT correctly
predicted `BACKWARD_ENTAILMENT` for the original pair but continued to predict
`BACKWARD_ENTAILMENT` after the pair was reversed.

For example:

```text
Original:
anti-protest law -> anti-protest laws
Gold:      BACKWARD_ENTAILMENT
Predicted: BACKWARD_ENTAILMENT

Reversed:
anti-protest laws -> anti-protest law
Gold:      FORWARD_ENTAILMENT
Predicted: BACKWARD_ENTAILMENT
```

This suggests that the model often identified that a directional relationship
existed but did not reliably adjust the **direction** of its prediction after
the input order changed.

---

## Proposed Next Step

A natural next experiment would be **symmetric pair augmentation**.

For every training example `(A, B)`, I would explicitly add its reversed pair
`(B, A)` with the corresponding reversed label.

For example:

```text
(A, B) -> FORWARD_ENTAILMENT
(B, A) -> BACKWARD_ENTAILMENT
```

Training explicitly on both directions could help the model learn the
relationship between an example and its reversal rather than treating the two
orders as independent classification problems.

Based on the observed SoftCons failures, I would expect this to improve
directional consistency and potentially improve both **SoftCons** and
**HardCons**.

A second limitation of the current experiment is that the recommended
DeBERTa-v3-base model could not be fully evaluated because of the available
computational resources. Future work could therefore investigate whether the
same directional-consistency errors occur with a stronger encoder.

---

## Repository Structure

```text
DiCo-NLI-UR2PhD/
├── README.md
├── dico_nli_assignment_CAMILA.ipynb
└── predictions/
    ├── system_a_seed42.csv
    ├── system_a_seed123.csv
    ├── system_b_zeroshot.csv
    └── system_b_6shot.csv
```

### Notebook Structure

The Colab notebook is organized into four main sections:

```text
Part 0 — Setup
Part 1 — Data Exploration
Part 2 — System A: Fine-Tuned DistilBERT
Part 3 — System B: Prompted Qwen2.5-1.5B
Part 4 — Analysis
```

Part 4 includes the confusion matrices, 10-error qualitative analysis,
SoftCons failure analysis, and proposed next step.

---

## Prediction Files

The `predictions/` directory contains the development-set predictions for
every experimental configuration reported in the results:

| File | Experiment |
|---|---|
| `system_a_seed42.csv` | DistilBERT — Seed 42 |
| `system_a_seed123.csv` | DistilBERT — Seed 123 |
| `system_b_zeroshot.csv` | Qwen — Zero-shot |
| `system_b_6shot.csv` | Qwen — 6-shot |

Each prediction file follows the required format:

```csv
instance_id,label
```

---

## Reproducing the Experiments

The experiments were run in **Google Colab with a T4 GPU**.

To reproduce the results:

1. Open `dico_nli_assignment_CAMILA.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Run **Part 0** to verify the GPU environment and clone the official
   DiCo-NLI repository.
4. Run **Part 1** to load and explore the official Track 1 train and
   development data.
5. Run **Part 2** to fine-tune DistilBERT using seeds 42 and 123 and generate
   the corresponding development predictions.
6. Run the official scorer for both System A prediction files.
7. Run **Part 3** to load Qwen and evaluate the zero-shot and 6-shot prompts.
8. Run the official scorer for both System B prediction files.
9. Run **Part 4** to reproduce the confusion matrices, qualitative error
   analysis, and SoftCons failure analysis.

The notebook saves development predictions using the required
`instance_id,label` format and reads the official scorer's `scores.json`
outputs when constructing the final results table.

---

## AI Assistance

AI coding assistance was used during development to support **debugging, code
organization, and explanations**.

I reviewed and executed the code in Google Colab and verified the final
experimental results using the official DiCo-NLI scorer. The reported results,
prediction files, error analysis, and consistency analysis correspond to the
completed notebook experiments.
