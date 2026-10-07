# DiCo-NLI UR2PhD Starter Assignment

This repository contains my work for the UR2PhD Starter Assignment on
SemEval-2027 Task 2: Directional-Consistent Fine-Grained Natural Language
Inference (DiCo-NLI), Track 1 (English).

## Systems

### System A — DistilBERT

I initially attempted to fine-tune `microsoft/deberta-v3-base`, but switched
to `distilbert-base-uncased` because of computational limitations in Google
Colab.

DistilBERT was fine-tuned using:

- Epochs: 4
- Learning rate: 3e-5
- Batch size: 16
- Maximum sequence length: 128
- Precision: FP16
- Random seeds: 42 and 123

### System B — Qwen

I evaluated `Qwen2.5-1.5B-Instruct` without fine-tuning using two prompting
strategies:

- Zero-shot prompting
- 6-shot prompting

The generated responses were converted to the four required DiCo-NLI labels
using a parser.

## Evaluation

All systems were evaluated on the official 660-item Track 1 development set
using the official DiCo-NLI scorer.

The three reported metrics are:

- Weighted F1
- SoftCons
- HardCons

## Repository Structure

- `DiCo_NLI_UR2PhD.ipynb` — Google Colab notebook containing the experiments
- `predictions/` — development-set prediction CSV files for each reported system

## Reproducing the Results

1. Open `DiCo_NLI_UR2PhD.ipynb` in Google Colab.
2. Run the notebook cells in order.
3. The notebook loads the DiCo-NLI data and required models.
4. Run the System A cells to fine-tune DistilBERT with seeds 42 and 123.
5. Run the System B cells to evaluate Qwen with zero-shot and 6-shot prompting.
6. Generate the development-set prediction CSV files.
7. Evaluate the predictions using the official DiCo-NLI scorer.

## AI Assistance

AI coding assistance was used to help with debugging, code organization,
and explanations while developing the experiments. I reviewed and ran the
code in Google Colab and verified the reported results using the official
DiCo-NLI scorer.
