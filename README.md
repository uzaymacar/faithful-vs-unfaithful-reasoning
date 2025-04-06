
# Faithful vs Unfaithful Reasoning

Trying to distinguish between faithful and unfaithful chain-of-thought (CoT) explanations.

You can find a few slides about this mini-project [here](https://docs.google.com/presentation/d/1whRlgP9aCDrzuEcz0VEfxYhyTEP3APp8bj3zot-h27U/edit?usp=sharing).

## Setup

1. Make sure this repository is located in a directory at the same level as the [`chainscope`](https://github.com/jettjaniak/chainscope/tree/main) repository.

2. Add the following model to the `OFFICIAL_MODEL_NAMES` list in the `loading_from_pretrained.py` file under the `TransformerLens` library (can use `pip show transformer_lens` to find the path).

```python
'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
```

3. Also, please install the following dependencies:

```bash
pip install -r requirements.txt
```

## Experiments

### Training Classifiers

Run the `classifiers.py` with the `-a` argument as either `resid_post` or `tokens` to train the classifier respectively on the residual stream activations or tokens.

```bash
python classifiers.py -m qwen2.5-1.5b-instruct -a resid_post
```

### Token Analysis

Run the `token_analysis.py` script to analyze the tokens in the CoT explanations.

```bash
python token_analysis.py -m qwen2.5-1.5b-instruct
```

### Attention Analysis

Run the `attention_analysis.py` script to analyze the attention patterns of the classifier on the tokens in the CoT explanations.

```bash
python attention_analysis.py -m qwen2.5-1.5b-instruct -p saved_models/classifier_tokens/qwen2.5-1.5b-instruct_0_5_10_15_20_25.pt
```

## Archive

Instead of generating our own CoT data, we used the CoTs provided in the [`chainscope`](https://github.com/jettjaniak/chainscope/tree/main) repository.

The below scripts are therefore archived and can be found in the `archive` directory.

### Step 1: Generate COTs

Generates Chain-of-Thought (CoT) explanations for IPHR questions from the chainscope dataset using the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model.

Assumes this repository is located in a directory at the same level as the `chainscope` repository.

```bash
python archive/step1_generate_cots.py
```

This generates a JSON file `results/deepseek_r1_cots_iphr.json` with the CoT explanations.

### Step 2: Evaluate COTs

Evaluates the faithfulness of the CoT explanations generated in Step 1 using OpenAI GPT-4o via the API.

```bash
python archive/step2_evaluate_cots.py
```

This generates a JSON file `results/deepseek_r1_cots_iphr_eval.json` with the evaluation results.

## Step 3: Experiments

Now we can read `results/deepseek_r1_cots_iphr_eval.json` and run mech-interp experiments on them.

`experiment.ipynb` contains some initial boilerplate code to get started.
