
# Faithful vs Unfaithful Reasoning

## Setup

For this repository to work, please add 

```
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

to the `OFFICIAL_MODEL_NAMES` list in the `loading_from_pretrained.py` file under the `TransformerLens` library (can use `pip show transformer_lens` to find the path).

Also, please install the following dependencies:

```bash
pip install -r requirements.txt
```

## Step 1: Generate COTs

Generates Chain-of-Thought (CoT) explanations for IPHR questions from the chainscope dataset using the `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model.

Assumes this script is located in a directory at the same level as the `chainscope` repository.

```bash
python step1_generate_cots.py
```

This generates a JSON file `results/deepseek_r1_cots_iphr.json` with the CoT explanations.

## Step 2: Evaluate COTs

Evaluates the faithfulness of the CoT explanations generated in Step 1 using OpenAI GPT-4o via the API.

```bash
python step2_evaluate_cots.py
```

This generates a JSON file `results/deepseek_r1_cots_iphr_eval.json` with the evaluation results.

## Step 3: Experiments

Now we can read `results/deepseek_r1_cots_iphr_eval.json` and run mech-interp experiments on them.

`experiment.ipynb` contains some initial boilerplate code to get started.
