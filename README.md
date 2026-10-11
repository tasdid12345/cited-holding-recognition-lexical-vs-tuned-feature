# Matching cited legal holdings to citing passages: TF-IDF baseline vs tuned LightGBM

## Motivation
The paper "Recognizing cited facts and principles in legal judgements" (2017) studies how to recognise which facts and principles a legal judgement cites, largely through n-gram similarity between texts. This experiment asks whether a modern gradient-boosting model on richer lexical-similarity features does better than a plain n-gram overlap matcher on a related task.

## Question
Can a tuned gradient-boosting model on richer lexical-similarity features match the correct cited legal holding to a citing passage more accurately than a standard n-gram overlap baseline, including when wording overlap is low?

## Dataset
CaseHOLD from LexGLUE (`coastalcph/lex_glue`, `case_hold` config), downloaded as parquet from the Hugging Face hub. Each US court passage has one correct holding among 5 candidates. The official train and test splits are used, subsampled to about 6,000 training and 2,000 test passages. This is not the paper's own dataset, and the baseline is a stand-in rather than the paper's method.

## Models
- **Baseline:** word 1-2-gram TF-IDF cosine between passage and candidate, then logistic regression on that single score; the top-scoring candidate is chosen.
- **Improved:** LightGBM classifier on word and character n-gram cosines, BM25, rare-term overlap, length features and each candidate's score relative to the other four. Hyper-parameters are tuned by cross-validation on the training set only.

Both share the same data, preprocessing and evaluation code. The metric is 5-way accuracy, with secondary accuracy on the lowest-overlap third of the test set and macro-F1 on binary correct/incorrect labels.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`) when the script runs.

## Results (latest run)

On CaseHOLD (LexGLUE) with 2,000 test rows, a tuned LightGBM model on richer lexical-similarity features reached 61.7% 5-way accuracy versus 56.4% for a standard n-gram TF-IDF similarity matcher (61.0% vs 52.4% on low-overlap cases), a modest single-subsample gain.

| Metric | Value |
|---|---|
| baseline_score | 0.564 |
| baseline_score_std | 0 |
| baseline_f1 | 0.7275 |
| baseline_f1_std | 0 |
| baseline_low_overlap_acc | 0.5238 |
| improved_score | 0.6167 |
| improved_score_std | 0.0006236 |
| improved_f1 | 0.7604 |
| improved_f1_std | 0.0003898 |
| improved_low_overlap_acc | 0.6099 |
| improvement | 0.05267 |
| improvement_low_overlap | 0.0861 |
| chance_accuracy | 0.2 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Data source repo: coastalcph/lex_glue. Train subsampled from the first parquet shard of the train split. Seeds [42, 43, 44]; baseline is deterministic so its std is ~0. Low-overlap subset = test third with lowest max candidate-token overlap with the passage (label-free). Macro-F1 is over binary correct/incorrect candidate rows, predicting 1 for the chosen candidate. Best baseline params {'C': 0.1}; best LightGBM params {'num_leaves': 7, 'n_estimators': 150, 'min_child_samples': 50, 'learning_rate': 0.1}. Single subsample, so differences of ~1-2 points are within sampling noise (n_test=2000). Runtime 64s.

## Reference

This project is motivated by the following paper:

> O Shulayeva, A Siddharthan, A Wyner (2017). *Recognizing cited facts and principles in legal judgements*. Artificial Intelligence and Law. https://link.springer.com/article/10.1007/s10506-017-9197-6

Paper author: Prof. Advaith Siddharthan, University of Aberdeen (advaith@abdn.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
