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

## Reference

This project is motivated by the following paper:

> O Shulayeva, A Siddharthan, A Wyner (2017). *Recognizing cited facts and principles in legal judgements*. Artificial Intelligence and Law. https://link.springer.com/article/10.1007/s10506-017-9197-6

Paper author: Prof. Advaith Siddharthan, University of Aberdeen (advaith@abdn.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
