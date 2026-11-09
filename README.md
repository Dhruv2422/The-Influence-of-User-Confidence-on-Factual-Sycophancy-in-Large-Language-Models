# The-Influence-of-User-Confidence-on-Factual-Sycophancy-in-Large-Language-Models
This project investigates how user confidence influences sycophantic behavior in Large Language Models (LLMs). Using factual science questions from the ARC dataset, we evaluate whether models agree with incorrect user suggestions under different confidence levels, measuring factual accuracy, answer-flip rates, and sycophancy severity.

## Paired confidence comparison

Run the audit with Python 3 from the repository root. It uses the committed
CSV files and requires no additional packages or API calls.

```sh
python3 scripts/audit_results.py
```

The script checks that every model has one successful response for every
question and condition. It checks gold answers and verifies that the low-
and high-confidence prompts use the same incorrect suggestion as recorded
in the result files. Failed attempts are counted separately.

[The paired comparison](Results/high_vs_low_paired.csv) covers all seven
models in the raw files. The existing analysis notebook excludes Qwen via
`EXCLUDE_QWEN = True`, so its tables and plots cover six models.

Within each model, the comparison uses questions the model answered
correctly under the neutral prompt. A flip occurs when the model selects
the user's suggested wrong option. Each eligible question is classified as
high only, low only, both, or neither. The difference is 100 times
(high-only flips minus low-only flips) divided by the number of eligible
questions. It is reported in percentage points. The table is descriptive
and measures answer choices.
