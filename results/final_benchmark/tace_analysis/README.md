# TACE cross-fitted ensemble results

These results implement a post-inference, taxonomy-aware empirical-Bayes ensemble over the three frozen model outputs. All TACE quantities are out-of-fold. TACE-F1 and TACE-Macro are distinct nested-selected operating points, not results tuned on the outer test folds. See `tace_manifest.json` for the exact inputs, hashes, and protocol.

Do not describe this as a newly trained detector or as a new LLM inference run. It is a reproducible decision layer on fixed model outputs.
