# Recovery baselines

This directory compares the existing out-of-fold TACE-F1 decisions with a no-model prevalence-only rule and a conventional class-wise logistic stacker. All systems use the same outer record partitions. Hyperparameters and thresholds for the new baselines are selected only within nested inner folds. No LLM inference is performed.
