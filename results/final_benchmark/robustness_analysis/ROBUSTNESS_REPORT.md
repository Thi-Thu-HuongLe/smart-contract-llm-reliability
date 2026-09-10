# Prediction robustness analysis

Generated at 2026-09-10T01:31:36.380125+00:00.

## Overall sensitivity decision

**Passed: False**

Pre-specified rule: Ranking unchanged across strict first-attempt, all-recovery-as-empty, and common-native-valid analyses; for the two full-record policies, max |primary metric delta| must not exceed the equivalence margin.

## Selective-repair exposure

Action counts are chunk-level; touched rates are record-level.

| Dataset | Model | Touched records | Rate | Inference chunks | Structural chunks |
|---|---|---|---|---|---|
| SmartBugs-Curated | Qwen2.5-Coder-7B-Instruct | 0/143 | 0.00% | 0 | 0 |
| SmartBugs-Curated | CodeLlama-7B-Instruct | 11/143 | 7.69% | 11 | 0 |
| SmartBugs-Curated | Mistral-7B-Instruct-v0.3 | 0/143 | 0.00% | 0 | 0 |
| ScrawlD | Qwen2.5-Coder-7B-Instruct | 234/5664 | 4.13% | 267 | 0 |
| ScrawlD | CodeLlama-7B-Instruct | 3/5664 | 0.05% | 3 | 0 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | 0/5664 | 0.00% | 0 | 0 |
| BCCC | Qwen2.5-Coder-7B-Instruct | 1687/19512 | 8.65% | 886 | 995 |
| BCCC | CodeLlama-7B-Instruct | 30/19512 | 0.15% | 32 | 0 |
| BCCC | Mistral-7B-Instruct-v0.3 | 0/19512 | 0.00% | 0 | 0 |

## Sensitivity metrics

`first_attempt_strict_all` reconstructs only schema-valid first responses. `all_recovery_as_empty` treats every recovery-exposed record as having no accepted finding. `common_native_valid` evaluates the identical subset that was valid on the first response for all three models. The two legacy policies are retained only to show how the earlier, narrower selective-repair definition differs.

| Dataset | Model | Policy | n | Primary metric | Value | Delta vs final |
|---|---|---|---|---|---|---|
| SmartBugs-Curated | Qwen2.5-Coder-7B-Instruct | final_all | 143 | micro_f1 | 0.9606 | +0.0000 |
| SmartBugs-Curated | Qwen2.5-Coder-7B-Instruct | first_attempt_strict_all | 143 | micro_f1 | 0.9015 | -0.0591 |
| SmartBugs-Curated | Qwen2.5-Coder-7B-Instruct | all_recovery_as_empty | 143 | micro_f1 | 0.8931 | -0.0674 |
| SmartBugs-Curated | Qwen2.5-Coder-7B-Instruct | common_native_valid | 39 | micro_f1 | 0.9610 | +0.0005 |
| SmartBugs-Curated | Qwen2.5-Coder-7B-Instruct | selective_repair_as_empty_legacy | 143 | micro_f1 | 0.9606 | +0.0000 |
| SmartBugs-Curated | Qwen2.5-Coder-7B-Instruct | common_selective_untouched_legacy | 132 | micro_f1 | 0.9732 | +0.0126 |
| SmartBugs-Curated | CodeLlama-7B-Instruct | final_all | 143 | micro_f1 | 0.6701 | +0.0000 |
| SmartBugs-Curated | CodeLlama-7B-Instruct | first_attempt_strict_all | 143 | micro_f1 | 0.3583 | -0.3117 |
| SmartBugs-Curated | CodeLlama-7B-Instruct | all_recovery_as_empty | 143 | micro_f1 | 0.3590 | -0.3111 |
| SmartBugs-Curated | CodeLlama-7B-Instruct | common_native_valid | 39 | micro_f1 | 0.6542 | -0.0158 |
| SmartBugs-Curated | CodeLlama-7B-Instruct | selective_repair_as_empty_legacy | 143 | micro_f1 | 0.7006 | +0.0305 |
| SmartBugs-Curated | CodeLlama-7B-Instruct | common_selective_untouched_legacy | 132 | micro_f1 | 0.7230 | +0.0530 |
| SmartBugs-Curated | Mistral-7B-Instruct-v0.3 | final_all | 143 | micro_f1 | 0.9122 | +0.0000 |
| SmartBugs-Curated | Mistral-7B-Instruct-v0.3 | first_attempt_strict_all | 143 | micro_f1 | 0.8760 | -0.0362 |
| SmartBugs-Curated | Mistral-7B-Instruct-v0.3 | all_recovery_as_empty | 143 | micro_f1 | 0.8760 | -0.0362 |
| SmartBugs-Curated | Mistral-7B-Instruct-v0.3 | common_native_valid | 39 | micro_f1 | 1.0000 | +0.0878 |
| SmartBugs-Curated | Mistral-7B-Instruct-v0.3 | selective_repair_as_empty_legacy | 143 | micro_f1 | 0.9122 | +0.0000 |
| SmartBugs-Curated | Mistral-7B-Instruct-v0.3 | common_selective_untouched_legacy | 132 | micro_f1 | 0.9304 | +0.0182 |
| ScrawlD | Qwen2.5-Coder-7B-Instruct | final_all | 5664 | micro_f1 | 0.0123 | +0.0000 |
| ScrawlD | Qwen2.5-Coder-7B-Instruct | first_attempt_strict_all | 5664 | micro_f1 | 0.0005 | -0.0117 |
| ScrawlD | Qwen2.5-Coder-7B-Instruct | all_recovery_as_empty | 5664 | micro_f1 | 0.0005 | -0.0117 |
| ScrawlD | Qwen2.5-Coder-7B-Instruct | common_native_valid | 42 | micro_f1 | 0.0000 | -0.0123 |
| ScrawlD | Qwen2.5-Coder-7B-Instruct | selective_repair_as_empty_legacy | 5664 | micro_f1 | 0.0110 | -0.0012 |
| ScrawlD | Qwen2.5-Coder-7B-Instruct | common_selective_untouched_legacy | 5427 | micro_f1 | 0.0116 | -0.0007 |
| ScrawlD | CodeLlama-7B-Instruct | final_all | 5664 | micro_f1 | 0.1083 | +0.0000 |
| ScrawlD | CodeLlama-7B-Instruct | first_attempt_strict_all | 5664 | micro_f1 | 0.1004 | -0.0079 |
| ScrawlD | CodeLlama-7B-Instruct | all_recovery_as_empty | 5664 | micro_f1 | 0.0449 | -0.0634 |
| ScrawlD | CodeLlama-7B-Instruct | common_native_valid | 42 | micro_f1 | 0.0364 | -0.0719 |
| ScrawlD | CodeLlama-7B-Instruct | selective_repair_as_empty_legacy | 5664 | micro_f1 | 0.1081 | -0.0001 |
| ScrawlD | CodeLlama-7B-Instruct | common_selective_untouched_legacy | 5427 | micro_f1 | 0.1018 | -0.0065 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | final_all | 5664 | micro_f1 | 0.0792 | +0.0000 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | first_attempt_strict_all | 5664 | micro_f1 | 0.0036 | -0.0756 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | all_recovery_as_empty | 5664 | micro_f1 | 0.0021 | -0.0771 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | common_native_valid | 42 | micro_f1 | 0.0222 | -0.0570 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | selective_repair_as_empty_legacy | 5664 | micro_f1 | 0.0792 | +0.0000 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | common_selective_untouched_legacy | 5427 | micro_f1 | 0.0745 | -0.0047 |
| BCCC | Qwen2.5-Coder-7B-Instruct | final_all | 19512 | balanced_accuracy | 0.4999 | +0.0000 |
| BCCC | Qwen2.5-Coder-7B-Instruct | first_attempt_strict_all | 19512 | balanced_accuracy | 0.4999 | +0.0000 |
| BCCC | Qwen2.5-Coder-7B-Instruct | all_recovery_as_empty | 19512 | balanced_accuracy | 0.4999 | +0.0000 |
| BCCC | Qwen2.5-Coder-7B-Instruct | common_native_valid | 120 | balanced_accuracy | 0.5000 | +0.0001 |
| BCCC | Qwen2.5-Coder-7B-Instruct | selective_repair_as_empty_legacy | 19512 | balanced_accuracy | 0.4999 | +0.0000 |
| BCCC | Qwen2.5-Coder-7B-Instruct | common_selective_untouched_legacy | 17805 | balanced_accuracy | 0.4999 | -0.0000 |
| BCCC | CodeLlama-7B-Instruct | final_all | 19512 | balanced_accuracy | 0.5018 | +0.0000 |
| BCCC | CodeLlama-7B-Instruct | first_attempt_strict_all | 19512 | balanced_accuracy | 0.5459 | +0.0441 |
| BCCC | CodeLlama-7B-Instruct | all_recovery_as_empty | 19512 | balanced_accuracy | 0.4891 | -0.0127 |
| BCCC | CodeLlama-7B-Instruct | common_native_valid | 120 | balanced_accuracy | 0.6289 | +0.1271 |
| BCCC | CodeLlama-7B-Instruct | selective_repair_as_empty_legacy | 19512 | balanced_accuracy | 0.5016 | -0.0002 |
| BCCC | CodeLlama-7B-Instruct | common_selective_untouched_legacy | 17805 | balanced_accuracy | 0.5019 | +0.0001 |
| BCCC | Mistral-7B-Instruct-v0.3 | final_all | 19512 | balanced_accuracy | 0.5202 | +0.0000 |
| BCCC | Mistral-7B-Instruct-v0.3 | first_attempt_strict_all | 19512 | balanced_accuracy | 0.5012 | -0.0190 |
| BCCC | Mistral-7B-Instruct-v0.3 | all_recovery_as_empty | 19512 | balanced_accuracy | 0.4936 | -0.0266 |
| BCCC | Mistral-7B-Instruct-v0.3 | common_native_valid | 120 | balanced_accuracy | 0.6644 | +0.1443 |
| BCCC | Mistral-7B-Instruct-v0.3 | selective_repair_as_empty_legacy | 19512 | balanced_accuracy | 0.5202 | +0.0000 |
| BCCC | Mistral-7B-Instruct-v0.3 | common_selective_untouched_legacy | 17805 | balanced_accuracy | 0.5194 | -0.0008 |

## Paired model comparisons

Confidence intervals use paired record bootstrap. P-values use paired random-swap permutation tests and are Holm-adjusted within each dataset/metric family.

| Dataset | Metric | A | B | Delta A-B | 95% paired CI | Holm p | Significant |
|---|---|---|---|---|---|---|---|
| SmartBugs-Curated | micro_f1 | Qwen2.5-Coder-7B-Instruct | CodeLlama-7B-Instruct | +0.2905 | [+0.2176, +0.3576] | 0.000300 | yes |
| SmartBugs-Curated | micro_f1 | Qwen2.5-Coder-7B-Instruct | Mistral-7B-Instruct-v0.3 | +0.0484 | [+0.0078, +0.0925] | 0.036496 | yes |
| SmartBugs-Curated | micro_f1 | CodeLlama-7B-Instruct | Mistral-7B-Instruct-v0.3 | -0.2421 | [-0.3112, -0.1687] | 0.000300 | yes |
| ScrawlD | micro_f1 | Qwen2.5-Coder-7B-Instruct | CodeLlama-7B-Instruct | -0.0960 | [-0.1027, -0.0892] | 0.000300 | yes |
| ScrawlD | micro_f1 | Qwen2.5-Coder-7B-Instruct | Mistral-7B-Instruct-v0.3 | -0.0669 | [-0.0743, -0.0597] | 0.000300 | yes |
| ScrawlD | micro_f1 | CodeLlama-7B-Instruct | Mistral-7B-Instruct-v0.3 | +0.0291 | [+0.0211, +0.0370] | 0.000300 | yes |
| BCCC | balanced_accuracy | Qwen2.5-Coder-7B-Instruct | CodeLlama-7B-Instruct | -0.0019 | [-0.0053, +0.0014] | 0.789521 | no |
| BCCC | mcc | Qwen2.5-Coder-7B-Instruct | CodeLlama-7B-Instruct | -0.0302 | [-0.0486, -0.0116] | 0.029497 | yes |
| BCCC | balanced_accuracy | Qwen2.5-Coder-7B-Instruct | Mistral-7B-Instruct-v0.3 | -0.0203 | [-0.0260, -0.0144] | 0.002800 | yes |
| BCCC | mcc | Qwen2.5-Coder-7B-Instruct | Mistral-7B-Instruct-v0.3 | -0.0571 | [-0.0760, -0.0376] | 0.000300 | yes |
| BCCC | balanced_accuracy | CodeLlama-7B-Instruct | Mistral-7B-Instruct-v0.3 | -0.0184 | [-0.0234, -0.0135] | 0.000300 | yes |
| BCCC | mcc | CodeLlama-7B-Instruct | Mistral-7B-Instruct-v0.3 | -0.0269 | [-0.0450, -0.0091] | 0.018598 | yes |

## Zero-support correction

Recall and F1 are reported as undefined (`N/A`) for classes with zero ground-truth positives. Supported-class macro metrics exclude those classes; false positives remain visible in the per-class CSV.

| Dataset | System | Zero-support classes | Legacy macro-F1 | Supported macro-F1 |
|---|---|---|---|---|
| ScrawlD | Qwen2.5-Coder-7B-Instruct | ["denial_of_service"] | 0.0382 | 0.0436 |
| ScrawlD | mythril | ["denial_of_service"] | 0.2210 | 0.2526 |
| ScrawlD | osiris | ["denial_of_service"] | 0.1063 | 0.1215 |
| ScrawlD | oyente | ["denial_of_service"] | 0.2860 | 0.3268 |
| ScrawlD | slither | ["denial_of_service"] | 0.2010 | 0.2297 |
| ScrawlD | smartcheck | ["denial_of_service"] | 0.1461 | 0.1669 |
| ScrawlD | CodeLlama-7B-Instruct | ["denial_of_service"] | 0.0942 | 0.1076 |
| ScrawlD | Mistral-7B-Instruct-v0.3 | ["denial_of_service"] | 0.0659 | 0.0753 |

## Interpretation rule

- If sensitivity passes, the principal ranking is robust to the selective-repair policy under the declared 0.01 margin.
- A paired difference is supported when its CI excludes zero and the Holm-adjusted p-value is below 0.05.
- If sensitivity fails, describe the experiment as a selectively repaired evaluation or perform a homogeneous full re-execution.
