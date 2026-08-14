# Benchmark inputs

Place these analysis-ready files in this directory before running inference or
evaluation:

| Filename | Role |
|---|---|
| `smartbugs_curated.json` | SmartBugs Curated labelled contracts |
| `scrawld_vulnerabilities.json` | ScrawlD vulnerability records |
| `bccc_secure.json` | BCCC secure records |
| `bccc_vulnerable.json` | BCCC vulnerable records |

The JSON files are excluded from Git because several exceed GitHub's
single-file limit. Retrieve the public source datasets from their official
repositories. Use the exact analysis-ready files from the associated research
archive when reproducing the reported results.
