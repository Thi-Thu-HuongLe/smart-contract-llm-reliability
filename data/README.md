# Benchmark inputs

Place the following analysis-ready files in this directory before running the
pipeline:

| Filename | Role |
|---|---|
| `smartbugs_curated.json` | SmartBugs Curated labelled contracts |
| `scrawld_vulnerabilities.json` | ScrawlD vulnerability records |
| `bccc_secure.json` | BCCC secure records |
| `bccc_vulnerable.json` | BCCC vulnerable records |

The JSON files are excluded from ordinary Git because several exceed GitHub's
single-file limit. Cite and retrieve the original public benchmarks from their
official sources. For exact reproducibility, deposit the analysis-ready files
in a DOI-backed research repository and publish their SHA-256 checksums.

`SOURCE_LINKS.txt` records the source locations used during data preparation.
`DATA_SHA256SUMS.txt` identifies the exact local analysis-ready files.
