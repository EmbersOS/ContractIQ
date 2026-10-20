# CUAD v1 evaluation data

ContractIQ evaluates on the official **CUAD v1** release from The Atticus Project.

- Source: Zenodo record [4595826](https://zenodo.org/records/4595826) (`CUAD_v1.zip`, 105.9 MB)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Archive MD5: `c38f490a984420b8a62600db401fafd5`

Raw data and generated metrics are intentionally ignored by Git. The expected local
file is `backend/data/cuad/CUAD_v1.json`.

## Setup

From the repository root (PowerShell):

```powershell
New-Item -ItemType Directory -Force backend/data/cuad | Out-Null
Invoke-WebRequest "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1" -OutFile backend/data/cuad/CUAD_v1.zip
Expand-Archive backend/data/cuad/CUAD_v1.zip -DestinationPath backend/data/cuad
Copy-Item backend/data/cuad/CUAD_v1/CUAD_v1/CUAD_v1.json backend/data/cuad/CUAD_v1.json
Get-FileHash -Algorithm MD5 backend/data/cuad/CUAD_v1.zip
Remove-Item backend/data/cuad/CUAD_v1.zip
```

On macOS/Linux, use `curl -L`, `unzip`, and `cp` for the same steps.

## Run held-out evaluation

From `backend/`:

```bash
python -m ml.evaluate --model all --split val --cuad-path data/cuad/CUAD_v1.json
```

The split is deterministic and contract-level (seed 42, 70/15/15). The classifier
is fitted only on train contracts. P1 classifies regex-anchored candidate passages,
and predicted offsets remain relative to the original contract text. Use
`--split test` only once, with `--confirm-test-run`, after validation decisions are frozen.

Without `--cuad-path`, the script falls back to the synthetic benchmark and marks
`held_out_contracts: false` in the result JSON.

## Final P1 pipeline

1. Regex clause candidates are generated from the raw contract text.
2. Each candidate is expanded to a sentence-like passage (minimum 120 characters,
   maximum 1,000), and nested passages are suppressed by containment.
3. Passages are cleaned with `clean_contract_text`, then classified by
   TF-IDF (`ngram_range=(1, 2)`, `max_features=5000`, `sublinear_tf=True`) plus
   logistic regression (`C=1.0`, `max_iter=1000`, `random_state=42`).
4. The classifier is fitted only on the 357 train contracts (1,652 mapped
   examples). Predicted offsets always refer to the original contract text.

## Metric scopes

- `classification`: sample-level classification. For each gold sample, the
  highest-span-IoU prediction is selected, so unmatched extra predictions do not
  affect this score.
- `span_metrics`: localization quality (exact match, token F1, span IoU) of those
  same sample-selected predictions.
- `span_detection`: end-to-end prediction-level precision/recall/F1. Matching is
  one-to-one, requires the same clause type, and requires span IoU >= 0.50; pairs
  are matched highest-IoU first (ties by lower gold then prediction index). Every
  unmatched prediction, including duplicates and wrong-type candidates, is a false
  positive.
- `candidate_coverage`: separate type-agnostic extraction recall. A gold span is
  covered when any predicted span reaches IoU >= 0.50, even if classified as the
  wrong type. `candidate_overlap_recall` is the same measure for any character
  overlap, making missed clauses and span-extraction gaps visible.
