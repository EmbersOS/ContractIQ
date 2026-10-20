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
is fitted only on train contracts. Use `--split test` only once, with
`--confirm-test-run`, after validation decisions are frozen.

Without `--cuad-path`, the script falls back to the synthetic benchmark and marks
`held_out_contracts: false` in the result JSON.
