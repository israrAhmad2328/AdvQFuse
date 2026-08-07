$ErrorActionPreference = "Stop"

Write-Host "[1/4] Creating environment"
if (-not (Test-Path ".venv")) {
    py -3.10 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

Write-Host "[2/4] Installing AdvQFuse-RS"
python -m pip install --upgrade pip
pip install -e ".[dev,plots]"

Write-Host "[3/4] Running tests"
pytest

Write-Host "[4/4] Generating advanced synthetic visualization checks"
python scripts\generate_advrs_demo_results.py `
  --results-dir results\advanced_demo `
  --figures-dir figures\advanced_demo

Write-Host "Done. Synthetic figures are in figures\advanced_demo."
Write-Host "They are watermarked and must be replaced by logged Bonsai experiments."
