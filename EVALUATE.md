# Evaluate FinanceSLM

Start here for setup and evaluation. Read AGENTS.md for operating instructions and README.md for the experiment, optional training and full data rebuild. The public repository and release assets require no GitHub account or hosted-model API key.

## Windows PowerShell: saved-model verification

Install Git and 64-bit Python 3.11 first. Use an official Python Software Foundation distribution; see docs/SETUP_VERIFICATION.md if Windows blocks an unsigned venv launcher. Start in a directory where a new FinanceSLM folder can be created:

```powershell
git clone https://github.com/HueiSiong/FinanceSLM.git
cd FinanceSLM
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe artifacts.py fetch --repo HueiSiong/FinanceSLM
.\.venv\Scripts\python.exe tests.py
.\.venv\Scripts\python.exe research.py smoke --device cpu
.\.venv\Scripts\python.exe research.py evaluate --device cpu
```

## Linux: saved-model verification

Install Git, Python 3.11 and its venv support using the system's normal installation process. Then:

```bash
git clone https://github.com/HueiSiong/FinanceSLM.git
cd FinanceSLM
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python artifacts.py fetch --repo HueiSiong/FinanceSLM
.venv/bin/python tests.py
.venv/bin/python research.py smoke --device cpu
.venv/bin/python research.py evaluate --device cpu
```

The default weight download is approximately 185 MB. All checksums are verified before installation. Other groups are optional; the default verification requires no GPU, Docker or new training. Package downloads require internet access. GitHub CI results are available in the repository's Actions tab; do not assume an untested platform passes.

## Expected results and interpretation

- tests.py: five regression tests pass.
- research.py smoke: `passed: true`, 49,737,600 parameters, verified tokenizer/data fingerprints and numerical reference checks.
- research.py evaluate: `partial: false`, **128,685 target tokens**, full test perplexity close to **57.6451578**. The historical-checkpoint check allows 1% relative numerical tolerance.
- Outputs are written to `work/smoke.json` and `work/evaluation.json`. Preserve these with your evaluation notes.

The smoke perplexity uses only 1,024 tokens and is not comparable to the full test score. CPU and GPU sampled words need not match. The model completes financial prose; it is not instruction-tuned, and generated amounts are unverified.

To inspect the interface, run `.\.venv\Scripts\python.exe app.py --device cpu --open-browser` on Windows, or `.venv/bin/python app.py --device cpu` on Linux. It serves http://127.0.0.1:7860 while the process is running. In a headless agent environment, complete the command-line evaluation first; browser access is optional.

## Run the live comparison interface

For three fresh continuations with full-test perplexity beneath each, follow [docs/LIVE_COMPARISON.md](docs/LIVE_COMPARISON.md). Install release groups `15m`, `30m` and `50m`, then run `python compare_app.py --open-browser` using your virtual environment's Python. It serves port 7863 and saves logs under `work/comparison/`. This requires no training. The default `demo` download alone supplies the single-model interface, not these matched-seed comparisons.

## Inspect the evidence

| Material | Location |
| --- | --- |
| Original aim and final Word report | `docs/Project_Aim.docx`, `docs/FinanceSLM_Final_Report.docx` |
| Same-prompt Word comparison | `docs/comparison/Same_Prompt_Comparison.docx` |
| Offline explorer: 360 saved continuations | `docs/comparison/Same_Prompt_Comparison.html` |
| All 18 recorded runs, histories and evaluations | `results/final/runs/` |
| Frozen cleaned prose, company/year metadata and tokenized datasets | `data/final/` |
| Original source URLs | `data/historical_metadata.csv`, `data/expanded_metadata.csv` |
| Setup protocol and historical verification logs | `docs/REPRODUCIBILITY.md`, `docs/verification/` |
| Latest Windows environment verification summary | `docs/SETUP_VERIFICATION.md` |

For all experimental checkpoint groups and the 164 raw SEC HTML snapshots, use `python artifacts.py fetch --repo HueiSiong/FinanceSLM --group all` with the environment's Python. This downloads roughly 2.4 GB. README.md explains individual groups, per-model evaluation, optional retraining and rebuilding the corpus. The original sources are HTML filings, not PDF files.

## Suggested instruction for an evaluating agent

> Clone https://github.com/HueiSiong/FinanceSLM and follow EVALUATE.md and AGENTS.md. Create a fresh Python 3.11 environment, install the pinned CPU dependencies, fetch and verify the default release checkpoint, run the regression tests, smoke test and full evaluation. Report the actual full-test perplexity and target-token count, inspect the saved same-prompt comparisons and research limitations, and preserve your logs under work/. Do not start training unless I explicitly request a replication experiment.
