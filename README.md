# FinanceSLM

A financial narrative-completion model pretrained from scratch. This repository contains the custom Transformer, frozen tokenizer and curated datasets, all 18 recorded experiment results, and a local Gradio interface. It does not call a hosted language model.

The selected checkpoint has **49,737,600 parameters**, validation perplexity **53.22**, and full test perplexity **57.65**. Generated financial claims can be inconsistent or invented. This is a research prototype, not a financial adviser or instruction-tuned chatbot.

**First:** start with [EVALUATE.md](EVALUATE.md) for complete CPU setup commands, expected results and evidence locations. Agent operating instructions are in [AGENTS.md](AGENTS.md). Model weights are attached to [release v1.0.0](https://github.com/HueiSiong/FinanceSLM/releases/tag/v1.0.0).

## Fast start: saved model, no retraining

Use **64-bit Python 3.11**. A CPU is sufficient for setup/evaluation; an NVIDIA GPU is optional for inference. CUDA/BF16 hardware is required for the original accelerated training recipe. The recorded GPU was an RTX 3060 Ti (8GB). Do not copy an old virtual environment from another computer.

After cloning this repository, open a terminal in its root. On Windows PowerShell:

```powershell
git clone https://github.com/HueiSiong/FinanceSLM.git
cd FinanceSLM
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip check
```

For NVIDIA GPU support, use `https://download.pytorch.org/whl/cu118` instead of `/cpu` in the PyTorch installation command. A compatible NVIDIA driver is required; the wheel includes its CUDA runtime. Do not install both wheel variants into the same environment. Official version-specific instructions: https://pytorch.org/get-started/previous-versions/#v231

On Linux, use `python3.11 -m venv .venv` and `.venv/bin/python` instead of the Windows executable path. CPU dependencies are pinned with Windows-only package markers where needed. **Fresh public-download verification passed on Windows and GitHub Actions Ubuntu Linux. See [publication verification](docs/PUBLICATION_VERIFICATION.md) for the tested commit, results and workflow links. macOS is not claimed as tested.**

### Obtain model weights

Weights are GitHub **Release assets**, not ordinary Git files. Download the default checkpoint:

```powershell
.\.venv\Scripts\python.exe artifacts.py fetch --repo HueiSiong/FinanceSLM
.\.venv\Scripts\python.exe research.py smoke --device cpu
.\.venv\Scripts\python.exe app.py --device cpu --open-browser
```

On a GPU installation, use `--device cuda` or omit `--device` for automatic selection. The browser opens at **http://127.0.0.1:7860**. Keep the terminal running; Ctrl+C stops it. `launch_demo.cmd` is a Windows shortcut after `.venv` has been installed.

If the port is already occupied, open the existing demo in your browser or start this one with `python app.py --port 7863 --open-browser`. Use the virtual environment's Python. A missing-module error usually means the wrong interpreter is selected; VS Code and PyCharm should both use this project's `.venv`. Docker Desktop is not required.

If the repository is private, or you downloaded assets manually, place the release ZIPs in a folder and run:

```powershell
.\.venv\Scripts\python.exe artifacts.py install --from-dir C:\path\to\release-assets
```

The default `demo` asset is about 185 MB. Every archive and extracted file is checked against `assets.json`. Downloading this file needs no training. An HTTP 404 means the named release/asset is not published yet or is inaccessible; check the repository name and the `v1.0.0` release. Installation is idempotent and rejects corrupt or mismatched files.

## What needs to be run

1. Install Python 3.11, the chosen PyTorch wheel, and `requirements.lock.txt` as above.
2. Fetch the default release asset.
3. Run `python tests.py` and `python research.py smoke --device cpu` using the virtual environment's Python.
4. Run `python research.py evaluate --device cpu` for the full saved-model test score, or `--device cuda` on a GPU setup.
5. Run `python app.py --open-browser` to use the interface.

Smoke testing verifies input/evidence checksums, checkpoint/tokenizer pairing, tokenizer round trip, repeatable same-device sampling, and numerical logits against a reference. Its 1,024-token perplexity is **partial**, not the published full score. Full evaluation scores **128,685 target tokens** across the fixed test documents; use the actual result's token count as authoritative. It compares the saved asset's score with its recorded score at a 1% relative tolerance. Numerical equality is not promised across every OS/GPU/library combination. A freshly retrained checkpoint is not falsely required to equal the historical checkpoint.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for exact tests and limitations. No `.runtime` or copied `venv` package injection is used; `local_runtime.py` only locates DLLs inside the currently active environment.

All eight local verification suites passed again on 5 November 2026 using fresh PSF-signed Python environments. See [the latest Windows setup verification](docs/SETUP_VERIFICATION.md).

## Same-prompt comparison, no Python needed

Open [the offline HTML explorer](docs/comparison/Same_Prompt_Comparison.html) from the downloaded repository in your browser. GitHub's file preview does not execute HTML: download it or open it locally.

- [Word comparison](docs/comparison/Same_Prompt_Comparison.docx): all 20 prompts, both training seeds, model-size and dataset-size comparisons (80 tables).
- [Plain-text comparison](docs/comparison/Same_Prompt_Comparison.md).
- [Final research report](docs/FinanceSLM_Final_Report.docx).

The explorer contains all **360 saved continuations**. It changes one comparison axis at a time. Model-size comparisons hold dataset and training seed fixed; dataset-size comparisons hold model size and training seed fixed. The 20 starting prompts, tokenizer, context and generation settings are identical across runs. Settings: 80 maximum new tokens, temperature 0.8, top-K 40, generation seed 42, 256-token context. Different trained models are expected to produce different text from the same prompt.

## Optional models and raw reports

```powershell
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group 15m
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group 30m
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group 50m
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group raw
```

Use the virtual environment's Python in these and subsequent commands. Each model-size group contains six validation-selected checkpoints (three datasets × two seeds). Available comparison models appear in the UI after restarting it. The UI catalog selects a seed by validation for each combination; for rigorously matched-seed comparisons use the offline explorer or explicit checkpoint paths. The optional raw asset contains the 164 original HTML reports. `--group all` installs all five groups and requires roughly 2.4 GB of download space plus extracted files.

## Re-evaluate another saved model

```powershell
python research.py evaluate --device cuda --checkpoint checkpoints/final/15m_large_seed42.pt --output work/15m_evaluation.json
```

Full evaluation uses document-bounded blocks of at most 256 target tokens, omits the first token of each document, and scores every remaining token once. Perplexity is `exp(total negative log likelihood / scored tokens)`. Outputs go under `work/`; recorded evidence is never overwritten. Add `--max-tokens 1024` for a labelled partial check, or `--split validation` to score validation data.

## Train from scratch (separate, optional)

The frozen tokenized data is already included. A quick functional check, **not a research result**:

```powershell
python research.py train --device cpu --model 15m --budget small --steps 2 --batch-size 1 --output work/training-smoke
```

One original-size CUDA run:

```powershell
python research.py train --device cuda --model 30m --budget large --seed 42 --output work/30m-large-seed42
```

The default recipe is 3,072 steps × batch 16 × context 256 = 12,582,912 token positions, BF16 autocast, FP32 parameters, fused AdamW, 100-step warmup, cosine decay, peak learning rate 0.0003 and norm clipping 1.0. Validation probes at step 128 and every 256 steps select `best.pt`; `last.pt` includes final optimizer state. Scripts do not automatically resume interrupted runs; use a new output directory. Recorded training time was approximately 2.4/3.6/5.2 minutes per 15M/30M/50M run on the RTX 3060 Ti, excluding separate evaluation. Other hardware can differ substantially.

To repeat the matrix, run each `--model` in `15m,30m,50m`, each `--budget` in `small,medium,large`, and each `--seed` in `42,123`, with a distinct `work/` directory. Fixed seeds do not promise bit-identical GPU retraining. The original final-step optimizer snapshots remain in the owner's workspace; release assets contain selected inference weights, while the provided trainer can create new final-step optimizer snapshots.

## Rebuild the corpus (optional)

```powershell
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group raw
python curate.py --output-dir work/rebuilt-data --audit-dir work/rebuilt-audit
```

This preserves frozen `data/final/`. The script re-extracts the local SEC HTML snapshots, deduplicates paragraphs, rebuilds splits and trains a tokenizer. Original source URLs remain in the metadata. To reproduce the recorded models' inference, use the **frozen tokenizer and token IDs** shipped in `data/final`, not an independently retrained vocabulary. Online SEC pages can change; raw snapshots are checksum-verified. There is no need to redownload from SEC for evaluation.

Both rebuild directories must be separate, empty directories under `work/`. The local Windows rebuild reproduced all 149 frozen dataset files byte-for-byte; this does not guarantee identical tokenization or path serialization across every library/OS version.

## Results and scope

The experiment crossed 15.46M/29.40M/49.74M models with 357,288/706,370/1,048,452 training token IDs and two seeds. Larger datasets reduced mean test perplexity by about 20–23%; increasing model size over the tested range improved it by about 4–7%. The 30M and 50M largest-dataset mean test scores were almost tied. Financial consistency and coherence remain limited. Earlier proposed 20M/60M/120M datasets were not available; repeated token exposure is not new source data. NeMo/Megatron was not used; the custom training loop uses PyTorch.

All recorded results are under `results/final/`. `checksums.json` fingerprints the frozen datasets and evidence. `work/`, checkpoints, raw-release files and virtual environments are ignored by Git. See [NOTICE.md](NOTICE.md) for provenance and licensing scope.
