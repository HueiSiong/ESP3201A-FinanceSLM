# Local reproducibility verification

Verified on **31 October 2026**, before publication. No GitHub repository, push or release was created by this work. The package is ready for the owner to publish using the separate `publication/PUBLISHING_GUIDE.md` in the original workspace.

## Test environment and isolation

Windows x64, Python 3.11.16, NVIDIA GeForce RTX 3060 Ti (8 GB), NVIDIA driver 616.56. Two newly created virtual environments installed dependencies from package indexes: PyTorch 2.3.1+cpu and 2.3.1+cu118 respectively, plus the committed requirements.lock.txt. Both passed pip check.

Tests used an extracted source archive and an actual Git clone of a disposable local repository. Git's Windows line-ending conversion was enabled in the test origin. All frozen fingerprints survived the commit/clone. The publishable source folder itself has no .git directory or remote. The disposable fixture used an explicitly synthetic author identity and was never pushed.

Both environments use the existing Python installation's standard library as their base, as normal virtual environments do. Application packages were installed into the new environments; no packages were injected from the old research venv or .runtime directories. This is a fresh-environment test on the same physical computer, not a second-machine or clean-OS certification.

## Results

| Check | Outcome |
| --- | --- |
| Fresh CPU and CUDA dependency installations | Passed, including pip check |
| Source archive extraction and real local Git clone | Passed |
| Model download via a local HTTP release server | Passed with archive/member SHA-256 verification |
| All five release groups installed | Passed; all selected weights and 164 raw reports verified |
| Repeated asset installation, corrupt ZIP rejection and path traversal rejection | Passed |
| 222 frozen data/evidence fingerprints | Passed |
| Default model CPU full test | PPL **57.6451208370**; 128,685 targets |
| Default model CUDA full test | PPL **57.6451578049**; 128,685 targets |
| Every released experimental model, full CUDA test | All **18** matched their recorded perplexities exactly in this environment |
| First saved prompt regenerated on CUDA for every run | **18/18** exact continuation matches |
| Full 15M / large data / seed 42 retraining | 3,072 steps, 12,582,912 token positions; completed in 150.3 seconds |
| Fresh two-step CPU training and new-checkpoint evaluation | Passed; functional check only |
| Raw-source extraction, cleaning, splits, tokenizer and token arrays | All **149 dataset files byte-identical** to frozen files |
| Word comparison contents | 80 tables; all 240 displayed continuations verified verbatim |
| Offline HTML comparison | 240 selector combinations / 720 displayed outputs checked; mobile layout passed |
| Gradio in a fresh CPU environment | Browser generation, repeatability, blank prompt validation and 15M/30M/50M switching passed |
| Original workspace preservation | All 342 original file fingerprints and 15 final training-input fingerprints unchanged |

CPU full-test relative difference from the recorded default-model result was **0.00006413%**. GPU relative difference was **0%**. Direct reference-logit checks also passed. The script allows 1% relative perplexity tolerance for a hash-verified historical checkpoint to accommodate other compatible environments; this is a verification threshold, not an accuracy claim.

The retrained 15M model selected step 1024 and achieved full-test PPL **64.37976534**, compared with the historical run's **64.40190417** (relative difference **0.0344%**). This demonstrates the complete training/evaluation path while showing why fixed seeds do not guarantee bit-identical retraining. These new verification weights and metrics do not replace any of the original 18 research runs.

## What changed for portability

- Replaced legacy environment path injection with a helper that only finds DLLs inside the active environment.
- Made the published app default to the selected final model, with repository-relative data paths, model asset checks, available-model filtering and output under work/.
- Separated Git source files from large release archives. assets.json fingerprints archives and every extracted member. The downloader supports a GitHub release or local archives and rejects unsafe destinations.
- Added portable evaluation and optional training commands, a full dependency lock, agent instructions and optional Linux CI.
- Protected frozen file bytes from Git line-ending conversion. Training/rebuilding writes only under work/ and refuses nonempty output directories.
- Removed the obsolete data/corpus dependency from the optional raw-data rebuild. That dependency was discovered by the clean clone test; the corrected rebuild reproduced the entire frozen dataset.
- Added exact saved-output Word, Markdown and self-contained HTML comparisons. No outputs were rewritten or cherry-picked.

Browser tests initially needed a locator correction because the disabled Gradio output uses its placeholder as its accessible name. The UI generated successfully; the corrected browser test passed. Earlier failed attempts are not represented as successful checks.

## Reproduce the checks

Follow README.md to create a fresh environment and fetch/install assets, then run:

```text
python -m pip check
python tests.py
python research.py smoke --device cpu
python research.py evaluate --device cpu
python app.py --device cpu --open-browser
```

Use the virtual environment's Python. With the CUDA wheel and a compatible NVIDIA GPU, substitute --device cuda. Optional training and raw-data rebuild commands are documented in README.md. With all model groups installed, evaluate each checkpoint explicitly; the model's embedded run identifier selects the correct recorded score for comparison.

The smoke test scores only 1,024 target tokens. Compare the published full-test result only with the complete 128,685-target evaluation. Do not compare training loss, validation-prefix loss, partial perplexity and full-test perplexity as if they were the same metric.

## Evidence and limits

Machine-readable results and logs are in [verification/](verification/). They record actual executable paths to show environment isolation; those paths are historical evidence, not paths required by the scripts. The local workspace also retains browser screenshots and disposable test environments under publication/test-results and publication/clean-room. New retraining checkpoints remain in publication/git-clone/work and are intentionally excluded from the release.

The 80-table Word document covers all 20 prompts for both training seeds at fixed largest data when comparing models, and at fixed 30M model when comparing data sizes. The HTML explorer includes all 360 saved continuations. The saved-output comparison is exhaustive; the additional fresh-GPU generation check regenerated the first prompt for each run, not all 360 outputs.

The real GitHub release URL cannot be tested until the owner publishes. Local HTTP tested the download/verification/install route; a post-publication download check is included in the publishing guide. Linux CI is provided but was not run on GitHub. macOS, other GPUs and other Python versions were not tested. CPU/GPU sampled text may differ despite fixed seeds. The optional raw rebuild was byte-identical on this Windows setup; different platform path separators or tokenizer/library versions may change serialization or rebuilt token IDs. Use frozen tokenizer/data for recorded-checkpoint evaluation.

No paid services were used, no slides were created, and original research artifacts were preserved. Numerical reproducibility does not establish financial truthfulness: generated amounts and claims remain unverified.
