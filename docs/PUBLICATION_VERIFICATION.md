# Published setup verification

Verified **5 November 2026 (Asia/Singapore)** after publishing the public repository and release.

- Repository: https://github.com/HueiSiong/FinanceSLM
- Release and all five model/data archives: https://github.com/HueiSiong/FinanceSLM/releases/tag/v1.0.0
- Tested code commit: `cc4635dfbf08c4b36d3e37bb1bf929260420d093`
- Start here: [EVALUATE.md](../EVALUATE.md); agent instructions: [AGENTS.md](../AGENTS.md).

## Fresh Windows public-download test

Cloned the actual public GitHub repository with Git credential helpers disabled. Created a new virtual environment from PSF-signed Python 3.11.9, installed the pinned CPU PyTorch wheel and requirements.lock.txt, and passed pip check plus all five source regression tests.

Downloaded **all five release archives without GitHub credentials**, checked archive and extracted-file SHA-256 fingerprints, loaded the selected checkpoint, passed the smoke checks, and ran the full held-out evaluation.

Full-test perplexity: **57.6451208370**, over **128,685 target tokens**. Relative difference from the recorded checkpoint score: **0.00006413%**. The partial smoke score is a separate diagnostic.

## Independent Linux verification

The standard GitHub-hosted **ubuntu-24.04** runner installed Python 3.11 and the pinned CPU dependencies. Source tests passed. The model-check job downloaded the public default release asset, verified it, passed the numerical/generation smoke checks and completed the full held-out evaluation.

Linux full-test perplexity: **57.6451403989**, over **128,685 target tokens**.

- [Full source and model workflow](https://github.com/HueiSiong/FinanceSLM/actions/runs/41876271309) — **success**.
- [Initial source-check workflow](https://github.com/HueiSiong/FinanceSLM/actions/runs/41875344671) — **success**.

The workflows use standard public-repository runners and upload no Actions artifacts or caches. No paid services, Git LFS or hosted-model API keys were used.

## Scope

These checks validate the published download path on Windows and a separate Linux runner. The earlier GPU training, all-18-checkpoint evaluation, raw-data rebuild and browser checks are documented in [SETUP_VERIFICATION.md](SETUP_VERIFICATION.md). They were not relabelled as new Linux/GPU results. macOS and other hardware remain untested. Fixed seeds do not promise bit-identical retraining, and model-generated financial claims remain unverified.

The Word report, all recorded run results, frozen datasets and same-prompt comparisons are included in the repository. Large selected checkpoints and raw HTML snapshots are release assets. See [PUBLICATION_VERIFICATION.json](PUBLICATION_VERIFICATION.json) for machine-readable publication checks.
