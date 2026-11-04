# Instructions for an evaluating agent

Read EVALUATE.md and README.md first. Use Python 3.11 in a fresh virtual environment. Do not use a copied `.runtime` or `venv`. CPU inference is sufficient for verification; training is optional and must not be started just to demonstrate the saved model.

Install the chosen PyTorch 2.3.1 wheel plus requirements.lock.txt. Fetch the default demo asset with `python artifacts.py fetch --repo HueiSiong/FinanceSLM`. This is a public repository; no GitHub login or model API key is required to download it. If release archives are supplied locally, use `install --from-dir` instead. Verify assets before loading weights.

Run `python tests.py`, `python research.py smoke --device cpu`, then `python research.py evaluate --device cpu`. Start `python app.py` to expose the local Gradio interface. All generated outputs belong in `work/`. Do not overwrite `data/final`, `results/final`, or the recorded comparison documents. Training functions refuse nonempty output directories.

Frozen-model evaluation is distinct from stochastic training reproducibility. Do not treat a partial smoke-test perplexity as the published full-test score. Same-device generation should repeat with the same seed; CPU/GPU wording need not be identical. Full score comparison uses a 1% relative tolerance for the published, hash-verified checkpoint.

The original reports are HTML, not PDF. The model is a financial text-completion prototype and its generated numbers are unverified. Offline comparison files contain exact saved outputs, not newly generated or edited prose.

No publication, messages, paid services, or automatic retraining are necessary for verification.
