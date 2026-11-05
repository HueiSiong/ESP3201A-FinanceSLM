# Live comparison verification

Verified locally on **5 November 2026 (Asia/Singapore)** before publishing the comparison interface. This update adds a portable UI and documentation; the original datasets, tokenizer, checkpoint release archives, evaluation records and pinned dependencies are unchanged.

## Test scope

- A separate Git clone of implementation commit `0e03247` was used. The final documentation update does not alter that implementation.
- All three model-size release archives were installed locally through `artifacts.py`; archive and extracted checkpoint fingerprints passed. No training was started.
- The previously verified PSF Python 3.11.9 CPU and CUDA virtual environments were reused; this update introduces no new application dependencies. `pip check` passed in the CPU environment. This is a fresh source/asset checkout test, not a claim of another fresh dependency installation.
- All **five existing regression tests** and **seven new comparison tests** passed in that checkout. The new suite checks matched controls, source inventories, checksum/missing-asset failures, HTML escaping, correct per-model text/score association and output isolation.
- A real Chrome browser exercised **all seven comparison modes** on the RTX 3060 Ti using PyTorch 2.3.1+cu118. Each run displayed new text and fresh full-test scores inline. Every score covered 128,685 targets and matched its historical checkpoint within 1%.
- Same-checkpoint sampling modes displayed identical perplexities. The training-seed mode showed two real outputs and an explicit empty third-model explanation. Fixed-seed text repeated, switching factors cleared stale results, and text-only generation cleared scores. No file downloads or JavaScript errors occurred.
- Both light and dark display modes passed computed foreground/background contrast checks on 85 table/card elements per theme. Minimum measured ratio: **5.955:1**, above the 4.5:1 test threshold. The 1280-pixel viewport had no page-wide horizontal overflow.
- CPU-only PyTorch 2.3.1+cpu generated and repeated text from all three model sizes. A fresh full test on the 15.46M large-data seed-42 model returned **64.4018631301**, compared with recorded **64.4019041668**, over all **128,685 targets**.

The initial browser run passed its functional assertions but Windows sandbox restrictions blocked its temporary server cleanup. That server was stopped after verifying its process identity; the complete browser check was then rerun in the normal Windows user context. This did not change application code or security settings.

## Fresh GPU scores

| Comparison | Left | Middle | Right |
| --- | ---: | ---: | ---: |
| Model parameters: 15.46M / 29.40M / 49.74M | 64.4019 | 58.8972 | 59.2368 |
| Training tokens: 357,288 / 706,370 / 1,048,452 | 78.2211 | 65.2879 | 64.4019 |
| Temperature: 0.3 / 0.8 / 1.2 | 64.4019 | 64.4019 | 64.4019 |
| Top-K: 10 / 40 / 100 | 64.4019 | 64.4019 | 64.4019 |
| Generation seed: 42 / 123 / 7 | 64.4019 | 64.4019 | 64.4019 |
| Maximum new tokens: 40 / 80 / 120 | 64.4019 | 64.4019 | 64.4019 |
| Training seed: 42 / 123 | 64.4019 | 61.3086 | No third seed |

These are checkpoint-level live results, not means over training seeds. The common default prompt was `The company's revenue increased`. Perplexity scores held-out source documents, not the generated continuations.

## Evidence and repeating the checks

- [Browser check results](verification/comparison_browser_checks.json).
- [CPU check results and source fingerprints](verification/comparison_cpu_checks.json).
- [Actual side-by-side output screenshot](comparison/live_outputs_dark.png).

After installing the application dependencies, run `python tests.py` and `python tests_comparison.py`. CI also runs both source suites. For the optional browser verification, install Playwright in a developer environment and either install its Chromium browser or use an installed Chrome browser:

```text
python -m pip install playwright
python scripts/check_comparison_browser.py --device cuda --browser-channel chrome
```

Alternatively use `--device cpu`; full tests will take longer. `--python PATH` selects a separate application interpreter when Playwright is installed in its own environment. The test starts and stops its own temporary server and writes evidence under `work/comparison-browser/`. Playwright is not needed to use the interface.

This revision's live browser checks were performed on Windows; no new Linux browser or macOS result is claimed. The earlier published Windows/Linux single-model verification remains documented in [PUBLICATION_VERIFICATION.md](PUBLICATION_VERIFICATION.md). Hardware, elapsed times and sampled words can differ across machines.
