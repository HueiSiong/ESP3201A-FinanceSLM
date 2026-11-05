# Live comparison

This interface runs the saved FinanceSLM checkpoints locally. It displays newly generated continuations in adjacent columns, followed by fresh full-test perplexity for each checkpoint. You do not need to download JSON to see either result.

## Setup once

Follow the Python 3.11, virtual environment and pinned dependency setup in [README.md](../README.md). CPU works; a compatible NVIDIA GPU with the CUDA PyTorch wheel is faster. Neither Docker nor training is required.

From the repository root, using the virtual environment's Python:

```text
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group 15m
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group 30m
python artifacts.py fetch --repo HueiSiong/FinanceSLM --group 50m
python tests_comparison.py
python compare_app.py --open-browser
```

On Windows, replace `python` with `.\.venv\Scripts\python.exe`. On Linux, use `.venv/bin/python`. Downloads come from the existing public v1.0.0 release and are checked against `assets.json`. These groups contain all 18 selected research checkpoints; the interface uses six of them. Approximately 2.11 GB is downloaded and 2.27 GB extracted. The `demo` and `raw` groups are not needed for this interface.

If the release ZIPs are already on disk, use `python artifacts.py install --from-dir PATH --group 15m`, repeated for `30m` and `50m`. There is no download on each comparison run.

## Open and run

1. Run `python compare_app.py --open-browser`, or double-click `launch_comparison.cmd` on Windows after setup.
2. Open **http://127.0.0.1:7863** in your browser. Keep the terminal running. Ctrl+C stops the server.
3. Choose the exact factor values and review the table. Highlighted rows show every difference, including validation-selected checkpoint steps.
4. Keep the same starting words, then click **Run comparison: text + full-test scores**.
5. Generated text appears first; scores appear underneath when full evaluation finishes. Each score covers the same 15 test documents and 128,685 target tokens, not the typed prompt or its continuation.

**Generate text only (faster)** skips the full test and explicitly labels the score as unavailable. Switching factors clears the previous result. Optional evidence files are saved automatically under `work/comparison/` and can be downloaded from the collapsed accordion. Checkpoints, frozen data and historical results are never overwritten.

Use `--device cpu` to force CPU or `--device cuda` for CUDA. Automatic selection uses an available NVIDIA GPU, with no hardcoded GPU model. If port 7863 is occupied, use `python compare_app.py --port 7864 --open-browser` and open that port. If weights are missing, the interface tells you which release group to install. If a checksum differs, preserve/move the altered file before reinstalling its group.

## What each comparison changes

| Factor | Values | Controls |
| --- | --- | --- |
| Model parameters | 15,464,064 / 29,401,600 / 49,737,600 | Large training pool; training seed 42 |
| Training text | 357,288 / 706,370 / 1,048,452 tokens; 36 / 73 / 109 documents | 15.46M model; training seed 42 |
| Temperature | 0.3 / 0.8 / 1.2 | Same 15.46M large-data seed-42 checkpoint |
| Top-K | 10 / 40 / 100 | Same checkpoint |
| Generation seed | 42 / 123 / 7 | Same checkpoint |
| Maximum new tokens | 40 / 80 / 120 | Same checkpoint |
| Training seed | 42 / 123 | 15.46M model; large training pool; only two trained seeds |

Unless it is the changed factor, generation uses 80 maximum new tokens, temperature 0.8, top-K 40 and generation seed 42. Every column receives the same prompt. All models share the frozen 8,000-token tokenizer and 256-token context. The maximum training schedule and validation-selection rule are shared; selected steps and consumed token positions can differ and are displayed explicitly. Model size changes width, depth and heads together. Data size also changes report composition.

## A clear demonstration

Start with **Model parameters** and the default prompt `The company's revenue increased`. Run the combined button and show these reference scores (fresh values can differ slightly):

| Large-data, training-seed-42 model | Recorded full-test perplexity |
| --- | ---: |
| 15.46M | 64.4019 |
| 29.40M | 58.8972 |
| 49.74M | 59.2368 |

Then choose **Training text**, retaining the same prompt:

| Training tokens, 15.46M, seed 42 | Recorded full-test perplexity |
| --- | ---: |
| 357,288 | 78.2211 |
| 706,370 | 65.2879 |
| 1,048,452 | 64.4019 |

Finally choose **Temperature**: the generated continuations can differ while all three full-test scores are identical. The checkpoint is identical, so the full test is computed once and reused across those three columns. Temperature, top-K, generation seed and output length affect sampling, not teacher-forced test loss.

Perplexity is `exp(sum of target-token negative log probabilities / number of scored targets)`. Lower is better prediction of held-out text; it is not evidence that generated amounts are correct. The percentage reduction is relative to the left column. The 1% tolerance checks agreement with the same historical checkpoint, not statistical significance. These live scores use seed 42 and are not the research tables' means across two training seeds. The selected deployment checkpoint in the original app is seed 123, which explains its different 57.6452 score.

The training-seed option has only two outputs: no third trained seed exists. Financial coherence remains limited; do not present fluent-looking text as verified financial advice. Same-device fixed-seed generation should repeat, but CPU and GPU may sample different words. Full scoring on CPU can take several minutes; allow it to finish or use the text-only button for a shorter demonstration.
