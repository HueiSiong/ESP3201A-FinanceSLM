# Windows setup verification: 5 November 2026

All eight local verification suites passed again in fresh environments created from an official Python Software Foundation-signed Python 3.11.9 Windows installation. CPU and CUDA environments installed the unchanged pinned dependencies; Playwright 1.64.0 ran in a separate fresh browser-test environment.

| Suite | Result |
| --- | --- |
| CPU setup, full evaluation and short training | Passed |
| CUDA setup, full evaluation and complete 15M training | Passed |
| Actual local Git commit/clone | Passed |
| Raw-source corpus rebuild | Passed; all 149 files byte-identical |
| Gradio and offline comparison browser tests | Passed |
| Original-file preservation and Word comparison contents | Passed |
| All 18 saved checkpoint evaluations | Passed; full-test scores matched exactly on the tested GPU |
| Final source archive and fresh Git clone checks | Passed |

The selected model scored full-test perplexity **57.64512083695659 on CPU** and **57.64515780490899 on CUDA**, over **128,685 target tokens**. The fresh 15M/large/seed42 run completed 3,072 steps and scored **64.39790054917789**. This retraining check did not replace any historical experiment.

The Windows issue was traced to an unsigned virtual-environment redirector in the previous standalone Python distribution. CodeIntegrity event 3077 named Smart App Control's VerifiedAndReputableDesktop enforcement policy; its correlated 3089 event reported zero signatures. Fresh virtual environments from the PSF-signed installation executed normally. Defender, real-time protection, tamper protection and Smart App Control stayed enabled; no exclusions or bypasses were used.

For Windows, create your own virtual environment from an official Python installation instead of copying another computer's environment. The project's existing Python 3.11 baseline was retained. **3.11.9 is the last official Windows installer for that series, not the newest security release**; later 3.11 updates are source-only. See the [official release information](https://www.python.org/downloads/release/python-31118/). This local compatibility verification does not certify every operating system or GPU.

The original October verification and logs remain in REPRODUCIBILITY.md and verification/. The owner's detailed November repair logs remain in their local workspace. Current public-repository workflow results and the post-publication verification report should be consulted for actual GitHub/Linux download verification.
