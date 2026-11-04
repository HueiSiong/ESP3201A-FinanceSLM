# Provenance and licensing scope

This is an academic FinanceSLM project containing a custom PyTorch Transformer and recorded experiments. The architecture uses RoPE, SwiGLU and RMSNorm; references are in the research report. No pretrained language-model weights were used.

The financial-text dataset is extracted from company SEC 10-K filings. Source URLs, report years and company tickers are retained in the JSONL records and metadata CSVs. This package does not claim copyright ownership of the underlying company filings. Model outputs can repeat source wording and are not verified company facts.

No new blanket software/data license has been assigned during packaging. The project team can select an appropriate code license when publishing. Third-party Python dependencies retain their respective licenses and are downloaded by pip rather than bundled into the Git repository.
