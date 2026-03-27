# Analysis

This folder contains the document analysis pipeline used for policy-oriented qualitative analysis in WP5. It combines:

- policy extraction from source PDFs
- Gioia first-order concept extraction
- Gioia second-order theme generation

The scripts are designed for batch processing of PDF documents and produce structured PDF and spreadsheet outputs that can support qualitative coding, interpretation, and reporting.

## Overview

The pipeline is orchestrated by `main_analysis.py` and runs three stages in sequence:

1. `policy_analysis_simplified_chutes.py`
   Extracts explicitly named policy initiatives, strategies, programmes, directives, regulations, and related policy metadata from source PDFs.
2. `code_gioia_pdf_and_excel.py`
   Extracts Gioia first-order concepts from the same source PDFs and exports them for review.
3. `gioia_second_order.py`
   Builds on the first-order outputs to generate second-order themes using the Gioia methodology.

By default, the pipeline reads PDFs from `analysis/input/` and writes outputs under `analysis/output/`.

## Folder Structure

```text
analysis/
  .env
  env_config.py
  main_analysis.py
  policy_analysis_simplified_chutes.py
  code_gioia_pdf_and_excel.py
  gioia_second_order.py
  input/
  output/
```

## Main Files

### `main_analysis.py`

This is the top-level entry point for the full pipeline. It:

- resolves input and output locations
- runs the three stages in the correct order
- creates stage-specific output folders under the configured output root

Default output subfolders:

- `output/policy_analysis`
- `output/gioia_first_order`
- `output/gioia_second_order`

### `policy_analysis_simplified_chutes.py`

This script performs policy-focused extraction from PDFs. It:

- reads source PDFs with `pdfplumber`
- splits documents into chunks
- sends chunk prompts to the Chutes API
- parses and repairs JSON responses when needed
- checkpoints chunk-level progress to JSONL files
- merges chunk outputs into one document-level record
- generates a short document summary
- writes one PDF report per input PDF

The extracted information can include:

- title of named policy initiatives
- issuing body
- policy level
- jurisdictions
- year and date range
- policy type and instruments
- green and digital transition references
- objectives and measures
- target groups and sectors
- funding
- social inclusion
- monitoring and evaluation
- short verbatim quotes from the source text

### `code_gioia_pdf_and_excel.py`

This script performs Gioia first-order coding. It is aimed at extracting concepts that remain close to the wording of the source document. It:

- identifies first-order concepts from the text
- attaches short supporting quotes
- merges concepts across chunks
- writes one PDF report per source document
- exports a spreadsheet of first-order concepts

Typical concept categories include:

- objectives
- measures
- instruments
- target groups
- governance
- funding
- problem framing
- resilience framing
- green transition
- digital transition
- social inclusion
- monitoring and evaluation

### `gioia_second_order.py`

This script generates second-order themes from the first-order concepts. It:

- reuses the first-order extraction logic
- merges document-level first-order concepts
- makes a second LLM pass to generate higher-level interpretive themes
- writes one output PDF per document

This stage is intended to support more abstract qualitative synthesis while still preserving a structured audit trail back to first-order concepts and quotes.

### `env_config.py`

This file centralizes environment-variable loading and typed config helpers. It:

- loads environment variables from the project `.env`
- resolves relative paths from the project root
- converts values into `str`, `int`, `float`, `bool`, `list`, and `Path`

## Input and Output

### Input

The default input folder is:

```text
analysis/input/
```

Expected input type:

- `.pdf` files for the main analysis pipeline

### Output

The default output root is:

```text
analysis/output/
```

The pipeline creates separate subfolders for each stage.

Common output artifacts include:

- PDF extraction reports
- PDF coding reports
- Excel files for first-order concepts
- JSONL checkpoint files for chunk-level resume support
- debug files such as bad-output or HTTP-error snapshots when API responses fail

## Requirements

Install the required Python packages in your environment. Based on the current scripts, you will likely need:

```bash
pip install python-dotenv requests pdfplumber reportlab openpyxl
```

Depending on your broader setup, you may already have these available in a project environment.

## Environment Variables

This project depends heavily on environment-based configuration. The scripts read values through `env_config.py`.

At minimum, you will usually want to define:

```env
CHUTES_API_KEY=your_api_key
CHUTES_BASE_URL=https://llm.chutes.ai/v1
CHUTES_MODEL=deepseek-ai/DeepSeek-V3-0324
```

### Pipeline Paths

These variables control the main pipeline entry point:

```env
ANALYSIS_INPUT_FOLDER=analysis/input
ANALYSIS_OUTPUT_ROOT=analysis/output
```

### Policy Analysis Settings

Useful variables for `policy_analysis_simplified_chutes.py` include:

```env
POLICY_ANALYSIS_MODEL=Qwen/Qwen2.5-72B-Instruct
POLICY_ANALYSIS_INCLUDE_CHUNK_OUTPUTS=false
POLICY_ANALYSIS_INCLUDE_MERGED_RECORD=true
POLICY_ANALYSIS_INCLUDE_LLM_MERGE=false
POLICY_ANALYSIS_INPUT_FOLDER=analysis/input
POLICY_ANALYSIS_OUTPUT_FOLDER=analysis/output/policy_analysis
```

### Shared Chunking / Rate Settings

These are used across the batch-style scripts:

```env
CHARS_PER_CHUNK=4000
SLEEP_BETWEEN_CALLS=0.5
```

### Gioia First-Order Settings

Common variables include:

```env
GIOIA_FIRST_ORDER_MODEL=deepseek-ai/DeepSeek-V3-0324
GIOIA_FIRST_ORDER_MAX_RETRIES=8
GIOIA_FIRST_ORDER_BASE_BACKOFF_S=2.0
GIOIA_FIRST_ORDER_MAX_BACKOFF_S=60.0
GIOIA_FIRST_ORDER_INCLUDE_CHUNK_OUTPUTS=false
GIOIA_FIRST_ORDER_INCLUDE_MERGED_RECORD=true
GIOIA_FIRST_ORDER_EXCEL_FILENAME=GIOIA_FIRST_ORDER_CONCEPTS.xlsx
```

### Gioia Second-Order Settings

Common variables include:

```env
GIOIA_SECOND_ORDER_MODEL=Qwen/Qwen2.5-72B-Instruct
GIOIA_SECOND_ORDER_MAX_RETRIES=8
GIOIA_SECOND_ORDER_BASE_BACKOFF_S=2.0
GIOIA_SECOND_ORDER_MAX_BACKOFF_S=60.0
GIOIA_SECOND_ORDER_INCLUDE_CHUNK_OUTPUTS=false
GIOIA_SECOND_ORDER_INCLUDE_MERGED_RECORD=true
GIOIA_SECOND_ORDER_INCLUDE_SECOND_ORDER=true
SECOND_ORDER_MAX_TOKENS=2500
SECOND_ORDER_RETRIES=3
```

## Running the Full Pipeline

From the repository root:

```bash
python analysis/main_analysis.py
```

You can also override the default folders:

```bash
python analysis/main_analysis.py --input-folder analysis/input --output-root analysis/output
```

## Running Individual Stages

### Policy Analysis

```bash
python analysis/policy_analysis_simplified_chutes.py
```

This script expects `POLICY_ANALYSIS_INPUT_FOLDER` and `POLICY_ANALYSIS_OUTPUT_FOLDER` to be set if you are not relying on defaults elsewhere.

### Gioia First-Order

```bash
python analysis/code_gioia_pdf_and_excel.py
```

### Gioia Second-Order

```bash
python analysis/gioia_second_order.py
```

## Workflow Notes

### Checkpointing and Resume

The policy analysis script includes crash-safe checkpointing using JSONL files. This helps with long-running jobs and API interruptions. If a run stops midway, the script can reuse saved chunk outputs instead of starting entirely from scratch.

### PDF Reports

The policy extraction stage produces human-readable PDF reports that include:

- a short document summary
- an executive summary of identified policies
- detailed policy metadata
- optional chunk-level technical appendices

### LLM Robustness

The scripts include several safeguards for batch processing:

- retry logic
- JSON repair prompts
- model fallbacks in Gioia scripts
- backoff behavior for capacity errors
- debug output files when responses fail

## GitHub and Security Notes

- Do not upload `.env`.
- Do not upload API keys.
- Review `analysis/input/` and `analysis/output/` before publishing, because they may contain private source material or generated research outputs.

A minimal `.gitignore` for this folder should usually include:

```gitignore
.env
__pycache__/
*.py[cod]
input/
output/
```

Adjust that depending on whether you want to publish example inputs or sample outputs.

## Suggested Repository Improvements

If you later want to turn `analysis` into a more reusable standalone project, useful next steps would be:

- add a dedicated `requirements.txt` or `pyproject.toml`
- provide an `.env.example`
- document exact output filenames per stage
- add a small sample input document and expected output structure
- separate research-specific prompt text from processing code

## Summary

This folder is a structured LLM-assisted qualitative analysis workflow for policy documents. It supports:

- document-level policy extraction
- Gioia first-order coding
- Gioia second-order synthesis
- batch processing of PDFs
- resumable chunked analysis
- research-oriented reporting outputs

It is best understood as a practical research pipeline rather than a generic software library.
