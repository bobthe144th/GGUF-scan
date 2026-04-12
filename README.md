# gguf-scan

A zero-dependency CLI tool that scans GGUF model files for structural corruption before they cause damage.

-----

## The Problem

Every few weeks, the open-source ML community loses collective hours — sometimes days — to corrupted GGUF releases. A researcher downloads a freshly published model, loads it into llama.cpp or Ollama, and gets a segfault, a silent NaN cascade, or outputs so degraded the model appears genuinely broken. Most of the time, the model isn’t broken. The file is.

This matters more than it sounds.

**It erodes user trust at the exact moment it’s most fragile.** A new model release is the highest-traffic window a researcher will ever have. Corrupted files during that window mean that the first impression thousands of users form is of a broken model — not a file that failed to transfer correctly. That impression sticks. Threads accumulate. Stars stagnate. The maintainer’s reputation takes a hit for an infrastructure failure that was never their fault.

**It derails launches.** Open-source releases don’t have a marketing department to clean up the narrative. When the initial wave of users hits a corrupted file and files issues, the maintainer is suddenly doing damage control instead of onboarding momentum. The launch window — which in OSS is often 48–72 hours — gets consumed by triage instead of excitement.

**It causes wrong conclusions about model quality.** This is the most insidious consequence. When a corrupted GGUF produces incoherent output, the natural assumption is that the underlying model is at fault — bad training, poor architecture, inadequate data. Reviewers publish benchmarks on corrupted weights. Community consensus forms around a model’s “failure” before anyone thinks to check whether the file arrived intact. Novel architectural innovations get dismissed as underperformers because the quantized artifact that reached end users was silently truncated or had misaligned tensor blocks.

The open-source community deserves better tooling here. `gguf-scan` is a small piece of that.

-----

## What It Checks

`gguf-scan` parses the GGUF binary format directly and validates:

|Check                                  |What it catches                                                                                                                                                                                          |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|**Magic bytes**                        |Files that aren’t GGUF at all, or are so corrupted the header is unreadable                                                                                                                              |
|**Version field**                      |Files from future format versions or with a garbage version integer                                                                                                                                      |
|**Header counts**                      |Implausibly large tensor or KV counts that indicate header corruption                                                                                                                                    |
|**KV metadata**                        |Every key-value entry: type validity, string bounds, nested array lengths                                                                                                                                |
|**Tensor info**                        |Every tensor: name, dimension count, shape, ggml_type, data offset                                                                                                                                       |
|**Tensor data bounds**                 |Computes expected byte size from shape + quantization type and confirms it fits in the file — catches truncated downloads and partial writes                                                             |
|**Quantization alignment**             |Verifies element counts are divisible by block sizes for block-quantized formats (Q4_K_M, Q6_K, Q8_0, etc.)                                                                                              |
|**Block scale headers** *(–stat-check)*|Reads every block’s scale field(s) — flags NaN, Inf, zero, and out-of-range values; detects degenerate tensors where all scales are identical                                                            |
|**Sampled stat scan** *(–stat-scan)*   |Randomly samples 1% of blocks per tensor; checks zero-block ratio and longest run of consecutive identical blocks — catches memcpy overwrites and partial-write corruption invisible to structural checks|
|**Deep checksum** *(–deep)*            |SHA-256 of the full file for transfer verification                                                                                                                                                       |

-----

## Installation

No dependencies beyond Python 3.10+. Just copy the script:

```bash
curl -O https://raw.githubusercontent.com/your-org/gguf-scan/main/gguf_scan.py
chmod +x gguf_scan.py
```

Or clone and run directly:

```bash
git clone https://github.com/your-org/gguf-scan
python gguf_scan.py model.gguf
```

-----

## Usage

```
python gguf_scan.py [targets] [options]
```

**Scan a single file:**

```bash
python gguf_scan.py model.Q4_K_M.gguf
```

**Scan a directory recursively:**

```bash
python gguf_scan.py ./models/
```

**Scan multiple files, output JSON:**

```bash
python gguf_scan.py *.gguf --json
```

**Deep scan with checksum (useful for verifying downloads):**

```bash
python gguf_scan.py model.gguf --deep
```

**CI / release pipeline (fail on any issue, including warnings):**

```bash
python gguf_scan.py ./release/*.gguf --warnings-as-errors
echo $?  # 0 = clean, 1 = issues found, 2 = no files found
```

**Only show files with problems:**

```bash
python gguf_scan.py ./models/ --quiet
```

### Options

|Flag                  |Description                                    |
|----------------------|-----------------------------------------------|
|`--json`              |Machine-readable JSON output                   |
|`--deep`              |Compute SHA-256 checksum of each file          |
|`--no-color`          |Disable ANSI color output                      |
|`--warnings-as-errors`|Treat warnings as errors for exit code purposes|
|`--quiet`, `-q`       |Only print files with issues                   |

-----

## Example Output

```
mistral-7b-instruct.Q4_K_M.gguf  ✓ OK  4.1 GB  312ms
  v3  291 tensors  24 KV entries

mixtral-8x7b.Q6_K.gguf  ✗ CORRUPT  26.4 GB  891ms
  v3  963 tensors  31 KV entries
  [ERROR]  TENSOR_DATA_TRUNCATED: Tensor 'blk.14.ffn_up.weight': data ends at
           28,521,496,832, but file is only 26,843,545,600 bytes (short by
           1,677,951,232 bytes) (offset 24601804800)

llama-3-70b.Q2_K.gguf  ✗ CORRUPT  26.0 GB  743ms
  [ERROR]  BAD_MAGIC: Expected b'GGUF', got b'\x00\x00\x00\x00'. Not a GGUF
           file (or badly corrupted).

Summary: 1 OK, 2 failed — 56.5 GB total
```

-----

## Integrating into Release Pipelines

For model maintainers, the highest-value use is scanning artifacts before publishing. A corrupted file caught before upload costs nothing. Caught after — in a GitHub issue from a confused user — costs trust.

**GitHub Actions example:**

```yaml
- name: Validate GGUF artifacts
  run: |
    python gguf_scan.py ./dist/*.gguf --warnings-as-errors --json > scan_report.json
    cat scan_report.json
```

**Pre-upload hook:**

```bash
#!/bin/bash
python gguf_scan.py "$1" || { echo "GGUF validation failed — aborting upload"; exit 1; }
huggingface-cli upload my-org/my-model "$1"
```

-----

## Supported Quantization Types

Block-based formats where `gguf-scan` can verify element-count alignment and data bounds:

`Q4_0` · `Q4_1` · `Q5_0` · `Q5_1` · `Q8_0` · `Q8_1` · `Q2_K` · `Q3_K_S` · `Q3_K_M` · `Q3_K_L` · `Q4_K_S` · `Q4_K_M` · `Q5_K_S` · `Q5_K_M` · `Q6_K`

Dense formats with per-element size validation: `F32` · `F16` · `BF16`

Unknown types trigger a warning rather than a false-negative pass.

-----

## Limitations

- `gguf-scan` validates structure, not numerical correctness. A file can be structurally intact and still contain weights corrupted by a bad quantization run — that requires loading and inference to detect.
- The `general.alignment` KV key (which overrides the default 32-byte alignment) is not yet applied when computing tensor data offsets. This is a minor issue for most models but may produce false warnings for non-standard alignment values.
- Files above ~200 GB trigger a warning but are still scanned.

-----

## Contributing

Issues and PRs welcome. If you find a GGUF variant that produces a false positive or false negative, please open an issue with the relevant metadata — tensor count, ggml types, and version field.

-----

## License

Apache 2.0
