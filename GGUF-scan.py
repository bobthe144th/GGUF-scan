#!/usr/bin/env python3
“””
gguf-scan: CLI tool to detect corruption in GGUF model files.

Usage:
python gguf_scan.py model.gguf
python gguf_scan.py ./models/          # recursive scan
python gguf_scan.py *.gguf –json
python gguf_scan.py model.gguf –deep  # checksum tensor data blocks
“””

import argparse
import json
import struct
import sys
import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

# ── F16 decode (no numpy required) ───────────────────────────────────────────

def f16_to_f32(bits: int) -> float:
“”“Decode a raw uint16 IEEE 754 half-precision value to Python float.”””
sign     = (bits >> 15) & 0x1
exponent = (bits >> 10) & 0x1F
mantissa =  bits        & 0x3FF
if exponent == 0x1F:
return math.copysign(math.inf if mantissa == 0 else math.nan, -1 if sign else 1)
if exponent == 0:
val = mantissa / 1024.0 * (2 ** -14)
else:
val = (1 + mantissa / 1024.0) * (2 ** (exponent - 15))
return -val if sign else val

# ── GGUF constants ────────────────────────────────────────────────────────────

GGUF_MAGIC = b”GGUF”
GGUF_VERSION_MIN = 1
GGUF_VERSION_MAX = 3

# GGUFMetadataValueType

class GGUFValueType(IntEnum):
UINT8   = 0
INT8    = 1
UINT16  = 2
INT16   = 3
UINT32  = 4
INT32   = 5
FLOAT32 = 6
BOOL    = 7
STRING  = 8
ARRAY   = 9
UINT64  = 10
INT64   = 11
FLOAT64 = 12

VALUE_SIZE = {
GGUFValueType.UINT8:   1,
GGUFValueType.INT8:    1,
GGUFValueType.UINT16:  2,
GGUFValueType.INT16:   2,
GGUFValueType.UINT32:  4,
GGUFValueType.INT32:   4,
GGUFValueType.FLOAT32: 4,
GGUFValueType.BOOL:    1,
GGUFValueType.UINT64:  8,
GGUFValueType.INT64:   8,
GGUFValueType.FLOAT64: 8,
}

# GGMLType sizes in bytes per element

GGML_TYPE_SIZE = {
0:  4,      # F32
1:  2,      # F16
2:  None,   # Q4_0  — block-based, handled separately
3:  None,   # Q4_1
6:  None,   # Q5_0
7:  None,   # Q5_1
8:  None,   # Q8_0
9:  None,   # Q8_1
10: None,   # Q2_K
11: None,   # Q3_K_S
12: None,   # Q3_K_M
13: None,   # Q3_K_L
14: None,   # Q4_K_S
15: None,   # Q4_K_M
16: None,   # Q5_K_S
17: None,   # Q5_K_M
18: None,   # Q6_K
19: 1,      # Q8_K (approx)
20: 2,      # IQ2_XXS
21: 2,      # IQ2_XS
22: 2,      # IQ3_XXS
23: 2,      # IQ1_S
24: 2,      # IQ4_NL
25: 2,      # IQ3_S
26: 2,      # IQ3_M
27: 2,      # IQ2_S
28: 2,      # IQ2_M
29: 2,      # IQ4_XS
30: 2,      # IQ1_M
31: 1,      # BF16 (2 bytes actual; placeholder)
}

# Block sizes (elements per block, bytes per block)

GGML_BLOCK_INFO = {
# type_id: (block_elems, block_bytes)
2:  (32, 18),    # Q4_0
3:  (32, 20),    # Q4_1
6:  (32, 22),    # Q5_0
7:  (32, 24),    # Q5_1
8:  (32, 34),    # Q8_0
9:  (32, 36),    # Q8_1
10: (256, 84),   # Q2_K
11: (256, 110),  # Q3_K_S
12: (256, 134),  # Q3_K_M
13: (256, 142),  # Q3_K_L
14: (256, 144),  # Q4_K_S
15: (256, 144),  # Q4_K_M
16: (256, 176),  # Q5_K_S
17: (256, 176),  # Q5_K_M
18: (256, 210),  # Q6_K
}

# Scale field layout per block type for stat-check.

# Each entry: list of (byte_offset_in_block, field_type)

# field_type: ‘f16’ | ‘i8’ | ‘u8’

# For K-quants the primary super-block scale is always the first f16.

BLOCK_SCALE_LAYOUT = {
2:  [(0,  ‘f16’)],            # Q4_0:   d  at byte 0
3:  [(0,  ‘f16’), (2, ‘f16’)],# Q4_1:   d, m
6:  [(0,  ‘f16’)],            # Q5_0:   d
7:  [(0,  ‘f16’), (2, ‘f16’)],# Q5_1:   d, m
8:  [(0,  ‘f16’)],            # Q8_0:   d
9:  [(0,  ‘f16’), (2, ‘f16’)],# Q8_1:   d, s
10: [(0,  ‘f16’), (2, ‘f16’)],# Q2_K:   d, dmin
11: [(0,  ‘f16’)],            # Q3_K_S: d
12: [(0,  ‘f16’), (2, ‘f16’)],# Q3_K_M: d, dmin
13: [(0,  ‘f16’), (2, ‘f16’)],# Q3_K_L: d, dmin
14: [(0,  ‘f16’), (2, ‘f16’)],# Q4_K_S: d, dmin
15: [(0,  ‘f16’), (2, ‘f16’)],# Q4_K_M: d, dmin
16: [(0,  ‘f16’), (2, ‘f16’)],# Q5_K_S: d, dmin
17: [(0,  ‘f16’), (2, ‘f16’)],# Q5_K_M: d, dmin
18: [(0,  ‘f16’)],            # Q6_K:   d
}

# Reasonable absolute upper bound on a block scale value.

# Scales encode the mapping from integer codes → float weights.

# Real-world transformer weights rarely exceed ±10 in magnitude;

# scales above ~100 almost certainly indicate bit corruption.

MAX_SANE_SCALE = 100.0

# Fraction of a tensor’s blocks sampled during –stat-check (Tier 2).

STAT_SAMPLE_FRACTION = 0.01   # 1 %
STAT_SAMPLE_MIN      = 64     # always check at least this many blocks
STAT_SAMPLE_MAX      = 4096   # cap so large tensors don’t dominate runtime

# A tensor whose scale std-dev is zero (all blocks identical) is suspicious

# unless it has very few blocks (bias vectors, small embeddings, etc.)

MIN_BLOCKS_FOR_VARIANCE_CHECK = 16

# If this fraction of sampled blocks are all-zero we flag it.

ZERO_BLOCK_RATIO_THRESHOLD = 0.20   # 20 %

# Consecutive identical blocks that trigger a run-corruption warning.

IDENTICAL_RUN_THRESHOLD = 8

# ── Result types ──────────────────────────────────────────────────────────────

SEVERITY_OK      = “OK”
SEVERITY_WARNING = “WARNING”
SEVERITY_ERROR   = “ERROR”

@dataclass
class Issue:
severity: str
code: str
message: str
offset: Optional[int] = None

@dataclass
class TensorStatResult:
name: str
ggml_type: int
n_blocks: int
n_sampled: int
bad_scale_count: int        # NaN / Inf / zero / out-of-range scales
zero_block_ratio: float     # fraction of sampled blocks that are all-zero
scale_mean: Optional[float]
scale_std: Optional[float]
max_identical_run: int      # longest run of consecutive identical blocks

@dataclass
class ScanResult:
path: str
file_size: int
ok: bool
version: Optional[int] = None
tensor_count: Optional[int] = None
kv_count: Optional[int] = None
issues: list = field(default_factory=list)
deep_checksum: Optional[str] = None
tensor_stats: list = field(default_factory=list)   # list[TensorStatResult]
elapsed_ms: float = 0.0

```
def to_dict(self):
    return {
        "path": self.path,
        "file_size": self.file_size,
        "ok": self.ok,
        "version": self.version,
        "tensor_count": self.tensor_count,
        "kv_count": self.kv_count,
        "issues": [
            {"severity": i.severity, "code": i.code,
             "message": i.message, "offset": i.offset}
            for i in self.issues
        ],
        "deep_checksum": self.deep_checksum,
        "tensor_stats": [
            {
                "name": ts.name,
                "ggml_type": ts.ggml_type,
                "n_blocks": ts.n_blocks,
                "n_sampled": ts.n_sampled,
                "bad_scale_count": ts.bad_scale_count,
                "zero_block_ratio": round(ts.zero_block_ratio, 4),
                "scale_mean": ts.scale_mean,
                "scale_std": ts.scale_std,
                "max_identical_run": ts.max_identical_run,
            }
            for ts in self.tensor_stats
        ],
        "elapsed_ms": round(self.elapsed_ms, 1),
    }
```

# ── Low-level reader ──────────────────────────────────────────────────────────

class BinaryReader:
def **init**(self, data: bytes):
self.data = data
self.pos = 0
self.size = len(data)

```
def remaining(self) -> int:
    return self.size - self.pos

def read(self, n: int) -> bytes:
    if self.pos + n > self.size:
        raise EOFError(f"Unexpected EOF at offset {self.pos}: wanted {n} bytes, {self.remaining()} remain")
    chunk = self.data[self.pos:self.pos + n]
    self.pos += n
    return chunk

def u8(self)  -> int: return struct.unpack_from("<B", self.read(1))[0]
def u16(self) -> int: return struct.unpack_from("<H", self.read(2))[0]
def u32(self) -> int: return struct.unpack_from("<I", self.read(4))[0]
def u64(self) -> int: return struct.unpack_from("<Q", self.read(8))[0]
def i8(self)  -> int: return struct.unpack_from("<b", self.read(1))[0]
def i16(self) -> int: return struct.unpack_from("<h", self.read(2))[0]
def i32(self) -> int: return struct.unpack_from("<i", self.read(4))[0]
def i64(self) -> int: return struct.unpack_from("<q", self.read(8))[0]
def f32(self) -> float: return struct.unpack_from("<f", self.read(4))[0]
def f64(self) -> float: return struct.unpack_from("<d", self.read(8))[0]
def bool_(self) -> bool: return bool(self.u8())

def string(self) -> str:
    length = self.u64()
    if length > 1_000_000:
        raise ValueError(f"Implausibly long string ({length} bytes) at offset {self.pos}")
    raw = self.read(length)
    return raw.decode("utf-8", errors="replace")
```

# ── Scanner ───────────────────────────────────────────────────────────────────

MAX_FILE_SIZE = 200 * 1024 * 1024 * 1024  # 200 GB sanity cap

# ── Quantization stat checker ─────────────────────────────────────────────────

def _read_f16_at(data: bytes, offset: int) -> float:
bits = struct.unpack_from(”<H”, data, offset)[0]
return f16_to_f32(bits)

def _read_i8_at(data: bytes, offset: int) -> int:
return struct.unpack_from(”<b”, data, offset)[0]

def _block_is_zero(data: bytes, block_start: int, block_bytes: int) -> bool:
return all(b == 0 for b in data[block_start:block_start + block_bytes])

def _blocks_equal(data: bytes, off_a: int, off_b: int, block_bytes: int) -> bool:
return data[off_a:off_a + block_bytes] == data[off_b:off_b + block_bytes]

def stat_check_tensor(
data: bytes,
tensor_name: str,
ggml_type: int,
abs_offset: int,
n_blocks: int,
full_scan: bool,          # True = Tier 2 (sampled); False = Tier 1 (headers only, sequential)
) -> TensorStatResult:
“””
Analyse block-level scale fields for a single quantized tensor.

```
Tier 1 (full_scan=False): walks ALL block headers sequentially, checks
every scale for NaN/Inf/zero/out-of-range. Fast — reads only the first
few bytes of each block.

Tier 2 (full_scan=True): randomly samples up to STAT_SAMPLE_MAX blocks,
computes mean/std of scales, zero-block ratio, and longest identical-block
run within the sample.
"""
block_info   = GGML_BLOCK_INFO.get(ggml_type)
scale_layout = BLOCK_SCALE_LAYOUT.get(ggml_type)

# Return empty result for types we can't inspect
if block_info is None or scale_layout is None:
    return TensorStatResult(
        name=tensor_name, ggml_type=ggml_type,
        n_blocks=n_blocks, n_sampled=0,
        bad_scale_count=0, zero_block_ratio=0.0,
        scale_mean=None, scale_std=None, max_identical_run=0,
    )

_block_elems, block_bytes = block_info

# ── Choose which block indices to visit ───────────────────────────────────
if full_scan:
    n_sample = max(STAT_SAMPLE_MIN, min(STAT_SAMPLE_MAX, int(n_blocks * STAT_SAMPLE_FRACTION)))
    if n_sample >= n_blocks:
        indices = list(range(n_blocks))
    else:
        # Evenly-strided sample (deterministic, no random seed needed)
        step = n_blocks / n_sample
        indices = [int(i * step) for i in range(n_sample)]
else:
    # Tier 1: all blocks, but we only read the scale bytes — very fast
    indices = list(range(n_blocks))

# ── Walk chosen blocks ────────────────────────────────────────────────────
bad_scale_count  = 0
zero_block_count = 0
scales: list[float] = []

prev_block_idx: Optional[int] = None
current_run   = 1
max_run       = 1

for idx in indices:
    block_start = abs_offset + idx * block_bytes

    # Bounds guard (structural check should have caught truncation, but be safe)
    if block_start + block_bytes > len(data):
        break

    # ── Scale field validation (Tier 1 core) ─────────────────────────────
    block_has_bad_scale = False
    for field_off, field_type in scale_layout:
        fpos = block_start + field_off
        if field_type == 'f16':
            val = _read_f16_at(data, fpos)
            if not math.isfinite(val) or val == 0.0 or abs(val) > MAX_SANE_SCALE:
                block_has_bad_scale = True
            else:
                scales.append(abs(val))
        elif field_type == 'i8':
            val = _read_i8_at(data, fpos)
            scales.append(float(val))
    if block_has_bad_scale:
        bad_scale_count += 1

    # ── Tier 2 extras: zero-block and identical-run ───────────────────────
    if full_scan:
        if _block_is_zero(data, block_start, block_bytes):
            zero_block_count += 1

        if prev_block_idx is not None and idx == prev_block_idx + 1:
            prev_start = abs_offset + prev_block_idx * block_bytes
            if _blocks_equal(data, prev_start, block_start, block_bytes):
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        else:
            current_run = 1

    prev_block_idx = idx

n_sampled = len(indices)
zero_ratio = zero_block_count / n_sampled if n_sampled > 0 else 0.0

# ── Compute mean / std of collected scale values ──────────────────────────
scale_mean: Optional[float] = None
scale_std:  Optional[float] = None
if scales:
    scale_mean = sum(scales) / len(scales)
    if len(scales) > 1:
        variance = sum((s - scale_mean) ** 2 for s in scales) / len(scales)
        scale_std = math.sqrt(variance)
    else:
        scale_std = 0.0

return TensorStatResult(
    name=tensor_name,
    ggml_type=ggml_type,
    n_blocks=n_blocks,
    n_sampled=n_sampled,
    bad_scale_count=bad_scale_count,
    zero_block_ratio=zero_ratio,
    scale_mean=round(scale_mean, 6) if scale_mean is not None else None,
    scale_std=round(scale_std,  6) if scale_std  is not None else None,
    max_identical_run=max_run,
)
```

def emit_stat_issues(ts: TensorStatResult, issue_fn) -> None:
“”“Convert a TensorStatResult into scanner issues.”””

```
if ts.n_sampled == 0:
    return

# Bad scales (Tier 1)
if ts.bad_scale_count > 0:
    frac = ts.bad_scale_count / ts.n_sampled
    sev  = SEVERITY_ERROR if frac > 0.02 else SEVERITY_WARNING
    issue_fn(sev, "BAD_BLOCK_SCALES",
             f"Tensor '{ts.name}': {ts.bad_scale_count}/{ts.n_sampled} blocks "
             f"have NaN/Inf/zero/out-of-range scale fields ({frac:.1%})")

# Degenerate variance — all scales identical (Tier 1, large tensors)
if (ts.scale_std is not None
        and ts.scale_std == 0.0
        and ts.n_blocks >= MIN_BLOCKS_FOR_VARIANCE_CHECK
        and ts.bad_scale_count == 0):
    issue_fn(SEVERITY_WARNING, "ZERO_SCALE_VARIANCE",
             f"Tensor '{ts.name}': all {ts.n_sampled} sampled blocks have "
             f"identical scale values (mean={ts.scale_mean:.4f}) — "
             f"possible copy-paste corruption")

# Tier 2 extras
if ts.zero_block_ratio > ZERO_BLOCK_RATIO_THRESHOLD:
    issue_fn(SEVERITY_WARNING, "HIGH_ZERO_BLOCK_RATIO",
             f"Tensor '{ts.name}': {ts.zero_block_ratio:.1%} of sampled "
             f"blocks are all-zero (threshold {ZERO_BLOCK_RATIO_THRESHOLD:.0%})")

if ts.max_identical_run >= IDENTICAL_RUN_THRESHOLD:
    issue_fn(SEVERITY_WARNING, "IDENTICAL_BLOCK_RUN",
             f"Tensor '{ts.name}': run of {ts.max_identical_run} consecutive "
             f"identical blocks detected — possible memcpy / write corruption")
```

def scan_file(path: Path, deep: bool = False, stat_check: bool = False, stat_scan: bool = False) -> ScanResult:
t0 = time.monotonic()
result = ScanResult(
path=str(path),
file_size=0,
ok=False,
)

```
def issue(severity, code, message, offset=None):
    result.issues.append(Issue(severity, code, message, offset))

# ── File existence / size ─────────────────────────────────────────────────
try:
    stat = path.stat()
except OSError as e:
    issue(SEVERITY_ERROR, "FILE_UNREADABLE", str(e))
    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result

result.file_size = stat.st_size

if stat.st_size == 0:
    issue(SEVERITY_ERROR, "EMPTY_FILE", "File is 0 bytes")
    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result

if stat.st_size > MAX_FILE_SIZE:
    issue(SEVERITY_WARNING, "HUGE_FILE", f"File exceeds {MAX_FILE_SIZE // (1024**3)} GB — proceeding cautiously")

# ── Load file ─────────────────────────────────────────────────────────────
try:
    data = path.read_bytes()
except OSError as e:
    issue(SEVERITY_ERROR, "READ_ERROR", str(e))
    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result

r = BinaryReader(data)

# ── Magic ─────────────────────────────────────────────────────────────────
try:
    magic = r.read(4)
except EOFError as e:
    issue(SEVERITY_ERROR, "TRUNCATED_MAGIC", str(e))
    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result

if magic != GGUF_MAGIC:
    issue(SEVERITY_ERROR, "BAD_MAGIC",
          f"Expected {GGUF_MAGIC!r}, got {magic!r}. Not a GGUF file (or badly corrupted).")
    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result

# ── Version ───────────────────────────────────────────────────────────────
try:
    version = r.u32()
except EOFError as e:
    issue(SEVERITY_ERROR, "TRUNCATED_VERSION", str(e))
    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result

result.version = version
if not (GGUF_VERSION_MIN <= version <= GGUF_VERSION_MAX):
    issue(SEVERITY_ERROR, "UNSUPPORTED_VERSION",
          f"Version {version} is outside known range [{GGUF_VERSION_MIN}–{GGUF_VERSION_MAX}]. "
          "File may be from a future format or corrupted.")

# ── Tensor / KV counts ────────────────────────────────────────────────────
try:
    tensor_count = r.u64()
    kv_count     = r.u64()
except EOFError as e:
    issue(SEVERITY_ERROR, "TRUNCATED_HEADER", str(e))
    result.elapsed_ms = (time.monotonic() - t0) * 1000
    return result

result.tensor_count = tensor_count
result.kv_count = kv_count

if tensor_count > 1_000_000:
    issue(SEVERITY_WARNING, "IMPLAUSIBLE_TENSOR_COUNT",
          f"tensor_count={tensor_count} is suspiciously large")
if kv_count > 100_000:
    issue(SEVERITY_WARNING, "IMPLAUSIBLE_KV_COUNT",
          f"kv_count={kv_count} is suspiciously large")

# ── KV metadata ───────────────────────────────────────────────────────────
def read_value(r: BinaryReader, vtype: int, depth=0):
    """Read a single metadata value. Returns True on success."""
    try:
        vt = GGUFValueType(vtype)
    except ValueError:
        raise ValueError(f"Unknown value type {vtype} at offset {r.pos}")

    if vt == GGUFValueType.STRING:
        r.string()
    elif vt == GGUFValueType.ARRAY:
        elem_type = r.u32()
        count = r.u64()
        if count > 10_000_000:
            raise ValueError(f"Array length {count} is implausibly large at offset {r.pos}")
        for _ in range(count):
            read_value(r, elem_type, depth + 1)
    elif vt == GGUFValueType.BOOL:
        r.bool_()
    elif vt in VALUE_SIZE:
        r.read(VALUE_SIZE[vt])
    else:
        raise ValueError(f"Unhandled value type {vt}")

kv_errors = 0
for i in range(kv_count):
    offset_before = r.pos
    try:
        key = r.string()
        vtype = r.u32()
        read_value(r, vtype)
    except (EOFError, ValueError, UnicodeDecodeError) as e:
        issue(SEVERITY_ERROR, "KV_PARSE_FAILURE",
              f"KV entry {i} parse failed at offset {offset_before}: {e}",
              offset=offset_before)
        kv_errors += 1
        if kv_errors >= 3:
            issue(SEVERITY_ERROR, "KV_PARSE_ABORTED",
                  "Too many KV parse failures — aborting metadata scan")
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            return result

# ── Tensor info ───────────────────────────────────────────────────────────
tensor_infos = []   # (name, n_dims, shape, ggml_type, offset)
tensor_errors = 0

for i in range(tensor_count):
    offset_before = r.pos
    try:
        name   = r.string()
        n_dims = r.u32()
        if n_dims > 8:
            issue(SEVERITY_WARNING, "HIGH_TENSOR_DIMS",
                  f"Tensor '{name}' has {n_dims} dimensions (expected ≤ 8)",
                  offset=offset_before)
        shape = tuple(r.u64() for _ in range(n_dims))
        ggml_type = r.u32()
        t_offset  = r.u64()
        tensor_infos.append((name, n_dims, shape, ggml_type, t_offset))
    except (EOFError, ValueError) as e:
        issue(SEVERITY_ERROR, "TENSOR_INFO_PARSE_FAILURE",
              f"Tensor info {i} parse failed at offset {offset_before}: {e}",
              offset=offset_before)
        tensor_errors += 1
        if tensor_errors >= 3:
            issue(SEVERITY_ERROR, "TENSOR_INFO_ABORTED",
                  "Too many tensor info parse failures — aborting tensor scan")
            result.elapsed_ms = (time.monotonic() - t0) * 1000
            return result

# ── Alignment / data section ──────────────────────────────────────────────
# GGUF v2+ aligns tensor data to alignment (default 32, configurable via metadata)
alignment = 32  # default; may be overridden by general.alignment key
# (We don't re-parse KV here; this is fine for structural checks.)

data_start = r.pos
# Round up to alignment
if data_start % alignment != 0:
    data_start += alignment - (data_start % alignment)

# ── Validate tensor data bounds + optional stat check ────────────────────
for name, n_dims, shape, ggml_type, t_offset in tensor_infos:
    abs_offset = data_start + t_offset

    if abs_offset > stat.st_size:
        issue(SEVERITY_ERROR, "TENSOR_OFFSET_OOB",
              f"Tensor '{name}': data offset {abs_offset} exceeds file size {stat.st_size}",
              offset=abs_offset)
        continue

    # Compute expected tensor byte size
    n_elements = 1
    for dim in shape:
        n_elements *= dim

    expected_bytes: Optional[int] = None
    n_blocks: Optional[int] = None
    if ggml_type in GGML_BLOCK_INFO:
        block_elems, block_bytes = GGML_BLOCK_INFO[ggml_type]
        if n_elements % block_elems != 0:
            issue(SEVERITY_WARNING, "TENSOR_SHAPE_MISALIGNED",
                  f"Tensor '{name}': {n_elements} elements not divisible by block size {block_elems}")
        else:
            n_blocks = n_elements // block_elems
            expected_bytes = n_blocks * block_bytes
    elif ggml_type in GGML_TYPE_SIZE and GGML_TYPE_SIZE[ggml_type] is not None:
        expected_bytes = n_elements * GGML_TYPE_SIZE[ggml_type]
    else:
        issue(SEVERITY_WARNING, "UNKNOWN_GGML_TYPE",
              f"Tensor '{name}': unknown ggml_type={ggml_type}, skipping size check")

    truncated = False
    if expected_bytes is not None:
        end_offset = abs_offset + expected_bytes
        if end_offset > stat.st_size:
            issue(SEVERITY_ERROR, "TENSOR_DATA_TRUNCATED",
                  f"Tensor '{name}': data ends at {end_offset}, "
                  f"but file is only {stat.st_size} bytes "
                  f"(short by {end_offset - stat.st_size} bytes)",
                  offset=abs_offset)
            truncated = True

    # ── Tier 1: fast scale-header scan (all blocks, header bytes only) ───
    if stat_check and not truncated and n_blocks is not None and n_blocks > 0:
        ts = stat_check_tensor(
            data, name, ggml_type, abs_offset, n_blocks, full_scan=False)
        result.tensor_stats.append(ts)
        emit_stat_issues(ts, issue)

# ── Tier 2: sampled stat scan (--stat-scan flag) ──────────────────────────
# Upgrades Tier-1 results to full_scan=True. We replace each tensor's
# TensorStatResult in-place rather than clearing and re-appending, so
# Tier-1 issues already emitted are not duplicated.
if stat_scan:
    tier1_by_name = {ts.name: i for i, ts in enumerate(result.tensor_stats)}

    for name, n_dims, shape, ggml_type, t_offset in tensor_infos:
        if ggml_type not in GGML_BLOCK_INFO:
            continue
        abs_offset = data_start + t_offset
        n_elements = 1
        for dim in shape:
            n_elements *= dim
        block_elems, block_bytes_t2 = GGML_BLOCK_INFO[ggml_type]
        if n_elements % block_elems != 0:
            continue
        n_blocks = n_elements // block_elems
        if abs_offset + n_blocks * block_bytes_t2 > stat.st_size:
            continue   # truncated — already flagged

        ts = stat_check_tensor(
            data, name, ggml_type, abs_offset, n_blocks, full_scan=True)

        if name in tier1_by_name:
            # Replace Tier-1 result; issues for this tensor were already
            # emitted so only emit the Tier-2-specific ones now.
            result.tensor_stats[tier1_by_name[name]] = ts
            # Emit only Tier-2 extras (zero-block ratio, identical runs)
            if ts.zero_block_ratio > ZERO_BLOCK_RATIO_THRESHOLD:
                issue(SEVERITY_WARNING, "HIGH_ZERO_BLOCK_RATIO",
                      f"Tensor '{ts.name}': {ts.zero_block_ratio:.1%} of sampled "
                      f"blocks are all-zero (threshold {ZERO_BLOCK_RATIO_THRESHOLD:.0%})")
            if ts.max_identical_run >= IDENTICAL_RUN_THRESHOLD:
                issue(SEVERITY_WARNING, "IDENTICAL_BLOCK_RUN",
                      f"Tensor '{ts.name}': run of {ts.max_identical_run} consecutive "
                      f"identical blocks detected — possible memcpy / write corruption")
        else:
            result.tensor_stats.append(ts)
            emit_stat_issues(ts, issue)

# ── Deep scan: SHA-256 of full file ───────────────────────────────────────
if deep:
    h = hashlib.sha256(data)
    result.deep_checksum = h.hexdigest()

# ── Final verdict ─────────────────────────────────────────────────────────
result.ok = not any(i.severity == SEVERITY_ERROR for i in result.issues)
result.elapsed_ms = (time.monotonic() - t0) * 1000
return result
```

# ── Output formatting ─────────────────────────────────────────────────────────

# ANSI colors (disabled automatically on non-TTY / –no-color)

class C:
RED    = “\033[91m”
YELLOW = “\033[93m”
GREEN  = “\033[92m”
CYAN   = “\033[96m”
GREY   = “\033[90m”
BOLD   = “\033[1m”
RESET  = “\033[0m”

def colorize(text: str, color: str, use_color: bool) -> str:
return f”{color}{text}{C.RESET}” if use_color else text

def format_bytes(n: int) -> str:
for unit in (“B”, “KB”, “MB”, “GB”, “TB”):
if n < 1024:
return f”{n:.1f} {unit}”
n /= 1024
return f”{n:.1f} PB”

def print_result(result: ScanResult, use_color: bool):
ok_str = (
colorize(“✓ OK”,      C.GREEN,  use_color) if result.ok
else colorize(“✗ CORRUPT”, C.RED, use_color)
)
name = Path(result.path).name
size_str = colorize(format_bytes(result.file_size), C.GREY, use_color)
elapsed  = colorize(f”{result.elapsed_ms:.0f}ms”, C.GREY, use_color)

```
print(f"{colorize(name, C.BOLD, use_color)}  {ok_str}  {size_str}  {elapsed}")

if result.version is not None:
    meta = []
    meta.append(f"v{result.version}")
    if result.tensor_count is not None:
        meta.append(f"{result.tensor_count} tensors")
    if result.kv_count is not None:
        meta.append(f"{result.kv_count} KV entries")
    print(f"  {colorize('  '.join(meta), C.GREY, use_color)}")

for iss in result.issues:
    if iss.severity == SEVERITY_ERROR:
        prefix = colorize("  [ERROR]  ", C.RED, use_color)
    elif iss.severity == SEVERITY_WARNING:
        prefix = colorize("  [WARN]   ", C.YELLOW, use_color)
    else:
        prefix = colorize("  [INFO]   ", C.CYAN, use_color)

    loc = f" (offset {iss.offset})" if iss.offset is not None else ""
    print(f"{prefix}{iss.code}: {iss.message}{colorize(loc, C.GREY, use_color)}")

if result.deep_checksum:
    print(f"  {colorize('SHA-256:', C.GREY, use_color)} {result.deep_checksum}")
```

# ── Path collection ───────────────────────────────────────────────────────────

def collect_paths(targets: list[str]) -> list[Path]:
paths = []
for t in targets:
p = Path(t)
if p.is_dir():
paths.extend(sorted(p.rglob(”*.gguf”)))
elif p.exists():
paths.append(p)
else:
print(f”Warning: {t!r} not found — skipping”, file=sys.stderr)
return paths

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
parser = argparse.ArgumentParser(
prog=“gguf-scan”,
description=“Scan GGUF files for structural and quantization corruption.”,
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog=”””
Examples:
gguf_scan.py model.gguf
gguf_scan.py ./models/
gguf_scan.py *.gguf –json
gguf_scan.py model.gguf –deep
gguf_scan.py model.gguf –stat-check
gguf_scan.py model.gguf –stat-scan
gguf_scan.py model.gguf –warnings-as-errors
“””,
)
parser.add_argument(“targets”, nargs=”+”, help=“GGUF file(s) or director(ies) to scan”)
parser.add_argument(”–json”,  action=“store_true”, help=“Output JSON instead of human-readable text”)
parser.add_argument(”–deep”,  action=“store_true”, help=“Compute SHA-256 checksum of each file”)
parser.add_argument(”–stat-check”, action=“store_true”,
help=“Tier 1: scan every block’s scale header for NaN/Inf/zero/out-of-range values”)
parser.add_argument(”–stat-scan”, action=“store_true”,
help=“Tier 2: sampled stat scan — also checks zero-block ratio and identical-block runs “
“(implies –stat-check)”)
parser.add_argument(”–no-color”, action=“store_true”, help=“Disable ANSI color output”)
parser.add_argument(”–warnings-as-errors”, action=“store_true”,
help=“Treat warnings as errors for exit code purposes”)
parser.add_argument(”–quiet”, “-q”, action=“store_true”,
help=“Only print files with issues”)
args = parser.parse_args()

```
use_color = not args.no_color and sys.stdout.isatty()
paths = collect_paths(args.targets)

do_stat_check = args.stat_check or args.stat_scan
do_stat_scan  = args.stat_scan

if not paths:
    print("No GGUF files found.", file=sys.stderr)
    sys.exit(2)

results = []
any_error = False

for path in paths:
    result = scan_file(path, deep=args.deep, stat_check=do_stat_check, stat_scan=do_stat_scan)

    if args.warnings_as_errors:
        effective_ok = not result.issues  # any issue = fail
    else:
        effective_ok = result.ok

    if not effective_ok:
        any_error = True

    results.append(result)

    if not args.json:
        if args.quiet and effective_ok:
            continue
        print_result(result, use_color)
        if len(paths) > 1:
            print()

if args.json:
    print(json.dumps([r.to_dict() for r in results], indent=2))

if not args.json and len(paths) > 1:
    n_ok   = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    total_size = sum(r.file_size for r in results)
    summary = (
        f"\n{colorize('Summary:', C.BOLD, use_color)} "
        f"{colorize(str(n_ok), C.GREEN, use_color)} OK, "
        f"{colorize(str(n_fail), C.RED if n_fail else C.GREY, use_color)} failed  "
        f"— {format_bytes(total_size)} total"
    )
    print(summary)

sys.exit(1 if any_error else 0)
```

if **name** == “**main**”:
main()
