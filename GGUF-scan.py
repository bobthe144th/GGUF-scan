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
import os
import struct
import sys
import hashlib
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

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
class ScanResult:
path: str
file_size: int
ok: bool
version: Optional[int] = None
tensor_count: Optional[int] = None
kv_count: Optional[int] = None
issues: list = field(default_factory=list)
deep_checksum: Optional[str] = None
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

def scan_file(path: Path, deep: bool = False) -> ScanResult:
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

# ── Validate tensor data bounds ───────────────────────────────────────────
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
    if ggml_type in GGML_BLOCK_INFO:
        block_elems, block_bytes = GGML_BLOCK_INFO[ggml_type]
        if n_elements % block_elems != 0:
            issue(SEVERITY_WARNING, "TENSOR_SHAPE_MISALIGNED",
                  f"Tensor '{name}': {n_elements} elements not divisible by block size {block_elems}")
        else:
            expected_bytes = (n_elements // block_elems) * block_bytes
    elif ggml_type in GGML_TYPE_SIZE and GGML_TYPE_SIZE[ggml_type] is not None:
        expected_bytes = n_elements * GGML_TYPE_SIZE[ggml_type]
    else:
        # Unknown type; skip size check
        issue(SEVERITY_WARNING, "UNKNOWN_GGML_TYPE",
              f"Tensor '{name}': unknown ggml_type={ggml_type}, skipping size check")

    if expected_bytes is not None:
        end_offset = abs_offset + expected_bytes
        if end_offset > stat.st_size:
            issue(SEVERITY_ERROR, "TENSOR_DATA_TRUNCATED",
                  f"Tensor '{name}': data ends at {end_offset}, "
                  f"but file is only {stat.st_size} bytes "
                  f"(short by {end_offset - stat.st_size} bytes)",
                  offset=abs_offset)

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
description=“Scan GGUF files for structural corruption.”,
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog=”””
Examples:
gguf_scan.py model.gguf
gguf_scan.py ./models/
gguf_scan.py *.gguf –json
gguf_scan.py model.gguf –deep
gguf_scan.py model.gguf –warnings-as-errors
“””,
)
parser.add_argument(“targets”, nargs=”+”, help=“GGUF file(s) or director(ies) to scan”)
parser.add_argument(”–json”,  action=“store_true”, help=“Output JSON instead of human-readable text”)
parser.add_argument(”–deep”,  action=“store_true”, help=“Compute SHA-256 checksum of each file”)
parser.add_argument(”–no-color”, action=“store_true”, help=“Disable ANSI color output”)
parser.add_argument(”–warnings-as-errors”, action=“store_true”,
help=“Treat warnings as errors for exit code purposes”)
parser.add_argument(”–quiet”, “-q”, action=“store_true”,
help=“Only print files with issues”)
args = parser.parse_args()

```
use_color = not args.no_color and sys.stdout.isatty()
paths = collect_paths(args.targets)

if not paths:
    print("No GGUF files found.", file=sys.stderr)
    sys.exit(2)

results = []
any_error = False

for path in paths:
    result = scan_file(path, deep=args.deep)

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
