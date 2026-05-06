"""Byte-signature sniffing for ingestion inputs.

This module is a pure, side-effect-free classifier. It never performs file
I/O, never parses structured content, and never raises. Callers use the
returned tag to choose a parser; the tag is a heuristic, not a guarantee.

The detection window is bounded to the first 512 bytes. That bound exists
because real Stim / DEM / NumPy inputs reveal their identity in the header,
and scanning further wastes time on large payloads.
"""

from __future__ import annotations

from typing import Final

# Maximum number of leading bytes inspected. Bounds sniff cost on large
# inputs and matches the contract documented in the module docstring.
MAX_SNIFF_BYTES: Final[int] = 512

# NumPy's on-disk .npy format begins with the literal magic b"\x93NUMPY"
# (see numpy/lib/format.py). Any other prefix rules out .npy.
NPY_MAGIC: Final[bytes] = b"\x93NUMPY"

# Stim circuit files use a compact, keyword-driven text format. These
# keywords are reserved tokens emitted by stim.Circuit.__repr__ and by
# stim.Circuit.to_file; they do not collide with DEM syntax. We check
# several keywords rather than one because small fragments of a larger
# circuit may omit any single keyword.
STIM_MARKERS: Final[tuple[bytes, ...]] = (
    b"QUBIT_COORDS",
    b"DETECTOR",
    b"OBSERVABLE_INCLUDE",
)

# Stim DetectorErrorModel text uses lowercase function-call syntax
# (``error(p) ...``, ``detector(...) ...``). Matching on the trailing
# "(" rules out accidental matches inside comments or identifiers.
DEM_MARKERS: Final[tuple[bytes, ...]] = (
    b"error(",
    b"detector(",
)

# JSON / JSONL inputs may be preceded by any ASCII whitespace. The set
# mirrors Python's ``bytes.isspace`` so callers get the intuitive result
# for files produced by common tools.
JSON_WHITESPACE: Final[bytes] = b" \t\n\r\v\f"

# Stable tag vocabulary. Exposed as constants so callers can branch on
# identity rather than typo-prone string literals.
TAG_STIM: Final[str] = "stim"
TAG_DEM: Final[str] = "dem"
TAG_NPY: Final[str] = "npy"
TAG_JSON: Final[str] = "json"
TAG_UNKNOWN: Final[str] = "unknown"


def sniff_signature(buf: bytes) -> str:
    """Classify the leading bytes of ``buf`` as one of five input kinds.

    The function inspects at most :data:`MAX_SNIFF_BYTES` bytes. Detection
    order is: ``npy`` (binary magic), ``stim`` (text keywords), ``dem``
    (text function-call markers), ``json`` (first non-whitespace byte is
    ``{`` or ``[``). If nothing matches, ``unknown`` is returned.

    Args:
        buf: Raw bytes from the start of an input. Shorter inputs are
            handled the same way as longer ones — the scan simply sees
            fewer bytes.

    Returns:
        One of the literals ``"stim"``, ``"dem"``, ``"npy"``, ``"json"``,
        or ``"unknown"``. The return is a heuristic tag, not a promise
        that the payload parses cleanly.
    """
    # Slice once. A ``bytes`` slice is cheap and gives the rest of the
    # function a single view to reason about, which makes the 512-byte
    # bound auditable from any single line.
    head = buf[:MAX_SNIFF_BYTES]

    # NumPy magic is checked first because it is the only binary marker
    # and its prefix cannot legitimately appear at byte 0 of text inputs.
    if head.startswith(NPY_MAGIC):
        return TAG_NPY

    # Stim before DEM: Stim circuits can contain the substring "error"
    # in comments, so DEM's lowercase ``error(`` is a weaker signal than
    # Stim's uppercase keywords. Checking Stim first avoids misclassifying
    # annotated circuits as DEM.
    for marker in STIM_MARKERS:
        if marker in head:
            return TAG_STIM

    for marker in DEM_MARKERS:
        if marker in head:
            return TAG_DEM

    # JSON / JSONL: skip ASCII whitespace and inspect the first payload
    # byte. ``lstrip`` allocates a new bytes object, but the input is
    # bounded to 512 bytes so the cost is negligible.
    stripped = head.lstrip(JSON_WHITESPACE)
    if stripped and stripped[:1] in (b"{", b"["):
        return TAG_JSON

    return TAG_UNKNOWN
