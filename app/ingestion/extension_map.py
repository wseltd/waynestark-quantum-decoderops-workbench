"""Canonical extension-to-parser-key map for ingestion inputs.

The dispatcher consults this map first; only when an extension is missing
or ambiguous does it fall back to :mod:`app.ingestion.signature_sniff`.
Keeping the map here (not inlined in the dispatcher) keeps the vocabulary
in one place and avoids extension-string drift between the CLI and API
ingest paths.

This module is pure: no filesystem access, no parsing, no logging.
"""

from __future__ import annotations

from typing import Final

from app.ingestion.signature_sniff import TAG_DEM, TAG_JSON, TAG_NPY, TAG_STIM

# Parser keys local to this module. ``bin`` and ``jsonl`` have no stable
# byte-level signature the sniffer can rely on, so they exist only as
# extension-driven routing targets — which is why they are defined here
# and not alongside the sniff tags.
_PARSER_KEY_BIN: Final[str] = "bin"
_PARSER_KEY_JSONL: Final[str] = "jsonl"

# Canonical lookup. Keys are lowercased extensions WITH a leading dot,
# matching the shape of :attr:`pathlib.Path.suffix`. Keeping the dot in
# the key avoids an ambiguity where ``"json"`` and ``"son"`` would both
# need to be handled as user input.
_EXTENSION_TO_PARSER_KEY: Final[dict[str, str]] = {
    ".stim": TAG_STIM,
    ".dem": TAG_DEM,
    ".npy": TAG_NPY,
    ".bin": _PARSER_KEY_BIN,
    ".jsonl": _PARSER_KEY_JSONL,
    ".json": TAG_JSON,
}

# Public, immutable view of the supported extensions. Exposed as a
# frozenset so callers can use ``ext in SUPPORTED_EXTENSIONS`` without
# receiving a mutable container they could silently corrupt.
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(_EXTENSION_TO_PARSER_KEY)


def extension_to_parser_key(ext: str) -> str | None:
    """Return the parser key for a filename extension, or ``None``.

    The input is normalised by lowercasing and, if no leading dot is
    present, prepending one. Trimming of surrounding whitespace is
    deliberately NOT performed: that is a caller concern, and silently
    accepting ``" .stim"`` would hide upstream bugs.

    Args:
        ext: Extension such as ``".stim"``, ``"stim"``, ``".STIM"``, or
            ``"STIM"``. Empty strings and a bare ``"."`` return ``None``.

    Returns:
        One of ``"stim"``, ``"dem"``, ``"npy"``, ``"bin"``, ``"jsonl"``,
        ``"json"`` when the extension is supported; otherwise ``None``.
    """
    if not ext:
        return None
    normalised = ext.lower()
    if not normalised.startswith("."):
        normalised = "." + normalised
    return _EXTENSION_TO_PARSER_KEY.get(normalised)
