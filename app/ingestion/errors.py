"""Exception hierarchy for the ingestion layer.

This module defines the single base exception for ingestion failures and
one concrete subclass used when the dispatcher cannot determine a supported
parser for an input. It is intentionally dependency-free so lightweight
ingestion components (and their tests) can import it without pulling in
numpy, pydantic, stim, or the dispatcher itself.

Design choice: a small hand-written hierarchy (rather than a @dataclass or
pydantic model) is used because exceptions must be cheaply constructible,
picklable by the standard library, and safe to raise during import-time
validation. A frozen dataclass would have added no value — the fields are
read-only by convention and the class has no behaviour beyond string
formatting.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion-layer failures.

    Carries structured context (``source`` and ``detail``) so callers can
    build user-facing messages and structured logs without re-parsing the
    exception message.

    Attributes:
        source: A short identifier of the input the parser was handling
            (e.g. a filesystem path, a URI, or an in-memory label).
        detail: A human-readable description of what went wrong, written
            so the caller can act on it without inspecting the traceback.
    """

    def __init__(self, source: str, detail: str) -> None:
        """Initialise an ingestion error.

        Args:
            source: Short identifier of the input being parsed.
            detail: Human-readable description of the failure.
        """
        self.source = source
        self.detail = detail
        super().__init__(self._format())

    def _format(self) -> str:
        """Return the canonical one-line rendering of this error."""
        return f"IngestionError(source={self.source!r}, detail={self.detail!r})"

    def __str__(self) -> str:
        """Return the one-line rendering used in tracebacks and logs."""
        return self._format()

    def __repr__(self) -> str:
        """Return an unambiguous developer-facing representation.

        ``__repr__`` and ``__str__`` share one format on purpose: the string
        form is already fully qualified and round-trip-friendly, and having
        two divergent formats tends to cause log/traceback drift.
        """
        return self._format()


class UnsupportedInputError(IngestionError):
    """Raised when the ingestion dispatcher cannot handle a given input.

    Adds the extension and sniffed signature to the base context so the
    deployment-readiness report can explain *why* an input was rejected
    (wrong extension, unknown magic bytes, mismatched combination, …).

    Attributes:
        source: As in :class:`IngestionError`.
        extension: The file extension observed on the input, including the
            leading dot (e.g. ``".dem"``). Empty string when no extension
            is available (for example when dispatching in-memory bytes).
        sniffed_signature: A short label describing the magic-byte
            signature the dispatcher sniffed from the head of the input
            (e.g. ``"npy"``, ``"jsonl"``, ``"unknown"``). Empty string when
            no sniffing was attempted.
        detail: As in :class:`IngestionError`.
    """

    def __init__(
        self,
        source: str,
        extension: str,
        sniffed_signature: str,
        detail: str,
    ) -> None:
        """Initialise an unsupported-input error.

        Args:
            source: Short identifier of the input being parsed.
            extension: File extension observed, including the leading dot.
                Use an empty string when no extension is available.
            sniffed_signature: Label describing the sniffed magic-byte
                signature. Use an empty string when no sniffing was done.
            detail: Human-readable description of the failure.
        """
        self.extension = extension
        self.sniffed_signature = sniffed_signature
        super().__init__(source, detail)

    def _format(self) -> str:
        """Return the canonical one-line rendering of this error."""
        return (
            "UnsupportedInputError("
            f"source={self.source!r}, "
            f"extension={self.extension!r}, "
            f"sniffed_signature={self.sniffed_signature!r}, "
            f"detail={self.detail!r})"
        )

    def __repr__(self) -> str:
        """Return an unambiguous developer-facing representation.

        Defined explicitly (rather than relying on the base class) so static
        analysis and governance checks can see the method on this class.
        """
        return self._format()
