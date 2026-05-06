"""Exception hierarchy for the Ising fast decoder asset loader.

These errors distinguish the precise reason an Ising-Decoding checkpoint
bundle cannot be used: a missing manifest, a malformed manifest, a
manifest entry missing for the requested model, a missing model file on
disk, or a SHA256 mismatch against the recorded digest. Callers use the
specific subclass to populate a capability-report reason string that is
actionable for an operator.

Pure stdlib on purpose. This module is imported early by capability
detection and the decoder's ``available()`` path; pulling torch in here
would defeat the point of guarded imports elsewhere in the package.
"""

from __future__ import annotations

import pathlib


class IsingAssetError(Exception):
    """Base class for all Ising-Decoding asset-loader failures.

    Catch this to handle any Ising bundle problem generically; catch a
    subclass to react to a specific failure mode.
    """


class IsingManifestMissingError(IsingAssetError):
    """The asset manifest JSON file was not found at the expected path."""

    def __init__(self, *, manifest_path: pathlib.Path) -> None:
        self.manifest_path = manifest_path
        super().__init__(f"Ising asset manifest not found at {manifest_path}")


class IsingManifestMalformedError(IsingAssetError):
    """The manifest file exists but could not be parsed as valid JSON.

    The underlying parser exception (typically ``json.JSONDecodeError``)
    is preserved on ``__cause__`` so that traceback output shows the
    original line/column of the parse failure even when callers
    construct the wrapper directly rather than via ``raise ... from``.
    """

    def __init__(
        self, *, manifest_path: pathlib.Path, original: Exception
    ) -> None:
        self.manifest_path = manifest_path
        self.original = original
        super().__init__(
            f"Ising asset manifest at {manifest_path} is malformed: {original}"
        )
        # Explicit chaining so forensic tracebacks survive even when the
        # caller constructs the wrapper outside of an ``except`` block
        # (for example when re-wrapping an exception read off a future).
        self.__cause__ = original


class IsingManifestEntryMissingError(IsingAssetError):
    """The manifest parsed cleanly but has no entry for the requested model."""

    def __init__(
        self, *, manifest_path: pathlib.Path, model_filename: str
    ) -> None:
        self.manifest_path = manifest_path
        self.model_filename = model_filename
        super().__init__(
            f"Ising asset manifest at {manifest_path} has no entry for "
            f"model filename {model_filename!r}"
        )


class IsingModelFileMissingError(IsingAssetError):
    """The model checkpoint file was not found on disk."""

    def __init__(self, *, model_path: pathlib.Path) -> None:
        self.model_path = model_path
        super().__init__(f"Ising model file not found at {model_path}")


class IsingAssetIntegrityError(IsingAssetError):
    """The model file's SHA256 digest does not match the manifest.

    Both digests are embedded verbatim (no abbreviation) in the message.
    Abbreviating hex digests would hide exactly which bytes differ, and
    SHA256 drift is the kind of failure operators need to diff
    byte-for-byte when investigating supply-chain or storage issues.
    """

    def __init__(
        self,
        *,
        model_path: pathlib.Path,
        expected_sha256: str,
        actual_sha256: str,
    ) -> None:
        self.model_path = model_path
        self.expected_sha256 = expected_sha256
        self.actual_sha256 = actual_sha256
        super().__init__(
            f"SHA256 mismatch for {model_path}: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
