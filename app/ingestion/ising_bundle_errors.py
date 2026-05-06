"""Exception types raised while parsing NVIDIA Ising-Decoding bundles.

This module owns the single error type used by the Ising-Decoding bundle
parser. It is intentionally tiny and dependency-free so that lightweight
ingestion components (and their tests) can import it without pulling in
torch, vendor checkpoint code, or the bundle parser itself.

Why a separate errors module: the parser will ultimately wire in vendor
checkpoint loading and SHA256 verification against ``.decoderops/ising_assets.json``.
Keeping the exception declaration in its own module avoids forcing those
heavy dependencies on every caller that only needs to *catch* the error.
"""

from __future__ import annotations


class InvalidIsingBundleError(Exception):
    """Raised when an Ising-Decoding bundle fails validation.

    Conditions that produce this error include:

      * required keys (``variant``, ``hparams``) missing from the bundle
      * ``variant`` is not one of the supported values (``fast``, ``accurate``)
      * the receptive_field declared in the bundle does not match the
        canonical value for the variant (9 for ``fast``, 13 for ``accurate``)
      * the referenced vendor checkpoint file does not exist on disk
      * the checkpoint's SHA256 does not match the recorded hash in
        ``.decoderops/ising_assets.json``
      * the bundle file suffix is not one of the supported formats
        (``.json``, ``.yaml``, ``.yml``)

    The error message must name the offending input and, where possible,
    the absolute path involved so the caller can act on it without
    re-deriving the failure context.
    """
