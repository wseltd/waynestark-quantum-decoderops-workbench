"""Capability report schema for decoder and runtime adapters.

Every decoder adapter's ``available()`` method and the unified
capability detector in ``app.core.capability`` return an instance of
:class:`CapabilityReport`. Downstream report pipelines (Deployment-
Readiness Report, Compatibility Matrix, Risk Register) consume these
records without running live probes themselves.

Design notes (why this shape, not another):
    * Frozen Pydantic v2 model. The Risk Register must be stable once
      emitted; allowing mutation after construction has caused silent
      divergence between on-disk manifests and in-memory views in other
      projects.
    * ``blocker_category`` is a ``Literal`` of seven explicit values.
      ``not_installed`` is kept distinct from ``software`` on purpose:
      "package missing" is actionable by ``pip install``, while
      "software present but misbehaving" demands a different remediation
      path in the Risk Register. Collapsing them would erase that signal.
    * ``required`` is always populated. Even when ``available=True`` we
      list the enforced preconditions so the deployment report can show
      the positive reason (e.g. "tensorrt 10.16.1.11 loaded and cu13
      stack present") next to the exact packages that were checked.
    * No probing logic lives here. This is pure schema. The live probes
      (torch import, tensorrt import, cudaq target enumeration, etc.)
      live in ``app/core/capability.py`` so that unit tests of this
      schema never need a GPU or vendor libraries in scope.

Rejected alternatives:
    * ``@dataclass(frozen=True, slots=True)``: loses boundary input
      validation (rejecting unknown blocker categories is the whole
      point of this schema) and forces hand-rolled ``model_dump`` for
      the reports layer.
    * A string ``blocker_category`` with free-form values: would silently
      accept typos like ``'not-installed'`` and break Risk Register
      joins. The Literal enforces the closed set.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

# Closed set of Risk Register categories. Keep this aligned with
# app/reports when the report templates are built — any change here is
# a contract change for the deployment-readiness pipeline.
BlockerCategory = Literal[
    "none",
    "machine",
    "software",
    "licensing",
    "runtime",
    "not_installed",
    "version_mismatch",
]

# Sentinel for the "nothing is blocking" state. Used by ``ready()`` so
# the string lives in exactly one place and cannot drift from the
# Literal above.
_NO_BLOCKER: Literal["none"] = "none"


class CapabilityReport(BaseModel):
    """Outcome of a single capability check (decoder backend or runtime probe).

    Attributes:
        available: Whether the capability is usable right now.
        reason: Human-readable explanation, required in both states.
            When available, describe what was detected (e.g. "tensorrt
            10.16.1.11 loaded and cu13 stack present"). When unavailable,
            describe the precise blocker.
        required: Packages, libraries, drivers, or hardware that must be
            present for this capability. Non-empty even when available;
            the list is the enforced precondition set.
        blocker_category: Risk Register bucket. ``'none'`` when available.
            The seven-value Literal is the contract with the downstream
            report layer.
        detected_versions: Observed versions of the required components.
            Empty when nothing was probed successfully.
        probe_latency_ms: Wall time of the live probe in milliseconds.
            ``None`` when no probe ran (e.g. fast-path ImportError check).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    available: bool
    reason: str = Field(..., min_length=1)
    required: list[str] = Field(..., min_length=1)
    blocker_category: BlockerCategory
    detected_versions: dict[str, str] = Field(default_factory=dict)
    probe_latency_ms: float | None = None

    def __repr__(self) -> str:
        # Explicit __repr__ so log lines and pytest failure output are
        # compact and predictable; Pydantic's default repr expands every
        # field including large version dicts, which is noisy when these
        # objects appear in Risk Register assertions.
        return (
            f"CapabilityReport(available={self.available!r}, "
            f"blocker_category={self.blocker_category!r}, "
            f"reason={self.reason!r})"
        )

    @property
    def is_available(self) -> bool:
        """Alias for :attr:`available`.

        The decoder backends' ``available()`` contract (see T022/T023)
        pins the attribute name as ``is_available`` in several ticket
        verify_cmds. Both spellings resolve to the same underlying
        field; the alias prevents call-site churn across the decoder,
        capability-detector, and report layers.
        """
        return self.available

    @classmethod
    def unavailable(
        cls,
        reason: str,
        required: list[str],
        category: BlockerCategory,
    ) -> CapabilityReport:
        """Construct a report for a capability that is not usable.

        Args:
            reason: Precise human-readable explanation of the blocker.
            required: Packages / libraries / hardware that were checked.
                Must be non-empty; an empty list would mean "nothing was
                actually verified", which is a bug at the call site.
            category: Risk Register bucket. Must be a
                :data:`BlockerCategory` value other than ``'none'``.

        Returns:
            A frozen CapabilityReport with ``available=False``.

        Raises:
            ValueError: If ``category`` is ``'none'`` — an unavailable
                capability with no blocker category would silently drop
                out of the Risk Register.
        """
        if category == _NO_BLOCKER:
            raise ValueError(
                "CapabilityReport.unavailable: blocker category 'none' is "
                "reserved for available capabilities; pass a specific "
                "category such as 'not_installed' or 'version_mismatch'."
            )
        return cls(
            available=False,
            reason=reason,
            required=required,
            blocker_category=category,
            detected_versions={},
            probe_latency_ms=None,
        )

    @classmethod
    def ready(
        cls,
        reason: str,
        required: list[str],
        detected_versions: dict[str, str],
        probe_latency_ms: float | None = None,
    ) -> CapabilityReport:
        """Construct a report for a capability that is usable.

        Args:
            reason: Human-readable confirmation (e.g. "torch 2.11.0+cu130
                imported, 2 CUDA devices visible").
            required: The precondition set that was verified. Non-empty —
                "ready with no preconditions" would mean nothing was
                actually checked.
            detected_versions: Observed versions keyed by package or
                library name. May be empty when the probe does not
                report versions (e.g. pure hardware checks).
            probe_latency_ms: Wall time of the live probe, if one ran.

        Returns:
            A frozen CapabilityReport with ``available=True`` and
            ``blocker_category='none'``.
        """
        return cls(
            available=True,
            reason=reason,
            required=required,
            blocker_category=_NO_BLOCKER,
            detected_versions=detected_versions,
            probe_latency_ms=probe_latency_ms,
        )
