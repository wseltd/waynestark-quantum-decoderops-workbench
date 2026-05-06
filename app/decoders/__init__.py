"""Decoder execution layer — pluggable backends for QEC syndrome decoding.

The layer exposes a structural ``Decoder`` protocol (see
``app.decoders.protocol``) and concrete backends conforming to it:

    * ``pymatching_baseline``   MWPM reference baseline
    * ``ising_fast``            NVIDIA Ising-Decoder-SurfaceCode-1-Fast     (RF=9)
    * ``ising_accurate``        NVIDIA Ising-Decoder-SurfaceCode-1-Accurate (RF=13)
    * ``onnx_validation``       ONNX Runtime validation path
    * ``tensorrt_optional``     TensorRT engine (customer-installed Tier 3)

Design principle: structural typing via ``typing.Protocol`` so third-party
decoders can plug in without inheriting our class tree. Every backend
reports its status via ``app.core.capability_report.CapabilityReport`` so
the unified capability detector (``app.core.capability``) can surface it
in run manifests, the compatibility matrix, and the deployment-readiness
report without running live probes itself.
"""
