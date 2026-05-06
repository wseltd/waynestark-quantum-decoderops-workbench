# Expected Metric Fixtures

Golden-reference metric values for seeded benchmark runs. Used by
`tests/integration` to compare actual decoder outputs against the committed
expected numbers.

Tolerances (`ler_rel_tolerance`, `latency_rel_tolerance`) are applied per
comparison; latency has a wider tolerance because it is hardware-dependent.

These fixtures are NOT updated during a run — regenerate them deliberately
with a seeded one-off benchmark when the upstream library versions change.
