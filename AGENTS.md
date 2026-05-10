# QIK agent guidance

- Correctness is paramount. Do not add silent fallback behavior that hides malformed inputs, missing calibration state, failed optional integrations, or degraded model paths.
- Prefer explicit validation errors over best-effort degradation. If a caller wants a lower-fidelity mode, require an explicit parameter that names that tradeoff.
- Keep sampling, timestamp alignment, and candle construction behind shared helpers instead of duplicating parallel `if`/fallback trees in each model.
- When fixing realism models, preserve the fit-once/generate-many contract: fitted profiles must carry the state needed for generation without requiring callers to duplicate `model_params`.
