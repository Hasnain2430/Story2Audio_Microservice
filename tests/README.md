# tests

| Directory | Runs in CI | What it covers |
|---|---|---|
| `unit/` | yes, on every push | Pure logic: schemas, prompt assembly, segmentation, state transitions. No network, no Docker. |
| `integration/` | on demand (`-m integration`) | Against the compose stack: real Redis, Postgres, MinIO. |
| `e2e/` | against preview deploys | Playwright. The happy path, refresh-mid-job, and the WebSocket-fails-falls-back-to-polling path. |
| `fixtures/` | — | Shared test data, including prompts ported from the v1 `TestCases.json`. |
