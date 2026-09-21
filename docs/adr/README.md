# Architecture Decision Records

One file per decision that was not obvious and would otherwise be re-litigated later.
Each records the context, the options considered, what was chosen, and what it costs.

| ADR | Decision | Status |
|---|---|---|
| [0001](./0001-shared-database-and-models.md) | Services share one database and one set of ORM models | Accepted |
| [0002](./0002-uuidv7-identifiers.md) | UUIDv7 for job and voice ids | Accepted |
| [0003](./0003-websocket-event-contract.md) | Discriminated union + monotonic `seq` for job events | Accepted |
| [0004](./0004-grpc-only-between-worker-and-engine.md) | gRPC kept, but only on the worker-to-engine link | Accepted |
