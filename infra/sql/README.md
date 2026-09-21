# infra/sql

| File | When it runs | What it does |
|---|---|---|
| `roles.sql` | Once per environment, at provisioning | Creates the per-service database roles and column-level grants required by ADR-0001 |

Schema changes are Alembic migrations under `services/gateway/alembic`, applied by an
admin role at deploy time. These roles are the runtime credentials, and they deliberately
cannot issue DDL.

After adding a column that a worker must write, add it to that worker's `GRANT UPDATE`
list here and re-run the file. The grant list failing loudly is the point: privileges get
reviewed rather than inherited.
