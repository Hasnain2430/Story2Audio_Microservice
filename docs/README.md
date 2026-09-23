# Story2Audio — Docs

| Doc | What it covers |
|---|---|
| [01 — System Analysis](./01-system-analysis.md) | The audit of v1: its runtime topology, the ten-minute blocking RPC, twenty catalogued defects, and what was worth keeping |
| [02 — What Changed](./02-what-changed.md) | What v2 does instead, defect by defect, with the numbers it measures at |
| [ADRs](./adr/) | One record per decision that was not obvious: what was chosen, what was rejected, and what it costs |

## Reading order

Start with [02](./02-what-changed.md). It opens with the measured result and walks the
architecture, then answers each of v1's defects in turn.

[01](./01-system-analysis.md) is the evidence base — read it for the receipts behind any
claim about how v1 behaved. The ADRs are for when a design choice in 02 looks arbitrary and
you want the reasoning and the alternatives that were turned down.
