# Upstream provenance

This provider adapts the official
[TencentDB Agent Memory](https://github.com/Tencent/TencentDB-Agent-Memory)
project.

- Source snapshot reviewed: `fe3230f176f1bf5832fee79d12494bbc2d19a8aa`
- Upstream adapter path: `MemoryCore/hermes-plugin/memory/memory_tencentdb/`
- Upstream license: MIT

The initial `client.py`, `supervisor.py`, and provider lifecycle structure were
derived from that adapter. The bundled Hermes version adds the current
`MemoryProvider` scope contract, profile-local non-secret configuration,
complete v3 L0–L3 client operations, stable tool schemas, session switching,
built-in-memory mirroring, backup discovery, bounded background work, and
Hermes-native tests.

The TypeScript MemoryCore engine is not vendored. Update API translation in
`client.py` when upstream changes its `/v3` contract; keep Agent Core isolated
from those changes.
