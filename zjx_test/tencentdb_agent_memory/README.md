# TencentDB Agent Memory experiments

This directory contains reproducible experiments for Hermes' built-in
`memory_tencentdb` provider. It is intentionally outside the formal test suite
so it can also be copied to a server and run by hand.

## Credential-free contract smoke

```bash
python zjx_test/tencentdb_agent_memory/smoke_provider.py
```

The script starts an ephemeral HTTP server on `127.0.0.1` that implements the
official TencentDB Agent Memory `/v3` contract used by Hermes. It then exercises
the real plugin loader, `MemoryManager`, provider, HTTP client, session switch,
cross-session recall, L1 user/project isolation, L2/L3 tools, exact L1
update/delete, and shutdown.

Expected final line:

```text
PASS: offline Hermes/TencentDB Agent Memory provider lifecycle succeeded.
```

This proves the Hermes integration contract without an API key. It does not
run the upstream TypeScript extraction pipeline and therefore does not measure
LLM extraction quality, embedding quality, or Tencent VectorDB behavior.

## Real Gateway L0 smoke (no LLM key required)

With the official Gateway already running:

```bash
python zjx_test/tencentdb_agent_memory/smoke_live_gateway.py \
  --endpoint http://127.0.0.1:8420
```

This uses the real Gateway implementation and a unique team/agent/user/session
scope. It verifies health, Hermes L0 capture, `/v3/conversation/query`, BM25 L0
search without a session filter, negative user/team isolation, session
rotation, exact test-message cleanup, and external-process ownership. It does
not assert L1/L2/L3 because those require the Gateway-side LLM.

## Live Gateway follow-up

After preparing the official Gateway, write its endpoint to the active Hermes
profile:

```json
{
  "endpoint": "http://127.0.0.1:8420",
  "auto_start": false,
  "service_id": "default"
}
```

Save this as `$HERMES_HOME/memory_tencentdb.json`, configure the Gateway's own
LLM/storage settings, and activate the provider:

```bash
hermes config set memory.provider memory_tencentdb
hermes memory status
```

The live acceptance criterion is: a Session A turn appears in L0, the official
asynchronous pipeline produces L1/L2/L3 artifacts, and Session B recalls the L1
fact under the same `(team_id, agent_id, user_id)` scope.

Upstream intentionally scopes L2/L3 at `(team_id, agent_id)`, so those profile
layers are shared across users in the same team; the smoke's negative user
check applies to the L1 marker, not to L2/L3.

## Real multi-layer and restart-persistence experiment

Against an isolated official Gateway, run:

```bash
python zjx_test/tencentdb_agent_memory/smoke_live_multilayer.py \
  --endpoint http://127.0.0.1:8420 \
  --gateway-data-dir /path/to/disposable-tdai-data
```

This verifies real L0 multi-turn/multi-session capture, L0 scope isolation,
official L2 and L3 API persistence, Hermes memory-tool routing, the exact
`<memory-context>` fence sent to the model, and external Gateway ownership.
L3 is seeded through the official v3 API. The current official
`scenario/write` endpoint is update-only because the LLM pipeline normally
creates L2 files, so the script first bootstraps one empty L2 file in the
disposable local `TDAI_DATA_DIR`, then performs the real versioned write,
index, read, list, count, and delete through v3. This checks the integration
independently of LLM variability. The script reports the real L1 count but
never fabricates an L1 record.

To prove storage survives a Gateway restart:

```bash
python zjx_test/tencentdb_agent_memory/smoke_live_multilayer.py \
  --endpoint http://127.0.0.1:8420 \
  --gateway-data-dir /path/to/disposable-tdai-data --keep

# Restart the Gateway with the same TDAI_DATA_DIR, then run:
python zjx_test/tencentdb_agent_memory/smoke_live_multilayer.py --verify-only
```

Use a disposable Gateway data directory for experiments. L0 and L2 are cleaned
after the normal/verify run, but the official v3 data plane currently exposes
no L3 delete endpoint; deleting the disposable data directory removes that
synthetic profile as well.

With the target Gateway stopped, verify non-fatal degradation:

```bash
python zjx_test/tencentdb_agent_memory/smoke_unavailable_fallback.py \
  --endpoint http://127.0.0.1:8420
```
