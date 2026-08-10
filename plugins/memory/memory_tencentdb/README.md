# TencentDB Agent Memory Provider

Hermes `MemoryProvider` adapter for the official
[TencentDB Agent Memory](https://github.com/Tencent/TencentDB-Agent-Memory)
Gateway.

The Python code in this directory is deliberately an adapter, not a fork of
the TencentDB engine. The official Node.js Gateway owns extraction, retrieval,
storage, scheduling, and the L0–L3 artifacts. Hermes owns Agent lifecycle and
scope propagation.

## Architecture

```text
Hermes Agent (Python)
└── MemoryManager
    └── MemoryTencentdbProvider
        ├── config.py                 $HERMES_HOME/memory_tencentdb.json
        ├── client.py                 official /v3 HTTP contract
        └── supervisor.py             optional local Gateway lifecycle
                 │
                 ▼ HTTP
        TencentDB Agent Memory Gateway (Node.js, default :8420)
        ├── L0 conversation evidence
        ├── L1 atomic/episodic memories
        ├── L2 scenario Markdown
        ├── L3 user core/persona
        └── SQLite/sqlite-vec or Tencent VectorDB backend
```

Nothing under `agent/` knows about TencentDB. Selecting another
`memory.provider` replaces this adapter without changing the Agent Loop.

## What is implemented

| Hermes boundary | TencentDB operation |
|---|---|
| `initialize()` | Connect to an existing Gateway or supervise a configured local sidecar |
| `prefetch()` | Parallel L1 search + L2 scene listing + L3 core read |
| `sync_turn()` | `POST /v3/conversation/add` (L0) on a bounded background path |
| `on_session_switch()` | Rotate the session ID used by subsequent L0 writes |
| `on_session_end()` | Drain pending writes, then `POST /session/end` to flush only that session's L1 buffer |
| `on_memory_write()` | Mirror successful built-in add/replace writes into L0 |
| `shutdown()` | Drain writes and stop only a Gateway process started by this provider |
| `backup_paths()` | Advertise local Gateway data for `hermes backup` |

Reliability behavior includes bounded write threads, a five-failure circuit
breaker, watchdog recovery, single-flight sidecar startup, request timeouts,
structured Gateway errors, and empty-context fallback when the Gateway is
unavailable.

## Native memory layers

- **L0**: immutable conversation evidence. Every completed Hermes turn enters
  here with `(team_id, agent_id, user_id, session_id)`.
- **L1**: structured atomic memories extracted by TencentDB's pipeline. Search,
  exact update, and exact delete are supported. Chat types are `persona`,
  `episodic`, and `instruction`; current upstream also accepts `work_fact`,
  `work_task`, `work_method`, and `work_artifact`.
- **L2**: scenario files. Recall lists available scenes and the model can read
  a selected scene.
- **L3**: synthesized user core/persona. Recall injects it and a tool can read
  it explicitly.

The adapter does not invent a direct L1-create operation because the official
v3 API has none. `memory_tencentdb_remember` writes L0 immediately; the native
pipeline promotes it to L1/L2/L3 asynchronously.

## Tools

Tools exist only while `memory.provider: memory_tencentdb` is selected.

| Tool | Layer | Purpose |
|---|---:|---|
| `memory_tencentdb_memory_search` | L1 | Semantic/hybrid search with IDs and provenance |
| `memory_tencentdb_conversation_search` | L0 | Search raw conversation evidence across sessions |
| `memory_tencentdb_remember` | L0 → pipeline | Submit an explicit durable memory |
| `memory_tencentdb_update` | L1 | Update one exact atomic memory ID |
| `memory_tencentdb_forget` | L1 | Delete exact atomic memory IDs |
| `memory_tencentdb_read_scene` | L2 | Read one scenario Markdown file |
| `memory_tencentdb_profile` | L3 | Read the synthesized user core/persona |

Tool schemas and the provider system-prompt block are static for a
conversation. A slow-starting or temporarily unavailable Gateway therefore
does not mutate the prompt/tool prefix and break prompt caching.

## Install the official Gateway

This Hermes repository contains only the Python adapter. Clone and prepare the
official engine separately:

```bash
git clone https://github.com/Tencent/TencentDB-Agent-Memory.git
cd TencentDB-Agent-Memory/MemoryCore
pnpm install
```

TencentDB's current package requires Node.js 22.16 or newer. Start the Gateway
yourself:

```bash
cd /absolute/path/TencentDB-Agent-Memory/MemoryCore
pnpm exec tsx src/gateway/server.ts
```

Or put an equivalent command in `gateway_cmd` so Hermes supervises it. Running
the Gateway separately is recommended in production; Hermes never stops a
process it did not start.

The Gateway's default standalone storage is local SQLite/sqlite-vec. Tencent
VectorDB is an optional Gateway-side backend; the Hermes adapter is unchanged.

## Hermes configuration

Activate interactively:

```bash
hermes memory setup
```

The setup wizard asks only for the Gateway endpoint and optional client
Bearer token. Advanced sidecar, scope, timeout, and storage settings belong in
the JSON file below so routine setup is not a long sequence of optional
prompts.

Or activate directly:

```bash
hermes config set memory.provider memory_tencentdb
```

Non-secret settings live in:

```text
$HERMES_HOME/memory_tencentdb.json
```

Minimal configuration for an already-running Gateway:

```json
{
  "endpoint": "http://127.0.0.1:8420",
  "auto_start": false,
  "service_id": "default"
}
```

Managed local sidecar example:

```json
{
  "endpoint": "http://127.0.0.1:8420",
  "gateway_cmd": "sh -c 'cd /opt/TencentDB-Agent-Memory/MemoryCore && exec pnpm exec tsx src/gateway/server.ts'",
  "gateway_config": "/opt/TencentDB-Agent-Memory/MemoryCore/tdai-gateway.standalone.yaml",
  "data_dir": "/srv/tencentdb-agent-memory",
  "auto_start": true,
  "recall_limit": 5,
  "request_timeout": 3.0,
  "write_timeout": 15.0,
  "session_flush_timeout": 30.0
}
```

Supported JSON fields:

| Field | Default | Meaning |
|---|---|---|
| `endpoint` | `http://127.0.0.1:8420` | Existing or managed Gateway URL |
| `gateway_cmd` | empty | Optional sidecar start command |
| `gateway_config` | empty | Passed to the child as `TDAI_GATEWAY_CONFIG` |
| `data_dir` | Gateway default | Passed to the child as `TDAI_DATA_DIR` |
| `service_id` | `default` | `x-tdai-service-id` memory-space header |
| `team_id` | Hermes workspace | Optional fixed tenant/team scope |
| `agent_id` | active Hermes profile | Optional fixed agent scope |
| `user_id` | platform user | Optional fixed user scope |
| `auto_start` | `true` | Permit local auto-discovery/startup |
| `recall_limit` | `5` | L1 results per automatic recall, clamped to 1–20 |
| `request_timeout` | `3.0` | Recall/read request seconds, clamped to 0.2–30 |
| `write_timeout` | `15.0` | Background L0 write seconds, clamped to 0.5–60; separate so cold Gateway initialization does not create false write failures |
| `session_flush_timeout` | `30.0` | Session L1 flush seconds, clamped to 1–300 |
| `llm_base_url` | Gateway default | Managed-sidecar LLM endpoint |
| `llm_model` | Gateway default | Managed-sidecar LLM model |

Secrets belong in `$HERMES_HOME/.env`:

```dotenv
# Only when Gateway authentication is enabled
MEMORY_TENCENTDB_GATEWAY_API_KEY=replace-me

# Only when Hermes starts the standalone Gateway and it needs an LLM
TDAI_LLM_API_KEY=replace-me
```

For an externally managed Gateway, configure its LLM, embedding, SQLite/TCVDB,
pipeline cadence, and authentication on the Gateway itself. Hermes does not
copy remote service configuration into its process.

Compatibility reads remain for upstream/older deployments using
`TDAI_MEMORY_ENDPOINT`, `TDAI_MEMORY_API_KEY`, `TDAI_MEMORY_SERVICE_ID`,
`MEMORY_TENCENTDB_GATEWAY_HOST`, `MEMORY_TENCENTDB_GATEWAY_PORT`, and
`MEMORY_TENCENTDB_GATEWAY_CMD`.

## Scope mapping

| TencentDB v3 field | Hermes source | Fallback |
|---|---|---|
| `team_id` | configured `team_id`, else `agent_workspace` | `default` |
| `agent_id` | configured `agent_id`, else active profile | `default` |
| `user_id` | configured `user_id`, else gateway/platform user | `default` |
| `session_id` | current Hermes session | `default` only for malformed callers |
| `task_id` | current `ProviderScope.task_id`, when present | omitted |

The upstream layer scopes are not identical:

- L0 writes retain team/agent/user/session/task.
- L1 reads and writes retain team/agent/user/task; automatic search omits only the
  session filter so a new session can recall earlier durable memories.
- L2/L3 are deliberately aggregated by team/agent and ignore user/session/task.
  They are shared profile layers for the team's agent, not private per-user
  files. Use separate team IDs if users must not share L2/L3.

At a real Hermes session boundary, pending L0 writes are drained before the
provider calls the upstream legacy-shaped `POST /session/end` route. Despite
its unversioned path, this is the current Gateway's scoped flush API: it
cancels that session's idle timer, processes residual L1 work, and leaves all
other sessions and shared services running.

Hermes `cron`, `flush`, and `subagent` execution contexts are read-only for
this provider. They may recall existing memory, but automatic capture and the
three mutating tools are rejected so synthetic/system activity cannot pollute
the primary user's long-term memory.

## Tests

Formal tests do not need TencentDB credentials or an external service:

```bash
scripts/run_tests.sh tests/plugins/memory/test_memory_tencentdb_provider.py
```

Credential-free HTTP/lifecycle smoke:

```bash
python zjx_test/tencentdb_agent_memory/smoke_provider.py
```

The smoke server implements the official v3 contract but does not run the
real TencentDB extraction model.

Real Gateway L0 smoke (no LLM credential required):

```bash
python zjx_test/tencentdb_agent_memory/smoke_live_gateway.py \
  --endpoint http://127.0.0.1:8420
```

This verifies real L0 persistence, official query/search response shapes,
user/team isolation, session rotation, exact cleanup, and that Hermes never
stops an externally managed Gateway. L1/L2/L3 materialization still requires
the Gateway-side LLM and should be assessed separately as an extraction-quality
test.

At upstream revision `fe3230f`, SQLite session-only L0 deletion can return zero
under team isolation because the upstream re-check omits fields from its row
projection. The live smoke safely works around that upstream defect by querying
the isolated session's exact message IDs and deleting with both IDs and session
scope; it never broadens or disables isolation.

## Upstream boundary and licensing

The adapter follows the official v3 endpoints and was initially derived from
the MIT-licensed Hermes adapter shipped in the TencentDB Agent Memory
repository. The heavy TypeScript engine is not vendored here. Upstream API
changes should be handled in `client.py`; Agent Core should remain untouched.
