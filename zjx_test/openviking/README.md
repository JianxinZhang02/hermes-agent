# OpenViking live provider experiment

This directory contains an opt-in smoke test for the real Hermes OpenViking
provider. It is not part of the default pytest suite and does not copy or
simulate the provider implementation.

## What it verifies

The script imports `plugins.memory.openviking.OpenVikingMemoryProvider` from
the current checkout and exercises this real path:

```text
Hermes OpenVikingMemoryProvider
  -> OpenViking HTTP server
  -> viking_remember (write and enqueue indexing)
  -> viking_read (exact-content verification)
  -> viking_search (eventual index verification)
  -> viking_forget (exact cleanup)
  -> provider shutdown
```

It creates a uniquely named memory and deletes it by default. It does not call
the session commit/extraction path because that path can generate multiple
derived memories that cannot be deterministically identified and cleaned up by
a smoke test.

OpenViking may return a canonical URI containing the trusted-mode user, for
example `viking://user/default/peers/...`, even when Hermes submitted
`viking://user/peers/...`. The smoke test treats these as the same record when
their complete peer-relative path matches and prints the canonical URI.

## Server setup

In terminal A:

```bash
cd /dfs/data/zjx
source hermes_env/bin/activate
openviking-server
```

In terminal B:

```bash
cd /dfs/data/zjx
source hermes_env/bin/activate
cd hermes-agent
python zjx_test/openviking/smoke_provider.py
```

The endpoint defaults to `http://127.0.0.1:1933`. Override it when necessary:

```bash
python zjx_test/openviking/smoke_provider.py \
  --endpoint http://127.0.0.1:1933 \
  --search-timeout 120
```

For a server with authentication enabled, export the credential instead of
putting it in the command line or source code:

```bash
export OPENVIKING_API_KEY='your-key'
python zjx_test/openviking/smoke_provider.py
```

Existing `OPENVIKING_ACCOUNT`, `OPENVIKING_USER`, and `OPENVIKING_AGENT`
environment settings are honored by the Hermes provider.

## Diagnostic options

If the OpenViking server can store and read content but its embedding/index
backend is not ready, isolate that issue with:

```bash
python zjx_test/openviking/smoke_provider.py --skip-search
```

To retain the generated memory for manual inspection:

```bash
python zjx_test/openviking/smoke_provider.py --keep
```

The script prints the exact `viking://` URI. Delete retained test data after
inspection with Hermes' `viking_forget` tool or OpenViking's filesystem API.

## Multi-session Agent lifecycle experiment

`smoke_agent_lifecycle.py` exercises the MemoryManager integration used by
Hermes rather than calling `viking_remember` directly. It runs two completed
turns in Session A, uses Hermes' in-process `/new` lifecycle to commit and
switch to Session B, recalls the extracted facts, adds and commits a third
fact, then starts a fresh Manager for Session C and verifies all three facts
plus Hermes' actual `<memory-context>` injection format.

The conversation uses meaningful project decisions rather than opaque test
labels: a unique project name, MySQL 8.4 with `utf8mb4`, production port 3306,
and a daily backup policy. Assertions check durable business values (`MySQL`,
`utf8mb4`, and `3306`) that a summarizing VLM should preserve.

```bash
python zjx_test/openviking/smoke_agent_lifecycle.py
```

The server must report both `Embedding: PASS` and `VLM: PASS` first:

```bash
openviking-server doctor
```

VLM extraction is asynchronous. The default wait is four minutes per recall
boundary; it can be increased for a slow model endpoint:

```bash
python zjx_test/openviking/smoke_agent_lifecycle.py \
  --extraction-timeout 600 \
  --poll-interval 10
```

Every run uses unique `OPENVIKING_USER` and `OPENVIKING_AGENT` identities. The
script attempts to recursively delete only those unique namespaces and exact
test sessions at the end. Use `--keep` to retain them for inspection.

Assistant messages in this experiment are deterministic strings. The real
Hermes MemoryManager, background synchronization, OpenViking server, VLM
extraction, embedding, retrieval, and API-bound memory-context composition are
used; no chat model is called to decide or generate the assistant response.
