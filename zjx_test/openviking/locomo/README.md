# Hermes Native vs OpenViking LoCoMo reproduction

This directory reproduces the official OpenViking `v0.3.22` Hermes LoCoMo
protocol while keeping all experiment code outside the Hermes core. The two
arms are:

- `native`: Hermes with `memory.provider: ""`;
- `e2e`: the same Hermes configuration with `memory.provider: openviking`.

Hermes built-in memory remains present in the e2e arm because that is how the
official Hermes+OpenViking benchmark is defined. The answer model may differ
from the published report, but both arms always use the same selected model.

## What is pinned

- The six benchmark scripts are copied without algorithm changes from
  `volcengine/OpenViking` tag `v0.3.22`, commit
  `18897e46d272de4653352666d877348ee95f2dcd`.
- `source_manifest.json` stores a SHA256 for every vendored file. Every run
  verifies these hashes before starting.
- LoCoMo `locomo10.json` is downloaded from commit
  `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` and verified as
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Category 5 is excluded and categories 1–4 are judged by the unmodified
  official judge code at temperature 0.

See `ATTRIBUTION.md` for the Apache-2.0 and CC BY-NC 4.0 notices.

## Server preparation

Run from the server checkout with the existing sibling virtual environment:

```bash
cd /dfs/data/zjx
source hermes_env/bin/activate
cd hermes-agent

# Only needed once. Editable mode means later source updates are used directly.
pip install -e ".[dev]"

# Strict mode checks this exact OpenViking package version.
python -c "import importlib.metadata as m; print(m.version('openviking'))"
```

The experiment uses the model already selected in the base Hermes config,
normally `/root/.hermes/config.yaml`. Configure the desired Hermes answer model
there before creating a new run. The runner snapshots only `config.yaml` into
clean experiment homes; it does not copy `state.db`, `MEMORY.md`, `USER.md`,
sessions, or the base `.env` file.

Configure an independent OpenAI-compatible judge in the base Hermes `.env` or
in the shell environment:

```text
JUDGE_BASE_URL=https://your-judge-endpoint/v1
JUDGE_TOKEN=your-secret-token
JUDGE_MODEL=your-judge-model
```

The existing `/root/.openviking/ov.conf` supplies embedding and VLM settings.
The runner copies it into the external run directory, changes only
`storage.workspace`, and starts a dedicated service on `127.0.0.1:1934`.
It never touches the existing service or data on port 1933.

## Run

Download and verify LoCoMo:

```bash
python zjx_test/openviking/locomo/prepare_dataset.py
```

Probe the editable Hermes import, real Hermes answer model, independent judge,
dedicated OpenViking server, and Hermes OpenViking provider:

```bash
python zjx_test/openviking/locomo/run_benchmark.py preflight
```

Run a low-cost aligned experiment first:

```bash
python zjx_test/openviking/locomo/run_benchmark.py pair \
  --sample 0 \
  --count 5
```

Then run one complete LoCoMo conversation or the full dataset:

```bash
python zjx_test/openviking/locomo/run_benchmark.py pair --sample 0
python zjx_test/openviking/locomo/run_benchmark.py pair
```

The command prints a run id. Resume an interrupted run without changing its
sample, count, model, judge, or concurrency settings:

```bash
python zjx_test/openviking/locomo/run_benchmark.py pair \
  --run-id locomo-pair-YYYYMMDD-HHMMSS-abcdef \
  --sample 0 \
  --count 5
```

Do not use `--force-ingest` against a non-empty e2e workspace. Deterministic
official Session IDs could reuse old extracted memory, so the runner refuses
that combination. Use a new run id for a fresh strict run.

## Output

Code and the downloaded dataset stay under `zjx_test/openviking/locomo`, while
runtime state and results default to the repository's sibling directory:

```text
/dfs/data/zjx/hermes-locomo-runs/<run-id>/
├── native/hermes-home/
├── native/results/
├── e2e/hermes-home/
├── e2e/openviking-workspace/
├── e2e/results/
└── comparison/
    ├── run_manifest.json
    ├── accuracy_comparison.csv
    └── summary.md
```

`run_manifest.json` records source revisions, hashes, model identifiers,
parallelism and paths. Tokens and API keys are redacted. Before producing the
comparison, the runner requires both arms to contain exactly the same
`sample_id + question index`, question, gold answer and category.

## Offline tests

```bash
scripts/run_tests.sh zjx_test/openviking/locomo/tests
```

These tests do not call any external model or service. The real-service checks
are intentionally kept in the explicit `preflight` and `pair` commands.
