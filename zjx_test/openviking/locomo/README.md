# Hermes Native vs OpenViking LoCoMo reproduction

This directory reproduces the official OpenViking Hermes LoCoMo protocol while
keeping all experiment code outside the Hermes core. It contains pinned
official benchmark scripts for OpenViking `v0.3.22` and `v0.4.12`, and
automatically selects the scripts matching the installed server package. The
two arms are:

- `native`: Hermes with `memory.provider: ""`;
- `e2e`: the same Hermes configuration with `memory.provider: openviking`.

Hermes built-in memory remains present in the e2e arm because that is how the
official Hermes+OpenViking benchmark is defined. The answer model may differ
from the published report, but both arms always use the same selected model.

## What is pinned

- The six benchmark scripts are copied without algorithm changes from
  `volcengine/OpenViking` tags `v0.3.22` and `v0.4.12`. The matching set is
  selected from the installed `openviking` package version.
- `source_manifest.json` stores a SHA256 for every vendored file. Every run
  verifies these hashes before starting.
- Between these two official versions, LoCoMo processing, answer judging, and
  statistics are unchanged. The `v0.4.12` E2E scripts remove the obsolete
  `X-OpenViking-Agent` header.
- LoCoMo `locomo10.json` is downloaded from commit
  `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` and verified as
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Category 5 is excluded and categories 1–4 are judged by the unmodified
  official judge code at temperature 0.

See `ATTRIBUTION.md` for the AGPL-3.0 and CC BY-NC 4.0 notices.

## Server preparation

Run from the server checkout with the existing sibling virtual environment:

```bash
cd /dfs/data/zjx
source hermes_env/bin/activate
cd hermes-agent

# Only needed once. Editable mode means later source updates are used directly.
pip install -e ".[dev]"

# The runner selects matching official scripts for 0.3.22 or 0.4.12.
python -c "import importlib.metadata as m; print(m.version('openviking'))"
```

OpenViking `0.4.12` is a supported strict-reproduction version and needs no
extra flag. A future or otherwise unpinned version is rejected by default. The
`--allow-openviking-version-mismatch` flag deliberately falls back to the
`v0.3.22` script protocol and marks that run as a non-strict compatibility run.

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

## Recommended three-stage run

Download and verify LoCoMo:

```bash
python zjx_test/openviking/locomo/prepare_dataset.py
```

Probe the editable Hermes import, real Hermes answer model, independent judge,
dedicated OpenViking server, and Hermes OpenViking provider:

```bash
python zjx_test/openviking/locomo/run_benchmark.py preflight
```

Build all 10 LoCoMo conversations exactly once with one command. Each conv gets
its own physical Native `state.db` and OpenViking workspace, so memories cannot
leak between dataset samples. This command deliberately has no `--count`
argument: the number of later questions is not part of the Memory Baseline.

```bash
python zjx_test/openviking/locomo/run_benchmark.py build \
  --run-id locomo10-memory-v1 \
  --import-parallel 1
```

The completed `memory_build_collection_manifest.json` references 10 child
`memory_build_manifest.json` files. Every child fingerprints both durable arms:

- native: the imported Hermes `state.db` session/message content;
- e2e: OpenViking extracted Markdown memory plus committed import sessions.

Re-running the same `build` command validates and reuses all completed
baselines; it does not replay their historical conversations. If an interrupted
collection has only built some convs, the next run reuses completed children
and continues the remaining convs. Use a new build run id when changing the
dataset, answer model, OpenViking/VLM/embedding configuration, or provider
version.

Run 1, 10, or all scored QA questions per conv against the same 10 baselines.
Give each question selection its own `qa-id`:

```bash
python zjx_test/openviking/locomo/run_benchmark.py qa \
  --run-id locomo10-memory-v1 --qa-id q1-per-conv --count 1

python zjx_test/openviking/locomo/run_benchmark.py qa \
  --run-id locomo10-memory-v1 --qa-id q10-per-conv --count 10

python zjx_test/openviking/locomo/run_benchmark.py qa \
  --run-id locomo10-memory-v1 --qa-id qall
```

Because `--count` follows the official evaluator's per-sample semantics, these
commands evaluate up to 10 questions (`1 × 10 convs`), up to 100 questions
(`10 × 10 convs`), and all 1,540 scored Category 1–4 questions respectively.

QA starts Hermes (and OpenViking for the e2e arm), but never runs the import
scripts. Every question uses an independent session and the official
`store:false` request. After both arms finish, the runner verifies that the
saved Memory Baseline fingerprints are unchanged.

Judge any saved QA set as a separate final stage:

```bash
python zjx_test/openviking/locomo/run_benchmark.py judge \
  --run-id locomo10-memory-v1 --qa-id q10-per-conv
```

Judge reads only the saved question, prediction, reference answer and official
judge rule. It does not start Hermes Gateway or OpenViking Server. The same
Memory Baseline can therefore support any number of independent QA/Judge runs.

For a cheap single-conv debugging build, add `--sample`:

```bash
python zjx_test/openviking/locomo/run_benchmark.py build \
  --run-id conv26-memory-debug --sample 0 --import-parallel 1
```

The corresponding `qa` and `judge` commands automatically recognize that this
run contains only one conv. You never need to invoke ten single-conv commands
manually for the full dataset.

For compatibility, the original one-command workflow remains available:

```bash
python zjx_test/openviking/locomo/run_benchmark.py pair --sample 0 --count 5
```

`pair` is the legacy all-in-one official flow and keeps its original resume
semantics. Use `build` + `qa` + `judge` when memory must be built once and reused.

## Output

Code and the downloaded dataset stay under `zjx_test/openviking/locomo`, while
runtime state and results default to the repository's sibling directory:

```text
/dfs/data/zjx/hermes-locomo-runs/<run-id>/
├── memory_build_collection_manifest.json
├── conv-builds/
│   ├── sample-0/
│   │   ├── memory_build_manifest.json
│   │   ├── native/hermes-home/state.db
│   │   └── e2e/openviking-workspace/
│   ├── sample-1/
│   └── ... sample-9/
└── evaluations/<qa-id>/
    ├── qa_manifest.json
    ├── judge_manifest.json
    ├── native/qa_results.csv
    ├── e2e/qa_results.csv
    └── comparison/
        ├── accuracy_comparison.csv
        └── summary.md
```

The three manifests record source revisions, hashes, model identifiers,
question selection and stage status. Tokens and API keys are redacted. Before
Judge, the runner requires both arms to contain exactly the same
`sample_id + question index`, question, gold answer and category.

## Offline tests

```bash
scripts/run_tests.sh zjx_test/openviking/locomo/tests
```

These tests do not call any external model or service. The real-service checks
are intentionally kept in the explicit `preflight`, `build`, `qa`, `judge`,
and compatibility `pair` commands.
