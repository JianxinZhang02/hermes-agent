# Hermes × TAU-2 Airline × OpenViking

这个目录实现一套可上传到 Linux/WSL 服务器执行的外部 benchmark harness。它不修改 Hermes 核心代码，也不会在导入模块时连接 OpenViking。

实验比较两组：

- `no_memory`：由当前源码 Hermes `AIAgent` 初始化模型客户端与请求配置、benchmark Step Adapter 驱动的 TAU-2 Airline Agent，不使用跨任务记忆；
- `openviking`：同一个 Hermes Agent、同一组 Airline 工具，再加入 OpenViking trajectory memory。

两组固定使用相同的 test split、4 个种子 `300..303`、每个种子 20 个任务、相同首条用户消息、相同 User Simulator、`temperature=0` 和最多 200 steps。总计 160 个评测 simulation。

每个 simulation 默认设置 900 秒 wall-clock 上限，每次 Hermes 模型请求默认设置 180 秒超时。TAU-2 checkpoint 会自动恢复；普通重跑不会删除已经完成的 simulation，只有显式 `--force` 才重置该阶段对应的 checkpoint。

## 重要边界

1. 两组都使用 `Hermes-configured external Step Adapter`。它复用当前源码 `AIAgent` 的模型客户端、provider 配置、请求构造、system prompt 与工具 schema，但不调用会内部执行工具的 `run_conversation()`。
2. 工具不是 Hermes 的 `terminal`、`session_search` 等通用工具，而是当前 TAU-2 Airline Environment 绑定的业务工具；这是 TAU-2 正确评测所必需的。
3. Hermes 原生 Session DB、Memory Provider、`MEMORY.md/USER.md` 和上下文文件均关闭。基线是 no-memory，不是 Hermes Native Memory。
4. OpenViking build 只提交 reward=1、DB evaluator 完整的 train trajectory；使用官方 `role_tool_blocks` 和显式 `cases/trajectories/experiences` memory policy，不会把 reward、断言或 test 数据交给记忆提取器。
   Build 会把每条成功提交立即写入 `corpus/commit_progress.json`，失败重跑时跳过已完成项；`corpus_revision` 同时隔离旧版客户端可能留下的半成品 session。
5. eval 不创建或提交 OpenViking Session，只执行 `search/read`。开始与结束会比较 API 级 trajectory snapshot，并审计所有 Hermes trace 中的 OpenViking write count 必须为 0。
6. Hermes 只产生 tool call；TAU-2 Environment 是业务工具唯一执行者。TAU 返回的真实 ToolMessage 再送回 Hermes Step Adapter。写操作前 Top-2 recall 会丢弃未执行候选并重新生成，不存在 speculative execution、rollback 或 replay。
7. OpenViking snapshot 只接受 URI 位于 `/memories/trajectories/` 的可读结果，并保存 URI/内容 hash；`events/entities/profile` 不计入 Agent experience corpus。

## 服务器准备

在 `hermes-agent` 使用的虚拟环境中执行：

```bash
cd /dfs/data/zjx/hermes-agent
source ../hermes_env/bin/activate

bash zjx_test/openviking/tau2_airline/setup_tau2.sh
source zjx_test/openviking/tau2_airline/.env.tau2
```

脚本会把 TAU-2 checkout 放进被忽略的 `.external/`，并固定到官方要求的 `refs/pull/297/head`。建议执行后记录脚本打印的 commit；harness 也会把实际 commit 写入 corpus manifest。

启动已经配置好 Embedding 与 VLM 的 OpenViking Server：

Agent trajectory 还要求实例级 Agent Evolution 总开关。在
`/root/.openviking/ov.conf` 的 `server` 对象内设置：

```json
"agent_evolution": {
  "enabled": true
}
```

若该开关关闭，OpenViking 只会归档 Session，并在 commit 结果中返回
`agent_memory_skip_reason=agent_evolution_disabled`；benchmark 会在第一次提交后立即失败。

```bash
openviking-server
```

另一个终端设置模型凭据。不要把 key 写入配置文件或提交 Git：

```bash
source ../hermes_env/bin/activate
source zjx_test/openviking/tau2_airline/.env.tau2

export HERMES_AGENT_API_KEY='...'
export HERMES_AGENT_BASE_URL='https://api.deepseek.com/v1'
export OPENAI_API_KEY="$HERMES_AGENT_API_KEY"
export OPENAI_API_BASE="$HERMES_AGENT_BASE_URL"
# 服务端启用认证时才需要：
# export OPENVIKING_API_KEY='...'
```

这里 `HERMES_AGENT_*` 供 Hermes 使用；`OPENAI_*` 供 TAU-2 User Simulator 和辅助 evaluator 通过 LiteLLM 使用。

## 分阶段运行

```bash
cd /dfs/data/zjx/hermes-agent/zjx_test/openviking/tau2_airline
RUN_DIR="$PWD/results/airline-step-agent-trajectory-v4"

# 只检查路径与 Python 依赖，不访问本机/服务器 OpenViking
python run_benchmark.py preflight --offline --run-dir "$RUN_DIR"

# 用一次 test bootstrap 固定 20 个场景的首条用户消息
python run_benchmark.py bootstrap --run-dir "$RUN_DIR"

# Hermes 跑 30 个 train task；只把成功 trajectory 提交给 OpenViking
python run_benchmark.py build --run-dir "$RUN_DIR"

# 零 Agent LLM 调用地验证 trajectory URI、正文结构和冻结 hash
python run_benchmark.py audit-memory --run-dir "$RUN_DIR"

# 零模型 Token：task8 的 Step Adapter 只产出调用，TAU-2 在一次性环境执行一次
python run_benchmark.py diagnose-tools --run-dir "$RUN_DIR"

# 只跑 seed300/task8；audit 会在任何 Agent LLM 调用前执行
python run_benchmark.py smoke --run-dir "$RUN_DIR"

# 冻结 corpus，运行 2 arms × 4 seeds × 20 test tasks
python run_benchmark.py eval --run-dir "$RUN_DIR"

# 只读取已有 artifact 统计
python run_benchmark.py report --run-dir "$RUN_DIR"
```

也可以执行 `all`，但分阶段更容易确认 corpus build 与只读 eval 的边界。`--force` 会重新生成对应阶段，请谨慎使用 build 的 `--force`，它会再次向同一 OpenViking scope 提交训练会话。更稳妥的做法是为新实验设置新的 `--openviking-user` 与 `--search-uri`。

若服务端实际 trajectory URI 与默认配置不同，可显式传入：

```bash
python run_benchmark.py build \
  --run-dir "$RUN_DIR" \
  --openviking-account default-hermes-airline-step-agent-trajectory-v4 \
  --openviking-user tau2-airline-hermes-step-agent-trajectory-v4 \
  --search-uri 'viking://user/memories/trajectories'
```

旧 `async-loop-v2` 的 22 个 Session及 Agent Evolution 关闭时生成的 v3 archive
只保留作诊断，不得作为新实验 corpus。`build` 后必须运行 `audit-memory`；只有严格
trajectory URI 与冻结内容 hash 均通过后才能进入 smoke/eval。

## 结果

- `fixed_first_user_fixture.json`：按 scenario SHA 固定首条用户消息；
- `corpus/train_results.json`：Hermes 训练任务原始 TAU-2 结果；
- `corpus/corpus_manifest.json`：成功轨迹、commit task、源码 commit 与冻结指纹；
- `memory_audit_manifest.json`：trajectory/experience URI、内容 hash、结构与只读检查；
- `zero_token_tool_diagnostic.json`：task8 初始 3 座、2 人预订及 TAU 单次执行审计；
- `cells/*.json`：每个 arm/seed 的 20 个 simulation；
- `eval_manifest.json`：覆盖数量、首句 fixture hash、eval 前后 fingerprint、零写审计；
- `scoreboard.json`：accuracy/reward、DB match、tokens、Airline 工具调用、Top-4/Top-2 注入与 paired win/loss/tie。

TAU-2 reward 是正式指标，不使用额外 LLM Judge。`scoreboard.json` 中 token 是 Hermes Agent LLM 的统计，不把 User Simulator 和 OpenViking 内部 VLM/Embedding 消耗混入同一个数字。

## 离线测试

本机没有 OpenViking、也没有安装 TAU-2 时，仍可运行不依赖服务的单元测试：

```bash
scripts/run_tests.sh zjx_test/openviking/tau2_airline/tests/test_offline.py
```

本机仅能证明配置、Step Adapter 状态机、官方 role/tool 编码、strict URI 过滤、secret 过滤和模块语法正确；真实 Airline reward、OpenViking trajectory 产出与检索效果必须到服务器运行后验证。
