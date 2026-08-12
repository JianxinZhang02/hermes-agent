# Memory Provider 基准测试三阶段设计

## 1. 目标

将 Memory Provider 的评估拆分为三个相互隔离、可单独恢复的阶段：

```text
历史对话
  → 阶段 1：Memory Build
  → 固化并校验 Memory Baseline
  → 阶段 2：Read-only QA
  → 保存模型预测
  → 阶段 3：Judge
  → 生成准确率和诊断指标
```

这种设计适用于 TencentDB Agent Memory，也可作为 OpenViking、Mem0 等
Memory Provider 的后续基准测试改造参考。

## 2. 阶段 1：Memory Build

输入选中 `conv` 的全部历史 Session，通过真实 Hermes 和目标 Memory
Provider 完成：

- 原始对话写入；
- Memory 提取、聚合和索引；
- Session end/commit/flush；
- 等待异步任务完成；
- 检查每个作用域的写入覆盖率。

每个 `conv` 应有独立的逻辑作用域，例如：

```text
team/workspace + agent_id + user_id
```

Build 完成后生成 `memory_build_manifest.json`，至少记录：

- 数据集及其哈希；
- 已构建的 conv 和 Memory scope；
- 预期与实际 Session 数量；
- Provider、Embedding 和索引配置；
- 写入结果清单的哈希；
- Memory Baseline 状态。

Manifest 是基线一致性证明；物理数据仍由 Provider 自己保存。需要进行严格
Provider 对比时，可另外复制或快照实际数据目录。

## 3. 阶段 2：Read-only QA

QA 必须只检索阶段 1 已构建的 Memory，不能把问题和模型答案重新写入正式
Memory。

每一道问题应使用独立 Session：

```text
qa-<conv>-q1
qa-<conv>-q2
qa-<conv>-q3
```

每次 QA 的基本流程为：

```text
Question
  → 使用当前 conv 的 Memory scope 检索
  → 将相关 Memory 注入 Hermes 上下文
  → 模型生成预测答案
  → 保存检索记录、预测答案、token 和延迟
  → 验证 QA Session 没有写入长期 Memory
```

建议明确检查：

- QA 请求使用 `store=false` 或 Provider 对应的只读模式；
- QA Session 的新增 Memory/L0 消息数量为零；
- 不触发 session commit、consolidation 或 profile 更新；
- 不继承上一道 QA 的消息历史；
- 不检索其他 conv 的作用域。

完整 Memory Baseline 可以服务于较小的 QA 子集。例如先构建全部 10 个
conv，之后只测试一个 conv 的 10 道问题。但被测试 conv 的历史 Memory 必须
完整，且 QA 结果文件只能包含本次选中的问题。

## 4. 阶段 3：Judge

Judge 只读取已经保存的：

```text
Question + Prediction + Reference Answer + Judge Rule
```

Judge 阶段不应：

- 启动或查询 Memory Provider；
- 启动 Hermes Agent；
- 接收检索到的 Memory；
- 接收完整历史对话；
- 将评分结果写回 Memory。

进入 Judge 前应停止 Memory Gateway/Server，从运行环境上保证评分与 Memory
检索隔离。

## 5. 推荐产物结构

```text
<run-dir>/
├── run_metadata.json
├── runtime/
│   └── provider-data/
└── results/
    ├── memory_build_manifest.json
    ├── import_success.csv
    ├── qa_results.csv
    ├── summary.json
    └── stats.log
```

建议在 `qa_results.csv` 中保存：

- conv、question ID 和独立 QA Session ID；
- 预测答案和标准答案；
- 实际检索结果及 provenance；
- 检索数量和延迟；
- 输入、输出及缓存 token；
- QA 期间的 Memory 写入数量；
- Judge 标签与理由。

## 6. OpenViking 后续改造原则

OpenViking 对应实验可复用同一思想，但应通过 OpenViking 自身接口实现，而
不是复制 TencentDB 的 L0/L1/L2/L3 或存储结构：

1. Build 阶段完成 OpenViking 的历史写入、commit 和索引等待；
2. 为每个 conv 分配独立的 user/agent/session namespace；
3. Build 后记录 OpenViking 配置、namespace、资源数量和数据快照信息；
4. QA 阶段只执行 recall/search/context 注入；
5. 每道 QA 使用独立 Session，并验证没有新增长期记忆；
6. Judge 前关闭 OpenViking Client/Server 或禁止其参与评分；
7. keyword、vector、hybrid、Top-K 等对比必须复用同一 Memory Baseline；
8. 不同 Provider 的 Judge 模型、Prompt、问题集合和评分规则保持一致。

## 7. 核心原则

```text
Build 决定“沉淀了什么”
QA 决定“检索并回答了什么”
Judge 决定“答案是否正确”
```

三者只有通过显式产物传递数据，不通过隐式 Session、后台写入或共享模型上下文
传递状态。这样才能定位错误来自 Memory 构建、检索、Hermes 回答还是 Judge，
并保证不同 Memory Provider 的实验结果可复现、可比较。
