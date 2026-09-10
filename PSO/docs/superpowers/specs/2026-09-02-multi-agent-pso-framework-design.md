# 通用 Multi-Agent PSO 框架设计

> 状态：已批准进入实现规划
>
> 日期：2026-09-02
>
> 首个领域示例：基于 Wiki、MoleculeEditor 与外部光谱计算的红光吸收分子搜索

## 1. 目标

本项目构建一个通用 Python 框架，用粒子群优化（PSO）协调多个独立 Codex Agent 完成可评价任务。每个逻辑粒子拥有独立 Codex thread、工作目录、研究历史、位置、速度和个体最优；所有粒子可以共享用户提供的知识库、skills、工具和计算资源。

框架必须支持以下有界闭环：

```text
基于知识库建立假说
→ 提出结构化工具请求
→ 执行并验证假说
→ 独立 Evaluator 计算指标与 reward
→ 反思预测和结果
→ 更新假说、pbest、social best 与下一轮 position
```

框架核心不得包含分子、SMILES、光谱、TDDFT 或 MoleculeEditor 等领域概念。红光分子任务只作为一个可替换的任务包和端到端示例。

## 2. 第一版范围

第一版包括：

- 通用 `PositionSpace`、PSO 状态、收缩因子更新式和社会拓扑；
- 同步代际调度；
- Orchestrator 驱动的 Agent 显式状态机；
- 基于本地 `openai-codex` Python SDK 的 `LocalCodexRuntime`；
- YAML 配置与类型化 Python 任务插件；
- SQLite 状态存储和文件产物存储；
- 确定性模拟任务、标准连续函数和故障恢复测试；
- 红光吸收分子任务包接口；
- 5 个粒子、5 个 iteration 的可配置真实运行流程。

第一版不包括：

- Pareto 多目标 PSO；
- 异步或陈旧 gbest 更新；
- Web 服务或图形界面；
- PostgreSQL、对象存储或集群级控制面；
- 自动写回 Wiki；
- 框架内置量化化学程序；
- 对找到红光候选或性能提升作结果保证。

### 2.1 本地开发环境

项目使用名为 `multi-agent-pso` 的 Conda 环境和 Python 3.12。`pyproject.toml` 是 Python 包依赖与工具配置的权威来源；Conda 负责解释器和需要的本地二进制依赖。核心运行依赖限定为 NumPy、Pydantic v2、PyYAML 和 `openai-codex`，SQLite 使用 Python 标准库。开发依赖为 pytest、pytest-asyncio 和 Hypothesis。红光分子示例额外需要 RDKit、NetworkX 及用户提供的光谱计算后端。

环境创建和依赖安装属于实现阶段。在设计规格获用户复核前不修改本机 Conda 环境。

## 3. 设计原则

1. Codex thread 是工作记忆，不是状态真值。
2. Agent 只能提出受 Schema 约束的动作；受管工具和计算资源由 Orchestrator 调用。
3. Agent 不得自行声明 reward、pbest 或 gbest。
4. 工具成功、候选合法、计算收敛和性质改善是四个独立事实。
5. 所有随机性、配置、输入、工具结果和状态转移必须可审计。
6. 已完成产物不可原地覆盖；状态更新必须具备事务边界。
7. 同一个 task package 可以替换 Agent runtime、存储和资源后端，而不修改 PSO 核心。
8. 同一个 PSO 核心可以运行完全不同的 task package。

## 4. 总体架构

```text
User CLI / Python API
          │
          ▼
SwarmOrchestrator
├── 同步 iteration 调度
├── Agent 状态机
├── 并发与资源限流
└── 恢复与停止条件
          │
          ├───────────────┐
          ▼               ▼
PSO Core              Task Package
├── PositionSpace     ├── task.yaml
├── ParticleState     ├── TaskAdapter
├── UpdateRule        ├── Evaluator
├── SocialTopology    ├── ToolProvider
└── RNG               └── prompts / skills / schemas
          │               │
          └───────┬───────┘
                  ▼
Ports / Protocols
├── AgentRuntime
├── RunStore
├── ArtifactStore
├── WikiRetriever
└── ResourceManager
                  │
                  ▼
Infrastructure Adapters
├── LocalCodexRuntime
├── SQLiteRunStore
├── FileArtifactStore
├── LocalWikiRetriever
└── LocalProcessResourceManager
```

依赖规则：

- `core/` 不导入 Codex SDK、SQLite、分子库或任务插件；
- Orchestrator 只依赖 Protocol；
- Codex SDK 只出现在 `LocalCodexRuntime`；
- TaskAdapter 不直接修改 PSO 或数据库状态；
- Evaluator 不读取 Agent 隐藏推理；
- 领域代码只存在于任务包；
- 基础设施实现可以依赖核心 Protocol，核心不能反向依赖基础设施。

## 5. 建议项目布局

```text
src/multi_agent_pso/
├── __init__.py
├── cli.py
├── core/
│   ├── models.py
│   ├── position_space.py
│   ├── update_rule.py
│   └── topology.py
├── orchestration/
│   ├── runner.py
│   ├── agent_loop.py
│   ├── iteration.py
│   ├── failure_policy.py
│   └── recovery.py
├── protocols/
│   ├── agent_runtime.py
│   ├── task_adapter.py
│   ├── evaluator.py
│   ├── tools.py
│   ├── storage.py
│   └── resources.py
├── runtimes/
│   └── local_codex.py
├── storage/
│   ├── sqlite_store.py
│   └── file_artifacts.py
└── configuration/
    ├── models.py
    └── loader.py

examples/red_absorption/
├── task.yaml
├── adapter.py
├── evaluator.py
├── prompts/
└── schemas/

tests/
├── core/
├── orchestration/
├── contracts/
├── integration/
└── examples/
```

包通过 `__all__` 暴露小型公共 API。初始公共入口为 `SwarmRunner`、`RunSpec`、`TaskPackage`、`PositionSpace` 和核心状态/评价类型；基础设施内部类不从顶层包导出。

## 6. 核心数据模型

### 6.1 RunSpec

`RunSpec` 是完成解析和校验后的不可变运行配置，至少包含：

- task package 标识和版本；
- population size、iteration 数和 run seed；
- PSO 更新参数和拓扑；
- Agent/工具/Evaluator 并发限制；
- 阶段超时与重试限制；
- thread 轮换策略；
- 存储位置；
- prompt、skill、Wiki、工具和计算协议快照哈希。

### 6.2 ParticleState[P, V]

```text
ParticleState
├── particle_id
├── thread_id
├── thread_generation
├── position: P
├── velocity: V
├── pbest: PersonalBest[P] | null
├── latest_episode_id
├── consecutive_failures
├── rng_state
└── lifecycle_status
```

### 6.3 PersonalBest[P]

```text
PersonalBest
├── evaluated_position: P
├── candidate_reference
├── hypothesis_reference
├── evaluation_reference
├── fitness
└── iteration_id
```

### 6.4 AgentEpisode

`AgentEpisode` 保存一次粒子 iteration 的可审计记录：目标位置、实际位置、位置遵循度、假说、证据引用、工具请求和结果、候选引用、Evaluation、反思、阶段事件、尝试次数、耗时和资源用量。

框架不要求也不依赖 Codex 隐藏推理。只保存结构化消息、工具边界数据、证据和最终解释。

### 6.5 Evaluation

```text
Evaluation
├── status: SUCCESS | INVALID | FAILED | TIMEOUT
├── feasible: bool
├── metrics: JSON-compatible mapping
├── constraints: list[ConstraintResult]
├── fitness: finite float | null
├── uncertainty: JSON-compatible value | null
└── provenance: JSON-compatible mapping
```

只有 `SUCCESS` 可以具有有限 fitness。`INVALID`、`FAILED` 和 `TIMEOUT` 的 fitness 必须为 `null`。

## 7. 可替换 Protocol

### 7.1 PositionSpace[P, V]

任务负责实现：

- `sample_position(rng) -> P`；
- `zero_velocity() -> V`；
- `difference(target, origin) -> V`；
- `scale_velocity(velocity, scalar) -> V`；
- `random_scale(velocity, upper, rng) -> V`；
- `add_velocities(parts) -> V`；
- `clamp_velocity(velocity, fraction) -> V`；
- `advance(position, velocity) -> P`；
- `project(position) -> Projection[P]`；
- `distance(left, right) -> float`；
- position/velocity 的确定性序列化和反序列化。

框架提供 `ContinuousBoxPositionSpace` 作为默认实现。离散或混合任务可以提供自己的类型和运算，不允许核心假设 position 必然是 NumPy 数组。

### 7.2 TaskAdapter

任务适配器负责：

- 构造每个 Agent 阶段的结构化上下文；
- 解析并校验 Agent 输出；
- 将候选映射为 `realized_position`；
- 选择 `evaluated_position` 并报告近似；
- 计算 position adherence；
- 生成 pbest、sbest、gbest 的受控摘要；
- 确定性比较两个成功 Evaluation；
- 对候选失败给出允许的恢复动作。

### 7.3 AgentRuntime

Agent runtime 负责：

- 为粒子创建独立 thread；
- 在指定 thread 中运行一个状态机阶段；
- 返回符合阶段 Schema 的结果和用量；
- 保存 thread checkpoint 元数据；
- 恢复、轮换和关闭 thread。

### 7.4 ToolProvider 与 ResourceManager

`ToolProvider` 接受结构化 `ToolRequest`，返回结构化 `ToolResult`。外部进程必须使用 argv 数组启动，不接受拼接 shell 字符串。`ResourceManager` 对 Agent 和 Evaluator 使用独立信号量，并负责超时、取消和资源租约。

### 7.5 WikiRetriever

WikiRetriever 只接受结构化查询，返回带来源定位和证据层级的结果。第一版 `LocalWikiRetriever` 只读本地 Markdown Wiki，不执行写回。

### 7.6 RunStore 与 ArtifactStore

`RunStore` 保存事务性状态；`ArtifactStore` 保存不可变文件并返回路径、SHA-256、大小和提交状态。两者均通过 Protocol 与核心隔离。

## 8. Task package

任务使用声明式 YAML 与类型化 Python 插件混合接入：

```text
task-package/
├── task.yaml
├── prompts/
├── skills/
├── adapter.py
├── evaluator.py
└── schemas/
```

`task.yaml` 保存：

- task 名称、版本和用户 prompt；
- Wiki 路径、只读策略和检索限制；
- skills 路径；
- Agent 模型、sandbox 和阶段预算；
- 工具提供者、计算资源和并发限制；
- PSO 参数；
- 插件入口；
- 输入、输出和工具 Schema 路径；
- thread 轮换和失败策略。

YAML 不承载自定义 Python 表达式。PositionSpace、TaskAdapter、Evaluator 和 ToolProvider 通过明确的模块入口加载，并在 run 创建前执行契约检查。

运行开始后，配置、prompt、skills 元数据和计算协议被复制或内容寻址为只读快照。源文件之后的修改不改变已启动 run。

## 9. LocalCodexRuntime

第一版使用官方 `openai-codex` Python SDK 控制本地 Codex app-server。开发机复用已有 ChatGPT 登录，不在项目中保存账号密码或 API key。

线程策略：

- 一个逻辑粒子在一次 run 内保持一个独立 thread；
- 不同粒子不共享 thread；
- position、pbest、gbest 和 iteration 始终由外部 RunStore 保存；
- 达到可配置的最大 turn 数或上下文预算后生成结构化 checkpoint，并创建下一代 thread；
- thread 丢失时可以从外部 checkpoint 创建新 thread；
- thread generation 递增，历史 thread ID 保留用于审计。

Codex sandbox 默认为 workspace-write，但每个粒子只获得自己的私有工作目录。共享 Wiki 通过受控 WikiRetriever 读取，受管计算工具通过 Orchestrator 调用。

## 10. Agent 状态机

```text
PENDING
→ HYPOTHESIZING
→ PROPOSING_ACTION
→ EXECUTING
→ EVALUATING
→ REFLECTING
→ COMPLETED
```

阶段职责：

1. `HYPOTHESIZING`：Codex 基于任务、position、个体历史、pbest、sbest 和 Wiki 证据建立假说。
2. `PROPOSING_ACTION`：Codex 返回一个或多个结构化工具请求。
3. `EXECUTING`：Orchestrator 校验请求、取得资源租约并调用 ToolProvider。
4. `EVALUATING`：TaskAdapter 校验候选，Evaluator 计算指标、约束和 fitness。
5. `REFLECTING`：Codex 比较预测与结果，输出机制解释、修正假说和下一轮建议。
6. `COMPLETED`：保存完整 AgentEpisode，使粒子进入本代终态。

Agent 不得直接调用受管计算资源，也不得写入 reward、pbest 或 gbest。

## 11. 同步 iteration 数据流

每轮开始冻结 `IterationSnapshot`：

- 所有 position 和 velocity；
- 所有 pbest；
- 每个粒子的 sbest；
- 全局 gbest；
- RNG 状态；
- task/config snapshot hash；
- 本轮资源预算。

所有粒子基于同一快照并发运行。完成顺序不影响本代最优值或下一代位置。`SUCCESS`、`INVALID`、`FAILED` 和 `TIMEOUT` 都是可结束等待的粒子终态。

所有粒子结束后，在一个数据库事务中：

1. 用 TaskAdapter 的比较规则更新各 pbest；
2. 更新全局 gbest；
3. 用 SocialTopology 为每个粒子选择下一轮 sbest；
4. 生成并记录随机系数；
5. 计算、限幅和投影 velocity/position；
6. 提交新的 ParticleState 和 IterationSnapshot。

事务失败时回滚整个代末状态更新。已经完成的计算产物继续作为不可变证据保留。

## 12. PSO 更新规则

默认使用 Clerc-Kennedy 收缩因子形式：

\[
v_{i,t+1}=\chi\left[v_{i,t}+c_1r_1\Delta(pbest_i,x_{i,t})+c_2r_2\Delta(sbest_i,x_{i,t})\right]
\]

\[
x_{i,t+1}=\operatorname{project}\left(x_{i,t}\oplus v_{i,t+1}\right)
\]

默认参数：

```yaml
pso:
  cognitive_coefficient: 2.05
  social_coefficient: 2.05
  constriction_factor: 0.72984
  velocity_clamp: 0.20
```

`random_scale` 对连续空间的每个维度独立生成随机系数。所有 RNG 流从 run seed、particle ID、iteration 和用途标签通过稳定 SHA-256 派生；不使用进程相关的 Python `hash()`。

配置可以覆盖参数，但每次 run 必须保存完整快照。第一版不实现自适应惯性权重或动态参数调度。

缺失最优值时不执行未定义运算：`pbest=null` 时认知项为零，当前邻域没有任何 pbest 时 `sbest=null` 且社会项为零。粒子连续两轮没有成功 Evaluation 时，默认 FailurePolicy 从 PositionSpace 重新采样 position、把 velocity 设为零，并保留之前的失败和 pbest 历史。

## 13. 社会拓扑

全局 gbest 始终计算并保存，用于报告、停止条件和最终结果。速度更新使用抽象 sbest。

内置拓扑：

- `RingTopology`：默认，`neighborhood_radius=1`；
- `GlobalBestTopology`：所有粒子的 sbest 等于 gbest。

RingTopology 的邻域包含粒子自身及左右邻居，并以稳定 particle ID 顺序成环。并列结果用确定性 tie-breaker 选择，不能依赖任务完成顺序。

## 14. Reward 与比较

Evaluator 保留完整 metrics、constraints、uncertainty 和 provenance，同时为经典 PSO 提供有限标量 fitness。TaskAdapter 提供确定性的 `compare(left, right)`：

1. `SUCCESS` 优于无 Evaluation；
2. feasible 优于 infeasible；
3. 同可行性内按 task 的 fitness 最大化；
4. fitness 相同时使用固定的候选哈希作为 tie-breaker。

计算失败、超时或非法候选没有物理 fitness，也不能通过人为极低分混入性质分布。

## 15. Position 归因

每次 AgentEpisode 区分：

```text
target_position
realized_position
evaluated_position
position_adherence
```

TaskAdapter 优先根据真实候选重建 realized position。无法完整反推时，必须在配置中声明使用 target position 进行 reward 归因，并把未知维度和近似写入 adherence；禁止静默假设 Agent 完全遵从指令。

## 16. 状态与产物存储

默认实现为 SQLiteRunStore 加 FileArtifactStore。

SQLite 表：

- `runs`；
- `particles`；
- `iterations`；
- `stage_events`；
- `hypotheses`；
- `tool_requests`；
- `tool_results`；
- `evaluations`；
- `pbest_history`；
- `gbest_history`；
- `thread_checkpoints`；
- `artifact_index`。

文件布局：

```text
runs/<run_id>/
├── config.snapshot.yaml
├── particles/<particle_id>/
│   ├── workspace/
│   └── iterations/<iteration_id>/
│       ├── agent/
│       ├── tools/
│       ├── evaluation/
│       └── result.json
└── exports/
```

大文件不写入 SQLite。数据库只保存相对路径、内容哈希、大小、媒体类型和状态。完整产物目录使用 `result.json` 作为最后提交标志；目录存在本身不表示成功。

## 17. 失败、重试和恢复

失败分类：

- Agent：`OUTPUT_SCHEMA_INVALID`、`THREAD_LOST`、`RUNTIME_ERROR`、`TIMEOUT`；
- Tool：`REQUEST_REJECTED`、`EXECUTION_FAILED`、`RESOURCE_UNAVAILABLE`、`TIMEOUT`；
- Candidate：`INVALID`、`DUPLICATE`、`POSITION_VIOLATION`；
- Evaluation：`PARSE_FAILED`、`NOT_CONVERGED`、`MISSING_RESULT`、`TIMEOUT`；
- Run：`CONFIG_INVALID`、`AUTHENTICATION_FAILED`、`STORE_CORRUPTED`、`INCOMPATIBLE_CHECKPOINT`。

幂等键为 `(run_id, particle_id, iteration_id, stage, attempt)`。

默认重试：

- Agent Schema 错误允许两次纠正；
- MoleculeEditor 编辑允许初次尝试加两次纠正；
- 确定性工具对同一请求不自动重复执行；
- 临时资源错误默认重试一次；
- 昂贵计算只在 ToolProvider 明确标记 `retryable` 时重试；
- 认证、配置或存储损坏暂停整个 run。

失败粒子不更新 pbest。已有 pbest 时保留它；尚无 pbest 时按第 12 节的缺失项规则更新，连续两轮失败后重新采样。整代没有任何成功 Evaluation 时不执行 PSO 位置更新，并暂停 run，等待用户选择继续、调整配置或终止。

恢复只信任最后一个已提交 IterationSnapshot。恢复器校验幂等键、产物哈希和提交标志，复用完整结果，从未完成阶段继续。thread 不能恢复时用结构化 checkpoint 创建新 thread。

## 18. 阶段 A：通用框架验收

阶段 A 不依赖真实 Codex 或量化计算，包括：

- `ContinuousBoxPositionSpace`；
- 5 粒子以内的确定性模拟 Agent 任务；
- Sphere 与 Rastrigin 连续函数；
- RingTopology 和 GlobalBestTopology；
- 2 个同步 iteration 的集成流程；
- SQLite checkpoint；
- 阶段失败、重试和中断恢复；
- 文件产物哈希、缓存和幂等性。

阶段 A 必须证明：固定 seed 的 PSO 状态完全复现；pbest/sbest/gbest 正确；任务完成顺序不改变结果；失败没有物理 fitness；中断后不会重复已经提交的工具结果。

## 19. 阶段 B：红光吸收分子示例

### 19.1 固定外部依赖

共享只读 Wiki：

```text
/Users/jiaoyuan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub/ChemAgent/.worktrees/codex-opencode-runtime-base/.openchem/wiki
```

本地 MoleculeEditor skill：

```text
/Users/jiaoyuan/Documents/ChatGPT/MolToGraph/.agents/skills/molecule-editor
```

MoleculeEditor 只负责合法、可追溯的单组分闭壳层共价分子编辑和确定性几何准备。电子结构计算、光谱解析和 reward 属于外部 Evaluator。

### 19.2 运行时必填值

真实 run 创建前必须提供并通过 preflight：

- 精确母体结构：SMILES、ChemicalGraph 或 SDF；
- 电荷、多重度和可选保护位点；
- 光谱计算 argv、输入/输出 Schema 和解析器；
- 固定计算方法、基组、环境和软件版本；
- 计算超时和资源并发上限。

任何必填值缺失时，CLI 在创建 run 和消耗模型/计算资源前失败。母体分子和光谱后端是运行数据，不硬编码进通用包。

### 19.3 分子工作流

```text
Codex 提出 WikiQuery
→ LocalWikiRetriever 返回具体 source page 与证据层级
→ Codex 输出 HypothesisProposal
→ MoleculeEditor inspect 母体
→ Codex 输出完整 edit transaction
→ MoleculeEditor 化学与几何门控
→ SpectrumEvaluator
→ RedAbsorptionEvaluator
→ Codex Reflection
```

每个候选保留母体哈希、编辑命令、候选哈希、几何哈希、证据引用和计算协议哈希。Wiki 在阶段 B 中保持只读。

### 19.4 光谱输入与目标

每个激发态至少包含：

- `state_index`；
- `energy_ev`；
- `wavelength_nm`；
- `oscillator_strength`；
- `converged`；
- 可选 `root_character`。

“第一个明显吸收态”定义为：按激发能从低到高，第一个 `converged=true` 且 `oscillator_strength >= 0.05` 的激发态。

可行条件：

```text
620 nm <= wavelength_nm <= 750 nm
```

标量 fitness：

- 有显著态且位于目标波段：`1.0 + oscillator_strength`；
- 有显著态但不在目标波段：`-distance_to_band_nm / 130 + 0.01 * min(oscillator_strength, 1.0)`；
- 无显著收敛态：`-2.0`。

TaskAdapter 的比较仍先比较 feasible，再比较 fitness，因此任何不可行候选不能超过可行候选。该目标是吸收强度代理，不是荧光亮度或器件亮度。

### 19.5 运行规模

```text
5 个粒子 × 5 个同步 iteration ≤ 25 个新的光谱评价
```

默认 RingTopology 半径为 1。Agent 并发和光谱计算并发分别配置。无效、重复或缓存命中的候选不触发新的计算。

在正式运行前，必须先对母体或一个已知候选执行一次 evaluator preflight，验证命令、单位、收敛字段和解析器。启动最多 25 次真实计算属于单独的资源操作，需要再次获得用户明确授权。

### 19.6 报告

阶段 B 报告至少包含：

- 每代 gbest fitness、波长和 oscillator strength；
- 红光范围合格率；
- 有效、重复、失败和超时数量；
- 唯一候选结构哈希和种群多样性；
- 每个粒子的 pbest 演化；
- 假说、反思和 Wiki 证据链；
- target/realized position 与 adherence；
- Codex 用量、计算次数、缓存命中和耗时；
- 未找到可行候选时的诚实结论。

## 20. 测试策略

### 20.1 单元测试

- PositionSpace 运算、限幅和投影；
- 收缩因子更新；
- RNG 派生和固定 seed；
- RingTopology 邻域和 tie-breaker；
- pbest/gbest 比较；
- Evaluation 不变量；
- 配置校验。

### 20.2 Protocol 契约测试

每个可替换接口提供可复用契约测试套件。第三方插件只有通过对应契约测试，才能被 run loader 接受。

### 20.3 集成测试

使用 FakeAgentRuntime、DeterministicTaskAdapter、SQLiteRunStore 和临时 ArtifactStore 验证完整两代流程、中断恢复、缓存、幂等和事务回滚。

### 20.4 Live 测试

真实 Codex 和真实计算测试使用显式 `live` 标记，默认 pytest 不执行。Live 测试先检查本地 Codex 登录，再运行最小单粒子单阶段调用。量化计算 preflight 与 5×5 run 是两个独立门禁。

## 21. 验收标准

通用框架完成的必要条件：

1. 所有默认单元、契约和集成测试通过；
2. 固定 seed 的阶段 A 状态快照一致；
3. 核心包中不存在 Codex 或分子领域依赖；
4. 一个全新测试 task package 无需修改核心即可运行；
5. SQLite 恢复不会重复已经提交的工具动作；
6. 无效、失败和超时不会被记录为低物理 fitness；
7. LocalCodexRuntime 可以复用本地登录完成受 Schema 约束的调用；
8. 红光任务能完成 preflight，并在获得资源授权后运行 5×5 搜索；
9. 最终报告能区分计划、已提交任务、中间产物、完成计算和评价结果。

## 22. 实施分解

实现按两个独立计划推进：

1. 通用核心与阶段 A：数据类型、Protocol、PSO、状态机、存储、模拟运行和恢复；
2. LocalCodexRuntime 与阶段 B：Codex SDK、任务包加载、Wiki/MoleculeEditor/光谱适配器和 5×5 运行流程。

阶段 A 完成并通过测试后，才允许阶段 B 启动真实模型和计算资源。
