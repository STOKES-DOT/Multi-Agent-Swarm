# Multi-Agent PSO

## Development

```bash
conda env create -f environment.yml
conda run -n multi-agent-pso python -m pip install -e '.[dev]'
conda run -n multi-agent-pso pytest -q
```

> 状态：Stage A 已实现
> 当前范围：可复现的通用 PSO core、持久化、checkpoint recovery 与 deterministic continuous benchmarks。分子设计、Codex、RDKit、MoleculeEditor、TDDFT 与 MOMAP 属于后续 Stage B 愿景，不会由 Stage A 执行。

## Stage A benchmarks

Stage A 使用真实 `SynchronousSwarmRunner`、SQLite run state 与 immutable artifact store；默认测试不运行任何 `live` 外部任务。

```bash
conda run -n multi-agent-pso multi-agent-pso benchmark sphere --runs-dir /private/tmp/multi-agent-pso-runs
conda run -n multi-agent-pso multi-agent-pso benchmark rastrigin --runs-dir /private/tmp/multi-agent-pso-runs --seed 42
```

每个 run 目录包含 `runs.sqlite` 与 `artifacts/summary.json`。summary 记录 protocol/config descriptor、`initial_gbest`、`final_gbest` 和相对输出引用。相同 descriptor 在独立目录生成相同的 snapshot state；改变 seed、dimension、边界、拓扑或更新参数会改变 run/config identity。

## 1. 目标

本项目讨论一种用于功能分子设计的 Multi-Agent PSO 架构。

核心思想是：

- 使用一个群体优化框架，当前以 PSO 为具体讨论对象；
- 一个独立的科研 Agent 对应一个 PSO 粒子；
- 各 Agent 拥有相互隔离的上下文和研究历史；
- 所有 Agent 共享同一个由大量分子设计类文献构建的 LLM Wiki 知识库；
- Agent 基于 Wiki 提出分子设计假设，在已知分子上进行改进；
- TDDFT 和 MOMAP 等理论计算提供外部物理反馈；
- reward 用于更新各粒子的 pbest、群体的 gbest 以及下一轮研究方向。

总体循环为：

~~~text
LLM Wiki 文献知识库
        ↓
共同 Research Question
        ↓
多个独立 Agent 粒子
        ↓
各自提出假设并改进已知分子
        ↓
TDDFT / MOMAP 理论计算
        ↓
统一 reward 评价
        ↓
更新 pbest / gbest / PSO 位置
        ↓
Agent 结合 Wiki 与新结果修改下一轮假设
        ↺
~~~

## 2. 当前已经确认的设计

### 2.1 单一优化框架

当前只讨论一种群体优化算法，即 PSO。暂不混用遗传算法、蚁群算法或其他群体搜索方法。

### 2.2 Agent 即粒子

一个独立 Agent 对应一个 PSO 粒子。Agent 是粒子的科研执行者和身份载体，但 PSO 的数学状态仍由外部 Orchestrator 显式保存。

### 2.3 独立上下文与共享知识

每个 Agent 具有独立上下文，不直接读取其他 Agent 的完整推理过程。所有 Agent 共享同一个文献 Wiki，Wiki 主要承担领域知识底座的作用。

### 2.4 一轮只完成一个有界科研循环

每个 PSO iteration 中，每个 Agent 只完成一次有界 auto-research loop，并产生一个主要候选及对应研究结果。完成后再进入群体评价和下一代更新。

### 2.5 共同问题、独立假设

同一群体围绕同一个 Research Question 工作，但不同 Agent 可以独立：

- 检索和解释 Wiki 知识；
- 提出机制或分子设计假设；
- 选择母体分子；
- 选择结构改进路径；
- 形成候选分子。

### 2.6 gbest 是完整研究结果

gbest 不是单纯性能最高的分子，而是综合 reward 最高、经过统一验证的 ResearchPacket。

### 2.7 position 表示完整研究方向

粒子的 position 不只表示一个分子结构，而是结构化的研究方向，可以包含：

- 机制假设方向；
- 母体分子家族；
- 分子组合方式；
- linker、coupling、rigidity 等设计变量；
- 预期电子结构变化；
- 目标光谱区域。

### 2.8 Agent 具有有界自主性

PSO 给出目标 position、移动方向和允许偏差。Agent 在该范围内进行科学解释和具体分子设计，而不是机械执行，也不能无记录地完全偏离。

## 3. Agentic PSO 的基本映射

| PSO 概念 | Multi-Agent auto-research 中的含义 |
|---|---|
| Particle | 一个上下文独立的 Research Agent |
| Position | Agent 当前的结构化研究方向 |
| Velocity | 下一轮各研究维度的移动趋势或搜索偏好 |
| pbest | 该 Agent 历史上综合 reward 最好的已验证 ResearchPacket |
| gbest | 群体中综合 reward 最好的已验证 ResearchPacket |
| Fitness | 统一 Evaluator 根据分子性质和计算可信度得到的 reward |
| Iteration | 每个 Agent 各完成一次有界科研循环，再统一更新群体 |

PSO 状态不应只保存在 Agent 的自然语言上下文中。至少应存在一个外部、机器可读的状态：

~~~text
ParticleState
├── particle_id
├── agent_thread_id
├── position
├── velocity
├── pbest
├── latest_reward
├── iteration
└── private_workspace
~~~

## 4. 单个 Agent 的研究循环

每轮输入：

~~~text
共同输入
├── Research Question
├── Wiki snapshot 或检索入口
├── reward 定义
└── 统一计算协议

粒子私有输入
├── 当前 position
├── PSOUpdateDirective
├── 自己的 pbest
├── gbest 的结构化摘要
└── 本轮预算
~~~

Agent 执行：

~~~text
检索 Wiki
→ 形成或修正分子设计假设
→ 选择已知母体分子
→ 根据 PSO 方向提出结构改进
→ 生成并检查候选结构
→ 执行或提交 TDDFT / MOMAP 计算
→ 分析结果
→ 返回结构化 ResearchPacket
~~~

Agent 输出至少应能够回答：

- 提出了什么假设；
- 从哪个已知分子出发；
- 修改了什么；
- 为什么预期该修改有效；
- 得到了哪些计算结果；
- 预测与结果是否一致；
- 相比母体改进了什么；
- 下一轮假设应如何变化。

## 5. ResearchPacket

ResearchPacket 是单个粒子一次 iteration 的主要输出，也是 reward、pbest 和 gbest 的评价对象。

当前概念结构为：

~~~text
ResearchPacket
├── Particle / Iteration ID
├── Wiki reference
├── Research Question
├── Hypothesis
├── Parent molecule
├── Candidate molecule
├── Design change
├── Prediction
├── TDDFT / MOMAP results
├── Parent-relative improvement
├── Mechanistic interpretation
├── Calculation and structure status
├── Reward components
└── Proposed next hypothesis
~~~

正式字段和 JSON Schema 尚未确定。

## 6. Reward 函数：目标颜色与高亮度示例

这是一个正在讨论的示例，不是已经冻结的最终 reward。

若目标暂定为固定溶剂、温度、质子化态和激发波长下的单分子荧光性能，分子亮度可写为：

\[
B_i(\lambda_{\mathrm{ex}})
=
\varepsilon_i(\lambda_{\mathrm{ex}})\Phi_{F,i}
\]

其中：

- \(\varepsilon(\lambda_{\mathrm{ex}})\) 是激发波长处的摩尔吸光系数；
- \(\Phi_F\) 是荧光量子产率。

若可获得辐射和非辐射速率：

\[
\Phi_F
=
\frac{k_r}
{k_r+k_{\mathrm{IC}}+k_{\mathrm{ISC}}+\cdots}
\]

概念阶段若只有发射峰，可先定义颜色代理：

\[
C_i
=
\exp\left[
-\frac{
(\lambda_{\mathrm{em},i}-\lambda_{\mathrm{target}})^2
}{
2\sigma_\lambda^2
}
\right]
\]

若具有完整发射谱，则应考虑由发射谱得到颜色坐标或目标谱带匹配，而不是只使用峰值波长。

当前建议的 reward 原则是：

> 先要求颜色进入目标范围，再在颜色合格的候选中优化亮度。

这可以避免“非常亮但颜色错误”的分子成为 gbest。具体分段形式、归一化方式和权重仍待确定。

需要保持以下边界：

- TDDFT oscillator strength 是吸收强度相关量，不能直接等同于发光亮度；
- 高辐射速率不必然意味着高荧光量子产率；
- 归一化发射谱的峰高不能直接用于比较不同分子的绝对亮度；
- 计算失败与真实低性能必须分开；
- 不同环境和不同计算协议下的 reward 不应直接比较。

此外，最终还需明确优化对象是溶液分子、固态发光材料还是器件性能。

## 7. 以 Codex 作为粒子 Agent

建议由外部 Python PSO Orchestrator 管理算法状态，并为每个逻辑粒子维护一个独立 Codex thread。

~~~text
Python PSO Orchestrator
├── Particle 001 → Codex thread 001
├── Particle 002 → Codex thread 002
├── ...
└── Particle 050 → Codex thread 050
~~~

50 个逻辑粒子不等于必须同时启动 50 个 Agent。实际并发度应由模型请求、TDDFT/MOMAP 计算资源和预算共同决定。

### 7.1 CLI 原型

Codex CLI 可以用非交互模式启动单个粒子：

~~~bash
codex exec \
  --json \
  --output-schema research-packet.schema.json \
  -C particle-001 \
  "<particle prompt>"
~~~

后续 iteration 可以通过 Codex 的 resume 能力继续该粒子的上下文。这里的 JSON 输出是 JSONL 事件流，而最终 ResearchPacket 应接受独立 schema 校验。

CLI 适合先验证少量粒子的端到端流程。

### 7.2 Python SDK 方向

正式系统可以考虑 Codex Python SDK 的异步接口，将每个粒子绑定到独立 thread：

~~~python
# 仅为概念示意，不是当前可运行实现
threads = [
    await codex.thread_start(...)
    for particle in particles
]

results = await asyncio.gather(*[
    thread.run(build_particle_prompt(particle))
    for particle, thread in zip(particles, threads)
])
~~~

PSO 的 position、velocity、pbest、gbest、reward 和 iteration 始终由 Orchestrator 持有。Codex Agent 负责完成一轮科研任务并返回 ResearchPacket，不负责修改全局 PSO 真值状态。

CLI 与 SDK 的最终选择、并发策略、thread 恢复方式和长期上下文管理尚未确定。

## 8. Wiki 的当前定位

Wiki 是利用大量分子设计类文献搭建的 LLM 知识库，主要用于：

- 检索已有分子设计策略；
- 查找结构—性质关系；
- 提供机制解释与文献证据；
- 提供已知成功、失败和边界案例；
- 帮助各 Agent 形成不同的设计假设。

Wiki 不等同于：

- PSO 的 gbest；
- 粒子的私有上下文；
- 本次 run 的计算结果；
- reward 或群体状态。

本次 run 产生的 TDDFT/MOMAP 结果应先作为独立研究记录保存。是否以及如何将经过验证的新认识回写 Wiki，目前尚未决定。

## 9. 主要职责边界

### PSO Orchestrator

- 管理粒子与 iteration；
- 维护 position、velocity、pbest 和 gbest；
- 调用 Codex Agent；
- 控制并发、预算、超时和重试；
- 调用统一 Evaluator。

### Codex Particle Agent

- 检索 Wiki；
- 提出和修正假设；
- 改进已知分子；
- 执行受约束的结构设计；
- 调用或提交理论计算；
- 返回 ResearchPacket。

### Evaluator

- 检查结构和计算结果是否可用；
- 根据冻结的 reward 计算 fitness；
- 保证不同粒子的评价可比较；
- 不接受 Agent 自行声明的 reward。

## 10. 尚待讨论

- ResearchPosition 的具体数值或离散编码；
- PSO velocity 在混合研究空间中的定义；
- pbest 和 gbest 向 Agent 暴露哪些信息；
- reward 的最终物理定义和权重；
- 溶液、固态或器件应用场景；
- TDDFT 与 MOMAP 的计算层级和资源分配；
- Wiki 的检索接口及是否允许后续写回；
- Agent thread 是否跨 iteration 长期保留；
- 50 个逻辑粒子的实际并发策略；
- 计算失败、Agent 失败和超时的处理；
- ResearchPacket 的正式 Schema；
- 如何防止所有 Agent 过早趋同到同一假设和分子家族。

## 11. 当前推荐的最小验证

在完整实现前，可先验证：

~~~text
1 个共同 Research Question
+
2–3 个 Codex 粒子
+
1 次 iteration
+
每个粒子 1 个候选
+
一个简化 reward
~~~

该验证只回答以下问题：

- 独立 Agent 是否能基于同一 Wiki 提出明显不同的假设；
- PSO 指令是否能约束而不完全扼杀 Agent 的自主推理；
- ResearchPacket 是否足以被统一评价；
- pbest/gbest 信息能否有效影响下一轮设计。

通过这一验证后，再扩展到约 50 个逻辑粒子及多轮搜索。

## 12. 参考资料

- OpenAI Docs: [Codex CLI](https://developers.openai.com/codex/cli/)
- OpenAI Docs: [Codex SDK](https://developers.openai.com/codex/sdk/)
- IUPAC Gold Book: [brightness](https://goldbook.iupac.org/terms/view/BT07338)
- MOMAP 1.0: [Molecular Physics, DOI 10.1080/00268976.2017.1402966](https://doi.org/10.1080/00268976.2017.1402966)
- CIE: [CIE 1931 colour-matching functions](https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer)
