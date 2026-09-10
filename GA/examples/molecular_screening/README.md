# GA 分子吸收筛选示例

流程：选择编辑基因父代 → 块交叉 → Wiki/Codex 变异及假说 → 固定母体上
MoleculeEditor 表达 → FLAME 四性质预测 → reward 与假说判定 → 反思 →
三代死亡/补位和种群更新。

本示例有可运行的 mock 与 live 入口。工具/runtime 复用仓库 PSO 子项目的
检查过的组件；独立 GA 核心不依赖这些组件。环境需有已有 PSO 依赖和本地
Codex 登录；当前配置 model 为 gpt-5.6-luna。

目标区间由 config.json 的 target_lower_nm/target_upper_nm 指定。目前沿用
620–750 nm 红光设置；如果目标是近红外，应先显式修改区间，并使用新运行目录。
FLAME 为 DCM 溶液相模型代理；它提供单个吸收标量，不证明首个明显吸收峰，
也不替代 TDDFT。PLQY 和 log10 epsilon 合计辅助权重最多 0.01。

从仓库根目录执行：

```sh
# 完整生命周期模拟，不使用真实 Codex 或 FLAME
python GA/examples/molecular_screening/run.py --runs-dir GA/runs/mock-check

# 真实母体和 FLAME 预检；不生成设计种群
python GA/examples/molecular_screening/run.py --live --preflight-only \
  --runs-dir GA/runs/live-check --confirm-max-new-evaluations 100

# 明确启动 20 个体 × 5 代
python GA/examples/molecular_screening/run.py --live \
  --runs-dir GA/runs/live-check --confirm-max-new-evaluations 100

# 进度读取不启动计算
python GA/examples/molecular_screening/status.py GA/runs/live-check
```

100 是最多新子代预测预算；初始母体预检另加 1，持久化 ledger 总上限为101。
相同化学结构/模型/溶剂命中缓存不新增预测。后端瞬态重试沿用其单次预算项；
这是新结构评价预算，不是底层模型子进程启动次数预算。

每个工作谱系有独立 Codex thread。存活谱系从已保存 thread 引用恢复；死亡
谱系不再恢复，新成员第一次请求创建新 thread 和独立目录。agent 在只读
sandbox 规划基因，所有分子修改和评价由控制器执行。

每次检索片段由控制器赋予 W0、W1 等显式 evidence_id；输出 schema 的 enum
限定为本次片段编号。同一论文的不同片段使用不同编号，不能填论文号或行号。
代码仍独立验证返回编号；编号错误会提示合法值并要求只修正引用格式。

最多三次提案（含格式/编辑修正），失败后保留该工作谱系原个体及已知评价，
记录 fallback/NOT_TESTED 并进行失败反思。失败分子不计作三代性质恶化。
反思调用失败会记录 reflection_error，不丢弃已经成功的外部评价。

运行产物：

- manifest.json：配置、源码、Wiki、技能和输入身份。
- preflight.json：检查后的参考母体和模型基线。
- population/population-*.json：代快照、谱系轨迹、死亡事件及最好结果。
- population/trials/：请求、遗传父代、供体和每个子代结果。
- contexts/：各谱系的独立线程引用；死亡补位使用新目录。
- agent-events/：提示、原始回答、token 用量、假说、反思和失败信息。
- expressions/：基因、编译后的命令、实际分子与状态哈希链。
- compiler/cli-records/：检查、片段和编辑调用的原始输入/输出及退出状态。
- scientific-artifacts/、evaluation_budget.jsonl：科学工件、缓存与预算。

后台运行须使用 `PSO/src/multi_agent_pso/launchd.py` 生成的单次 plist，
KeepAlive=false，禁止用 launchctl submit 保活有限任务。构建验证期间未启动
真实 GA 种群；mock 验证结果不能作为分子优化效果。

Codex 传输断开最多进行四次 runtime 尝试（初始加三次恢复），复用已提交
trial/agent-event/模型缓存；不设置外层运行时限。重试耗尽后退出并保留 checkpoint。
