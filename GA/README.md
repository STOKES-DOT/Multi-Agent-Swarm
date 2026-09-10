# Multi-Agent GA

通用算法与应用示例独立组织：

```text
GA/
├── src/multi_agent_ga/
│   ├── core.py          个体、遗传父代请求、参数及早期基础 GA
│   ├── lifecycle.py     完整生命周期 GA：选择、精英、死亡、补位、档案
│   └── persistence.py   单控制器锁、原子记录、摘要校验
├── examples/molecular_screening/
│   ├── genes.py         分子编辑操作染色体与块交叉
│   ├── compiler.py      固定母体的符号位点绑定及 MoleculeEditor 表达
│   ├── agent.py         Wiki 变异、假说、反思及 Codex 工作谱系会话
│   ├── services.py      FLAME、工件与持久化计算预算
│   ├── reward.py        可配置波段和弱辅助性质评分
│   ├── config.json      示例配置
│   ├── run.py           mock/live/preflight 入口
│   └── status.py        只读 checkpoint 进度
└── tests/
```

新任务使用 `LifecycleSearch`。通用核心不导入分子、Wiki、Codex、RDKit 或 FLAME。
worker 返回已由外部评价的 `Individual`，包括 `fitness`、`feasible` 和
`target_distance`。距离单位与容差由具体任务定义；核心只要求有限非负值。

每代所有工作谱系各测试一个子代。锦标赛选择遗传父代，精英机制可保留较好的
旧基因型，但该工作谱系仍记录本代实际试验距离。死亡优先于精英保留，历史最好
结果则永远保留在独立档案中。种群大小不因死亡或编辑失败缩小。

三次有效编辑需要四个距离样本（起点加三个结果）：

- 连续远离且累计恶化超过 tolerance：`moving_away`。
- 仍在目标区外，窗口内没有一次超过 tolerance 的净改善：`no_progress`。
- 编辑失败不计入距离历史；连续 failure_limit 轮无有效编辑：`execution_failures`。
- 死亡后复制该槽位的初始基线为新个体，生成新 lineage_id，年龄与失败计数归零。
  worker.reconcile 关闭旧线程。新谱系首次工作从初始基因型变异，无旧上下文或供体。

`lineage_id` 跟踪 Codex 工作谱系；遗传父代、供体基因型另行记录。两者不混用。

```sh
cd GA
python -m pytest -q
python examples/molecular_screening/run.py --runs-dir runs/mock-check
python examples/molecular_screening/status.py runs/mock-check
```

示例默认为 mock，不调用 Codex 或科学工具。live 使用独立显式开关和预算确认，
见 [分子示例](examples/molecular_screening/README.md)。

每代 checkpoint 有内容摘要和前代摘要链；完成的 trial 单独提交。重启会复用已
提交 trial 和代结果。崩溃时未提交的外部调用仍可能被再次发起，科学服务使用
持久化预算和化学/模型缓存避免盲目重算。PENDING 科学计算需核查后人工处理。
同一运行目录禁止两个控制器同时写入；参数、母体、代码或协议变化需要新目录。
没有运行全局时限，也没有加入 CPU 线程限制。
