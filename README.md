# Multi-Agent-Swarm

以多个科研 agent 执行假说、工具计算、评价和反思，比较不同的种群优化算法。

```text
Multi-Agent-Swarm/
├── PSO/       原 PSO 项目、分子设计适配器、测试及历史 runs
├── GA/        遗传算法核心、编辑操作基因及 Wiki 变异接口
├── tools/     仓库级检查工具
└── .worktrees/ 历史开发工作区
```

- [PSO](PSO/README.md)：保留 `multi_agent_pso` 包名、原实验配置和结果。
- [GA](GA/README.md)：基因是编辑操作编码；分子是表达后的表型。
- [GA 基因定义](GA/docs/genotype.md)：基因、交叉、Wiki 变异与评价边界。

两个子项目各有 `pyproject.toml`，独立安装和测试。算法通用核心不得依赖
Wiki、RDKit、FLAME 或某一种分子设计任务。Wiki/编辑器/评价器通过适配接口接入。

```sh
python -m pip install -e './PSO[dev]' -e './GA[dev]'
python tools/test_projects.py
```

`tools/test_projects.py` 分别运行两个测试集，默认不会启动 Codex 或科学计算。
分子编辑、量化协议、模型权重和已有评分规则仍在 PSO 子项目维护；GA 的
真实 Wiki 变异与基因表达服务接线尚待完成，当前提供可测试的接口和算法核心。

本地迁移保留 `Multi-Agent-PSO → Multi-Agent-Swarm/PSO` 兼容链接，以便历史
绝对路径和现有 conda editable install 继续工作。历史工件、哈希及任务快照
不重写。GitHub remote 名称本次不改；仓库的本地目录名称为 Multi-Agent-Swarm。
