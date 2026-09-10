# Multi-Agent GA

遗传算法选择并组合 agent 的**分子编辑操作编码**。执行编辑得到的分子是
表型；评价器计算分子性质并返回 reward，GA 根据 reward 选择染色体。

已实现：

- 可重现的锦标赛选择、精英保留、序列交叉、变异请求和定代更新。
- `EditGene` / `EditProgram`：操作类型、化学位点规则、操作参数。
- `MolecularGeneWorker`：交叉 → Wiki 证据驱动的变异接口 → 表达及外部评价接口。
- 并发限制、无效子代处理、每代原子发布的 checkpoint、已提交代的断点续跑。

当前是可测试的核心及服务接口。真实 Codex Wiki mutator、操作序列到
MoleculeEditor 命令的绑定器、FLAME 评价服务还未接线，不应直接启动分子搜索。
PSO 项目中的相关组件将作为这些服务的实现基础，不复制量化或 reward 代码。

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

初始实现参数：population=20、elite=2、tournament=3、crossover=0.6、
mutation=0.8、concurrency=4、seed=42。它们是可配置起点，尚未经过分子优化实验。
没有不同基因型可供交叉时请求变异。每代产生 population-elite 个子代请求，
变异/表达服务内部负责计算预算和最多三次提案等领域策略。

checkpoint 保存基因、表型引用、父代/供体、请求种子、失败信息和评价。
恢复时校验配置、评价协议和初始种群身份；已提交代不会重新调用 worker。
中途崩溃的未提交代可能重发相同 request_id，worker 必须使用持久化幂等缓存。
当前 checkpoint 仅供单一控制进程使用，不支持多个进程同时写同一目录。

参见 [基因定义](docs/genotype.md)。
