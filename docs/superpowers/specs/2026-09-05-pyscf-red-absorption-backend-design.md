# PySCF 气相红光吸收后端设计

日期：2026-09-05
状态：已批准并实现
范围：`examples/red_absorption` 的第一个真实量化后端，不改变通用 PSO 数学核心

## 1. 目标与非目标

目标是为现有 red-absorption adapter 增加一个本地、无 shell、JSON 输入输出的
PySCF evaluator。每次 evaluation 必须先进行气相 B3LYP/STO-3G 基态几何优化，
再在优化后的几何上计算 20 个单重 TD-B3LYP/STO-3G 激发态及 length-gauge
oscillator strength。

首轮仍采用 5 particles × 5 iterations，搜索阶段最多 25 次新 evaluation；
preflight 的一次母体计算单独计数。MoleculeEditor 继续拥有化学结构、稳定 AtomId、
编辑合法性和初始几何的权威。PySCF evaluator 只拥有量化优化、激发态计算和对应
数值 provenance。外部 `RedAbsorptionEvaluator` 继续独占 reward 与 feasible 判定。

实现时新增 `quantum` optional dependency group。先解析与 Python 3.12 兼容的 PySCF
和 geomeTRIC 正式版本，完成导入 smoke test 后立即把实际版本以 `==` 固定到项目依赖；
不得合并浮动的无上限版本范围。能力依据为 PySCF 官方 Quickstart 与
`pyscf.geomopt` 文档。

本阶段不计算频率、荧光量子产率、辐射速率、非辐射速率、系间窜越、MOMAP
光谱或 OLED 器件亮度。oscillator strength 只是吸收强度代理。

## 2. 初始母体与证据

首个母体固定为 DiKTa：

```text
canonical SMILES: O=c1c2ccccc2n2c3ccccc3c(=O)c3cccc1c32
formula: C20H11NO2
total atoms including H: 34
heavy atoms: 23
charge: 0
multiplicity: 1
```

证据链：

- Wiki source page：`sources/source-mr-tadf-009.md`；
- 原始论文：`raw/source-mr-tadf-009/v1.pdf` 第 2 页 Scheme 2；
- 交叉来源 Gaussian S0 log：
  `derived/structure-evidence/source-mr-tadf-033/selected-members/cross-source-original/DiKTa-Reson-Diketo-PBE0.log`；
- review-only XYZ：
  `derived/structures/xyz-pending-identity-method/source-mr-tadf-009/remaining-cross-source-geometry-0dddcb6becbc7fc528bfe8ac.xyz`；
- XYZ SHA-256：`19e56ebf9b5617f195e5deb7e4233b639a2301e8db6cd30426a59ea2f410d245`；
- 原始 Gaussian log SHA-256：
  `1066804b721305e8ad02acbc05349ff1ec455a2514b2aa747de572c9d1a71f61`。

SMILES 是从该源绑定坐标推导、与 Scheme 2 人工核对后得到的项目输入。
MoleculeEditor 0.1.0 已将其验证为单组分、闭壳层、`chemical_status=VALID`：

```text
state_hash: dd4f3963f9a21358a08688eb207ed3414331a29f233b50285d4f7d6eb4633852
chemical_identity_hash: d3c03170e54ba459120e915f2be6bf6d9def7f811228120a4c1550cb4c5913d1
```

这些记录不构成对外部 Wiki canonical structure manifest 的写回或晋升。

## 3. 计算协议

真实 run input 必须显式记录以下值，不提供隐式方法默认值：

```yaml
environment: gas_phase
ground_state_method: RKS
functional: B3LYP
basis: STO-3G
threads: 1
max_memory_mb: 4096
scf:
  convergence_energy_hartree: 1.0e-9
  max_cycles: 100
  grid_level: 3
geometry_optimization:
  engine: geomeTRIC
  max_steps: 100
  convergence_energy_hartree: 1.0e-6
  convergence_grms_hartree_per_bohr: 3.0e-4
  convergence_gmax_hartree_per_bohr: 4.5e-4
  convergence_drms_angstrom: 1.2e-3
  convergence_dmax_angstrom: 1.8e-3
excited_states:
  method: TDDFT
  multiplicity: singlet
  n_states: 20
  oscillator_strength_gauge: length
frequency_check: not_performed
```

`spectrum_timeout_seconds` 为 3600，evaluation concurrency 固定为 1。实际安装的
Python、PySCF、geomeTRIC、NumPy、libxc 和 BLAS/backend 版本必须在 preflight
中探测并写入 provenance；不得仅使用设计文档中的版本范围冒充实际版本。

## 4. 后端边界

新增 task-owned backend，固定路径为：

```text
examples/red_absorption/backends/pyscf_spectrum.py
```

它通过 stdin 接收一个严格 JSON 对象，通过 stdout 只返回一个严格
`SpectrumResult` JSON 对象。不得执行 shell，不得输出日志到 stdout，不得自行计算
reward。诊断日志进入 bounded stderr；超限、超时、取消和非零退出继续由
`JsonCommandProvider` 管理。

输入包含：

- MoleculeEditor evaluator payload；
- candidate chemical/state identity；
- `source_geometry_hash`；
- 完整 `CalculationProtocol`；
- 资源上限。

后端必须保持 MoleculeEditor `coordinate_order`。优化前后原子数、AtomId、元素、电荷
和多重度必须相同。PySCF 坐标使用 Å 作为交换单位，内部单位转换由 PySCF 完成并在
provenance 中记录。

## 5. 双几何 provenance

现有单一 `geometry_hash` 无法表达优化工作流，因此 spectrum contract 升级为 v2：

```text
source_geometry_hash
evaluation_geometry_hash
evaluation_geometry
geometry_optimization
```

`source_geometry_hash` 总是 MoleculeEditor 初始几何哈希。对于
`b3lyp_sto3g_optimized`，`evaluation_geometry_hash` 必须来自 B3LYP/STO-3G
优化后坐标；对于保留的 `vertical_from_molecule_editor`，两个哈希必须相同。

`evaluation_geometry` 是 bounded、内联的坐标记录：每个条目包含 AtomId、原子序数
和 x/y/z Å。宿主重新验证 AtomId/元素守恒并重新计算哈希，不能只相信后端给出的
字符串。

geometry hash 的 canonical 输入包含：

- coordinate order；
- AtomId；
- atomic number；
- 坐标按 `1e-8 Å` 量化后的十进制定点字符串；
- 电荷与多重度；
- geometry hash schema version。

量化前保留完整有限浮点坐标用于结果和后续计算。`-0.00000000` 规范化为
`0.00000000`。hash 不是旋转、平移或原子重排不变量，而是一次具体计算几何的内容
身份。

`geometry_optimization` 至少记录：状态、方法、基组、环境、backend/version、初末
能量、步数、收敛阈值、最终梯度指标和 `frequency_check=not_performed`。

schema version 固定为 `red-absorption:spectrum:v2`，同时把 evaluator version 从
`red-absorption-evaluator:v1` 升级为 `red-absorption-evaluator:v2`。旧 v1 cache
不能被 v2 请求复用。

请求级 cache key 保持：

```text
chemical_identity_hash
+ source_geometry_hash
+ protocol_hash
+ evaluator_version
```

因为 evaluation geometry 在启动计算前未知。完成后的 spectrum hash 覆盖完整优化
记录、evaluation geometry 和全部激发态。

## 6. 状态和失败语义

下列情况必须返回或映射成 `SpectrumResult(status="FAILED")`，且
`fitness=None`：

| code | 条件 |
|---|---|
| `INPUT_MISMATCH` | 初始结构、电荷、多重度或 source hash 不一致 |
| `SCF_NOT_CONVERGED` | RKS SCF 在 100 cycles 内未收敛 |
| `GEOMETRY_NOT_CONVERGED` | geomeTRIC 在 100 steps 内未收敛 |
| `GEOMETRY_IDENTITY_MISMATCH` | 优化改变 AtomId、元素或原子数 |
| `INVALID_OPTIMIZED_GEOMETRY` | 坐标非有限、单位错误或 geometry hash 不一致 |
| `TDDFT_NOT_CONVERGED` | TDDFT 根求解失败或返回不完整 |
| `INVALID_SPECTRUM` | 能量、波长或 oscillator strength 非法 |
| `TIMEOUT` | 命令超时并完成子进程清理 |
| `CANCELLED` | 调用取消且 terminal failure 已持久化 |
| `PROVENANCE_MISMATCH` | protocol/backend/geometry provenance 不一致 |

不得把 parser、SCF、优化或 TDDFT 失败转换成 `-2.0`；`-2.0` 仅表示一个成功光谱
中没有 oscillator strength ≥0.05 的收敛态。

## 7. Reward

按能量从低到高选择第一个 `converged=true` 且
`oscillator_strength >= 0.05` 的单重态。

```text
red band: 620 nm <= wavelength <= 750 nm
in band: fitness = 1.0 + oscillator_strength
out of band: fitness = -distance_to_band_nm / 130
                         + 0.01 * min(oscillator_strength, 1.0)
no significant state: fitness = -2.0
```

TaskAdapter 继续先比较 feasible，再比较 fitness。吸收 oscillator strength 不得在
报告中改称荧光或器件亮度。

## 8. 资源和持久化

- 5 × 5 搜索最多持久化 25 个新的 request reservation；
- 一次 reservation 覆盖优化和 TDDFT，不能分两次计数；
- preflight 的一次母体计算不占搜索的 25 次；
- 同一 request cache key 的并发调用等待同一 terminal result；
- `DurableBudgetLedger` 在外部计算前 fsync reservation；
- 成功和失败均写入 terminal event，取消不得留下永久 PENDING；
- backend stdout、stderr、输入 JSON、geometry 和事件文件继续受现有 byte/node/depth
  上限约束；
- 不保存大型 checkpoint。若后续需要 checkpoint，必须通过 `ArtifactRef` 增加独立
  规格，不能塞入 event JSON。

母体的 50 原子限制只用于初始 DiKTa 选择，不新增“所有候选必须 ≤50 原子”的科学
约束。候选继续受现有单事务 edit budget 和 fragment-heavy-atom cap 限制。

## 9. Preflight 与实际运行门禁

preflight 顺序：

1. 验证 task、run input、Codex 登录和源码版本身份；
2. 验证 MoleculeEditor DiKTa structure/initial geometry；
3. 探测 PySCF、geomeTRIC、libxc、BLAS、线程和内存设置；
4. 对 DiKTa 执行一次真实优化加 TDDFT；
5. 宿主重算 evaluation geometry hash 并验证 spectrum provenance；
6. 外部 evaluator 计算一次母体 Evaluation；
7. 所有 owned resources clean close；
8. 验证空 `runs.sqlite`，最后发布 passing preflight artifact。

preflight 本身会消耗一次量化计算，因此依赖安装和单元/live smoke test 完成后，仍需
用户再次明确授权才能执行。只有 matching passing artifact 和
`--confirm-max-new-evaluations 25` 同时存在时，才能启动 5 × 5。

真实输入文件固定写入
`examples/red_absorption/inputs/dikta-gas-b3lyp-sto3g.yaml`；其中不得包含本机凭据，
但 spectrum argv 必须使用当前 conda 环境 Python 和仓库内 backend 的绝对路径快照。

## 10. 测试矩阵

### 10.1 非 live

- CalculationProtocol 的气相、RKS、grid、SCF、geomeTRIC 和 20 roots 严格字段；
- `source_geometry_hash` / `evaluation_geometry_hash` 条件不变量；
- evaluated geometry 的 finite、AtomId、元素、数量、单位、量化和 hash 测试；
- vertical workflow 要求两个 hash 相同；
- optimized workflow 要求成功 optimization record 与 evaluation geometry；
- fake PySCF/geomeTRIC 的 SCF、优化、TDDFT、parser、provenance、timeout 和 cancel
  路径；
- request cache key 不错误地使用未知 evaluation hash；
- terminal result 的跨重启恢复和重复候选单次 execution；
- evaluator reward 与 v2 provenance；
- CLI preflight summary 显示 source/evaluation geometry、backend 和资源估计。

### 10.2 live smoke

使用乙烯而非 DiKTa，实际执行一次气相 B3LYP/STO-3G 优化和 20-root TDDFT。
该测试只验证后端可运行、坐标守恒、收敛字段、oscillator strength 和 JSON contract，
不把结果当作精度 benchmark。

### 10.3 DiKTa preflight

非 live 全套和乙烯 live smoke 均通过后，经单独授权执行一次 DiKTa preflight。
记录命令、CPU/BLAS、线程、内存、依赖版本、elapsed time、SCF/优化/TDDFT 状态、
两个 geometry hash、20 个根和 artifact path。只有这些证据完整时才讨论 5 × 5。

## 11. 验收条件

- 通用 PSO core 不导入 PySCF、RDKit、MoleculeEditor 或分子字段；
- task-owned backend 可在无 shell JSON 边界内独立运行；
- 双几何 provenance 可重算、不可伪造为单一初始几何；
- 所有失败 fail closed 且不产生 reward；
- 非 live 全套通过；
- 乙烯 live smoke 通过并留下完整版本/资源记录；
- 未经再次授权不运行 DiKTa preflight 或 5 × 5；
- 文档明确 frequency check 未执行、oscillator strength 不是发光亮度。
