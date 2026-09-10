# GA 构建验证：2026-09-10

## 范围

验证通用生命周期算法、基因表达、模型评价服务和 Codex 输出协议；没有启动
真实 GA 分子种群。以下 mock 轨迹仅用于测试死亡与补位，不表示分子优化性能。

| 验证 | 结果 |
|---|---|
| PSO 非 live 回归 | 1724 passed，19 deselected |
| GA 非 live 测试 | 22 passed，10 deselected |
| 真实 MoleculeEditor | 10 passed：九种操作及固定母体重放 |
| mock 20 个体 × 5 代 | 100 次请求；三代恶化后20次死亡/补位；保持20个成员 |
| 工作谱系上下文测试 | 旧线程关闭，新谱系创建不同线程；无旧线程恢复 |
| 真实 FLAME 单母体预检 | PREFLIGHT_PASSED |
| 真实 Codex production schema | gpt-5.6-luna，成功返回结构化编辑基因 |

## FLAME 基线

母体：`O=c1c2ccccc2[nH]c2ccccc12`，DCM；模型文件与原 PSO FLAME 配置相同。
吸收393.29831084021396 nm，目标距离226.70168915978604 nm，
fitness=-1.7410503399299164。当前目标仍为620–750 nm；近红外区间待明确配置。
FLAME为标量代理，没有进行TDDFT或首次明显吸收峰的验证。

本地工件：`/private/tmp/ga-live-preflight-20260910/preflight.json`。

## Codex 接口验证

仅要求生成明确标记为 mock fixture 的基因 JSON，不检索文献、不编辑分子。
返回字段与生产 proposal schema 一致。记录的 input_tokens=32859、
output_tokens=156、cached_input_tokens=0；这是 SDK 报告的整个请求用量。
本地工件：`/private/tmp/ga-codex-schema-b69rauya/result.json`。

## 重现

```sh
python tools/test_projects.py
cd GA
python -m pytest -q -m live tests/test_expression.py
python examples/molecular_screening/run.py --runs-dir runs/new-mock-directory
```

端到端工作谱系测试使用模拟 Codex/模拟评价；真实 Codex、MoleculeEditor、FLAME
分别验证了边界。真实完整 GA 搜索的性能、成功率和近红外适用性尚未测量。
