# 分子编辑操作编码

应用实现位于 `examples/molecular_screening/`。基因是编辑操作，完整分子是表型。

`EditGene` 保存 `operation`、`site_rule`、`parameters_json` 和 `block`。
生产表达器要求 site_rule 是 JSON 对象，使用冻结母体锚点或前序输出符号。
自然语言 site_rule 仅可用于设计讨论，不能进入实际表达。

```json
{
  "operation": "replace_atom",
  "site_rule": "{\"atom\":\"seed.atom.0\"}",
  "parameters_json": "{\"atomic_number\":16}",
  "block": "carbonyl_change"
}
```

`seed.atom.N` 和 `seed.bond.N` 是本次冻结参考母体检查表的零起始索引，
由控制器绑定到检查后的真实 AtomId/BondId。不能把它们跨不同母体协议解释。
它们不保护原子；删除某锚点后再引用它会被拒绝。

各操作的 site_rule 键：

| 操作 | 位点角色 |
|---|---|
| add_atom | 空对象 |
| remove_atom / replace_atom | atom |
| add_bond | begin, end |
| remove_bond / change_bond | bond |
| attach_fragment | anchor |
| detach_fragment / substitute_fragment | bond, retained |

新增原子/键可通过参数 `symbol` 声明输出；后续位点使用 `output.name`。
绑定器支持同一块内的事务局部引用，并从 CLI 返回映射记录已提交输出的实际 ID。
片段操作使用 `fragment_smiles`、`fragment_anchor`（检查图的零起始原子索引）
和 `bond_type`；片段由控制器调用 MoleculeEditor inspect 后装入命令。

相同 block 的相邻基因在同一事务提交。交叉仅发生于块边界，不能把
`add_atom + add_bond` 的连接依赖拆开。跨块输出依赖重新绑定；缺失、删除或
冲突的引用会产生无效子代，不能静默换位点。每个中间提交块都必须化学有效。

染色体始终在同一冻结母体上完整表达，不重复以已编辑表型作为全程序起点。
每次表达记录源状态哈希、编译命令和子状态哈希。没有人为原子数/片段大小限制，
但保留 JSON 字节预算、有效化学域和有限资源预算。

Wiki 变异返回完整的新操作程序、可证伪假说和本次检索命中编号；控制器验证
引用确实属于本次检索。假说、证据和反思单独归档，不进入 genotype hash。
仅交叉请求不允许 agent 顺便改变基因；需要修改必须由 mutation_requested 授权。

正式 JSON 编码仍兼容早期 v1 容器，block 有默认值；自然语言位点的旧原型
不会被生产绑定器接受。运行身份同时冻结代码与母体，因此不能把旧原型运行
当成当前表达协议恢复。
