# Dense molecular checkpoint budget

The AgentLoop JSON node limit and LocalCodexRuntime recovery node limit are
100,000. The byte budget remains 256 KiB, maximum depth 32, and single
collection limit 4,096. Larger scientific payloads still belong in artifacts.

On 2026-09-08, the 20x50 large-edit run
`flame-2710077666dd5b9a624c3a4a` failed after tool execution when proposal,
tool request, tool result, candidate, and inherited graph were combined into
a checkpoint. Reconstructing iteration 1 from persisted proposals and tool
results showed 19 affected checkpoints with 10,062–13,136 nodes and
115,227–146,032 UTF-8 bytes. They exceeded the previous 10,000-node guard,
despite fitting within the byte budget. All 20 reconstructed checkpoints and
326 existing stored checkpoints passed the updated persistence and runtime
serialization checks. No historical run records were rewritten.

Regression tests cover dense JSON, execution-to-evaluation interruption and
resume without repeating tool execution, and rejection above the new node
limit while still below the byte limit. Runtime and persistence budgets are
checked together so a newly persisted checkpoint remains restorable.

Existing terminal failure episodes remain historical failures. Resuming the
same run continues its current generation and reuses committed tool results
and the durable FLAME cache; this budget fix does not reclassify old outcomes.
