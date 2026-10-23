# vLLM / vllm-ascend Batch 一致性周报（2026-10-16 → 10-23）

核对日期：2026-10-23（北京时间）。范围：vLLM 与 vllm-ascend 的非量化模型 batch 一致性为主，SGLang、verl / Megatron 作为对齐参照。依据为本地提交树（已 fetch 快进到当日 origin/main）与 GitHub PR 状态核查；未运行 GPU/NPU 数值测试。

- [整体报告：SGLang / vLLM / vllm-ascend 一致性支持与适配汇总](sglang_vllm_ascend_consistency_report.md)（已刷新到 2026-10-23 HEAD）
- [阶段变化说明：2026-10-16 → 2026-10-23](consistency_changes_20261016_20261023.md)（本期新增）
- [阶段变化说明：2026-10-10 → 2026-10-16](consistency_changes_20261010_20261016.md)（上期，保持原文）
- [核查方法与 PR diff 证据](evidence/README.md)

## 本期三点结论

1. **vLLM**：Inductor BI 开关进入主线（[#61118](https://github.com/vllm-project/vllm/pull/61118)，10-17 合入），BI 验证名单扩容（[#61254](https://github.com/vllm-project/vllm/pull/61254)），cooperative TopK 溢出改精确选择（[#59645](https://github.com/vllm-project/vllm/pull/59645)）；但 DeepSeek-V4 拆分专项 PR（#60442~#60454、#59925/#59929）与 #60074 全部仍 Open，V4 0-diff 未进入主线。
2. **vllm-ascend**：主线 BI 面零代码改动（算子注册 / v2 sample / batch_invariance 文档 / TP4 consistency e2e 均无 diff），调度限制未解除；窗口内 151 个合入集中在 V4.1 bounded replay 跨 runner/PCP 扩展、PCP 图回放 padding、GDN 状态索引与 MTP 正确性，属于"入图 + replay + 并行组合"的正确性地基，不是新增 BI 能力。
3. **需要新增跟踪**（详见阶段变化文档第 5 节）：vLLM #62192 / #62017 / #62193 / #62219（GEMM 分桶不变性、Mamba2 SSD BI 与 scope 基础设施、draft 豁免作用域）、vllm-ascend #19442（Qwen3.5-35B-A3B 混合 GDN+MoE 的 TP4 BI e2e）与 #19237/#19238（DSA O-proj 通信分组）、SGLang #45063 / #44923（NPU 确定性算法、动态 batch 贪心标志刷新）；verl #8025 的 rollout/actor logprob mismatch 指标已合入，NPU 接入等价指标是当下动作。

## 与 rl-replay-consistency 的关系

[rl-replay-consistency](../rl-replay-consistency/README.md) 以六项特性（Sampling Mask / Index / R3 / Hidden States / Logprob / Batch 一致性）为主轴，统计窗为 10-12 → 10-19；本目录沿袭其中 B（Batch 一致性）线索，统计窗为 10-16 → 10-23，并把范围扩展到 vllm-ascend 的合入明细与训练依赖（verl / Megatron）。两者时间窗有重叠，检索口径不同，不互相取代。

## 证据等级

报告作者完成源码/diff 与公共状态核查，未复跑 PR 作者的模型测试；PR 描述中的性能数字与通过率不作为独立验收结论。涉及 Ascend 的结论需追加 NPU/HCCL 独立验证。GitHub API 匿名额度中途耗尽后，合入状态一律以本地提交树核对。
