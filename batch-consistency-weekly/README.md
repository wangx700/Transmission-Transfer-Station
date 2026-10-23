# vLLM / vllm-ascend Batch 一致性周报（2026-10-16 → 10-23）

核对日期：2026-10-23（北京时间）。范围：vLLM 与 vllm-ascend 的非量化模型 batch 一致性为主，SGLang、verl / Megatron 作为对齐参照。依据为本地提交树（已 fetch 快进到当日 origin/main）与 GitHub PR 状态核查；未运行 GPU/NPU 数值测试。

- [整体报告：SGLang / vLLM / vllm-ascend 一致性支持与适配汇总](sglang_vllm_ascend_consistency_report.md)（已刷新到 2026-10-23 HEAD）
- [阶段变化说明：2026-10-16 → 2026-10-23](consistency_changes_20261016_20261023.md)（本期新增）
- [阶段变化说明：2026-10-10 → 2026-10-16](consistency_changes_20261010_20261016.md)（上期，保持原文）
- [核查方法与 PR diff 证据](evidence/README.md)
- [DeepInfer / DeepSeek-V4.1：Batch 不变性流程、算子与 Ascend 复用说明](deepinfer_v41_batch_invariance.md)（固定版本源码专题，含调用链、明确阻塞和复用边界）

## DeepInfer V4.1 专题补充

DeepInfer 的 Ascend 部分算子已有确定性实现，包括 HC prenorm 固定 K 分片与有序累加、按序列身份采样，以及可显式启用的 EP dispatch 确定性排序；FlashMLA Ascend 路径本身不启用 split-KV。但框架仍禁止完整 BI，BI 与专家重排同时开启时要求的单次 combine 归约还与 DeepEP-Ascend 参数契约冲突。库能力存在不等于框架已启用，更不等于整模型验收通过。详见[专题第 4 节](deepinfer_v41_batch_invariance.md#4-ascend-当前可以复用什么)。

专题基线为 `8b161497` 及其锁定子模块，不计入本期新增提交；未进行 GPU/NPU 实测。

## 本期三点结论

1. **vLLM**：Inductor BI 开关进入主线（[#61118](https://github.com/vllm-project/vllm/pull/61118)，10-17 合入），BI 验证名单扩容（[#61254](https://github.com/vllm-project/vllm/pull/61254)），cooperative TopK 溢出改精确选择（[#59645](https://github.com/vllm-project/vllm/pull/59645)）；但 DeepSeek-V4 拆分专项 PR（#60442~#60454、#59925/#59929）与 #60074 全部仍 Open，V4 0-diff 未进入主线。
2. **vllm-ascend**：主线 BI 面零代码改动（算子注册 / v2 sample / batch_invariance 文档 / TP4 consistency e2e 均无 diff），调度限制未解除；窗口内 151 个合入集中在 V4.1 bounded replay 跨 runner/PCP 扩展、PCP 图回放 padding、GDN 状态索引与 MTP 正确性，属于"入图 + replay + 并行组合"的正确性地基，不是新增 BI 能力。
3. **需要新增跟踪**（详见阶段变化文档第 5 节）：vLLM #62192 / #62017 / #62193 / #62219（GEMM 分桶不变性、Mamba2 SSD BI 与 scope 基础设施、draft 豁免作用域）、vllm-ascend #19442（Qwen3.5-35B-A3B 混合 GDN+MoE 的 TP4 BI e2e）与 #19237/#19238（DSA O-proj 通信分组）、SGLang #45063 / #44923（NPU 确定性算法、动态 batch 贪心标志刷新）；verl #8025 的 rollout/actor logprob mismatch 指标已合入，NPU 接入等价指标是当下动作。

## vllm-ascend 待做事项速览

完整清单（12 项，含依据与对照）见[阶段变化文档第 7 节](consistency_changes_20261016_20261023.md#7-vllm-ascend-可以做--待做的事情)。P0 三项互相独立、不依赖未合入的上游 PR，建议最先启动：

1. **接入 verl #8025 等价的训推差异度量**（NPU rollout/actor raw logprob mismatch：count 求和、diff_max、one-ULP 掩蔽），Qwen3 dense 先出首轮统计。
2. **NPU GEMM/算子分派自查**：实现选择与 tile 分桶是否依赖 M（token 数），补算子级跨 batch 不变性测试（对照 vLLM #62192 思路）。
3. **V4/V4.1 图回放回归矩阵**：trace replay 固定 decode 历史 → eager vs graph、BS1/BSN、同图档/跨图档、有无 padding，逐 token logits/logprob bitwise 比较（覆盖 #18936/#19034/#19192/#19321/#19177 修复点）。

P1：Router/Top-K 规则核对（对照 #59985/#59645/#60453）、通信归约顺序核对（#19237 / #59925 契约）、动态 batch 贪心标志自查（对照 SGLang #44923）与 GDN 混合回归（#19378 + SGLang 三 PR）；编译等价约束与 DS4 数值契约清单随上游收敛推进。P2：CI 扩展（移植 ROCm"调度准确性 + BI"做法）、spec/MTP 单独验收。#19442 属切片基础覆盖，不是完整模型验收。

## 与 rl-replay-consistency 的关系

[rl-replay-consistency](../rl-replay-consistency/README.md) 以六项特性（Sampling Mask / Index / R3 / Hidden States / Logprob / Batch 一致性）为主轴，统计窗为 10-12 → 10-19；本目录沿袭其中 B（Batch 一致性）线索，统计窗为 10-16 → 10-23，并把范围扩展到 vllm-ascend 的合入明细与训练依赖（verl / Megatron）。两者时间窗有重叠，检索口径不同，不互相取代。

## 证据等级

报告作者完成源码/diff 与公共状态核查，未复跑 PR 作者的模型测试；PR 描述中的性能数字与通过率不作为独立验收结论。涉及 Ascend 的结论需追加 NPU/HCCL 独立验证。GitHub API 匿名额度中途耗尽后，合入状态一律以本地提交树核对。
