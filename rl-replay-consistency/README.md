# vLLM RL 数据回传、Replay 与 Batch 一致性

核对日期：2026-10-23（北京时间）。主线基准：[432b1003f8ea7c871750f27bc82815a7ad0252d5](https://github.com/vllm-project/vllm/commit/432b1003f8ea7c871750f27bc82815a7ad0252d5)。本报告以源码、GitHub PR 合入记录为证据，未做 GPU/NPU 推理测试。

上一周定义为北京时间 **2026-10-12 00:00 至 2026-10-19 00:00，左闭右开**，对应 UTC `[2026-10-11T16:00:00Z,2026-10-18T16:00:00Z)`。按 GitHub `merged_at` 判定，不按创建时间或描述更新时间。

- [特性接口、数据格式与流程说明](特性流程说明.md)
- [上一周合入 PR 与流程位置映射](上一周PR与流程映射.md)
- [详细数据格式与 token 对齐说明（10-21 固定版本附录）](数据格式参考_2026-10-21.md)
- [核查方法与原始 PR 状态摘录](evidence/README.md)

## 六项特性汇总

| 编号 / 特性 | RL 目的与记录对象 | 当前主线 / 主要接口 | 数据格式与训练端消费 | 当前限制与上周变化 |
| --- | --- | --- | --- | --- |
| M：Sampling Mask | 对齐 rollout 与训练的采样动作空间，记录过滤后的支持集 S | 已有 MRV2 采集返回；`return_sampling_mask=True` + `logprobs_mode=processed_logprobs` | `SamplingMask.token_ids: list[list[int]]`，外层对齐 completion token；trainer 在 S 内归一化 | `temperature>0`、`top_k>0`；adaptive verification 等不支持。#61996 优化上游 top-p 过滤执行，不是新增 mask API |
| I：Index Replay | 固定 sparse attention 实际选择的 KV 位置 | 主线有 indexer；完整公共采集返回仍待合入；[#49555](https://github.com/vllm-project/vllm/pull/49555) Open/Draft | 不能将内部 top-k buffer 当作公共返回；训练侧还需生产层/消费层、坐标、有效宽度映射 | 本周核查时仍未合入；上周未检出完整 Index Replay 上线 PR |
| R：R3 / Routed Experts | 固定每个输入 token 的 MoE 专家选择 | 已有；`enable_return_routed_experts=True`，AuxOutput 按 KV hash 保存 | ndarray `[P+G-1-O,L,K]`，导出 uint8/uint16；HTTP 为 base64 `.npy`；trainer 复用逻辑专家 ID | L 是总 hidden-layer 轴，非紧凑 MoE 层表；MRV2/prefix cache 等限制。#61388 恢复 DP=1、TP 分片采集；#62302 补文档 |
| H：Hidden States | 蒸馏、辅助模型和激活诊断 | 已有 EAGLE3/DFlash 等模型内部提取；普通生成逐请求窗口回传不能据此认定已支持 | 内部通常为 hidden tensor 与 aux tensor 列表，由模型/proposer 使用；不同模型层边界不同 | #61110 给 GraniteMoe/Shared 加内部 aux hidden states；[#57736](https://github.com/vllm-project/vllm/pull/57736) 窗口回传仍 Open；[#37002](https://github.com/vllm-project/vllm/pull/37002) 仍 Open |
| L：Logprob | 提供 PPO/GRPO ratio、KL、teacher scoring 和训推偏差诊断的分数 | 已有输出/prompt/指定 ID 打分；`logprobs`、`prompt_logprobs`、`logprob_token_ids` 等 | 离线 `logprobs[i][token_id].logprob`；原生 token-mode HTTP `logprobs.content`；raw/processed 语义必须一致 | #55717/#59518/#60437/#59549 修复全量/候选裁切/流式 parser/Rust 指定 ID 等返回环节；Trace Replay 可固定轨迹诊断 |
| B：Batch 一致性 | 减少同请求因 batch 大小、同批伙伴和顺序变化产生的数值漂移 | 已有 beta；`VLLM_BATCH_INVARIANT=1` | 不新增 replay 数据字段；稳定 GEMM、attention、归约及通信后比较 logits/logprob | #52231 ROCm、#59985 GateLinear、#60524 wNa16、#61118 Inductor；不保证跨硬件/TP 度/量化/框架一致；#52522 speculative BI 仍 Open |

表注：状态截至 2026-10-23 本次核对；P=实际 prompt token 数，G=实际输出 token 数，O=`routed_experts_prompt_start`，L=模型总 hidden layers，K=experts per token。R 的 dense/未绑定层槽不能解释为有效专家记录。当前主线“已存在”和“本周新增”分开表述；H 的内部提取不是生成 API 激活导出。上周变化严格按上述时间窗。源码入口与步骤编号见流程文档，各 PR 的时间、证据与改动环节见周报。

## 使用顺序

先锁定权重/tokenizer/采样参数/分数模式，再用固定 token 轨迹定位数值偏差；按需求加入 B（稳定计算）、R（专家选择）和 M（支持集）。I/H 的外部数据回传按未合入状态单独推进。每项独立存在不意味着组合已经验证。

SWA bounded / decoder replay 是 prefix 命中后的滑窗 KV 重建与模型后段重算，作为流程文档的边界说明保留；它不返回 trainer 使用的 R/M/I 数据。不展开 CUDA Graph Replay 和 Vision Encoder Graph Replay。
