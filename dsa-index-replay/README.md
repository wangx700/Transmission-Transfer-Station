# DSA Index Replay 上游跟踪

核对日期：2026-10-23（北京时间）。主线基线：vLLM `19c0e170d2`、vLLM-Ascend `db8836752`、SGLang `f4a2df3dac`（均为 10-23 提交，本地仓库已同步）。报告依据源码差异与 GitHub PR/RFC 页面核查，未做 GPU/NPU 推理验证。

- [当前整体支持与适配报告](DSA_index_replay_status.md)：vLLM、vLLM-Ascend、SGLang 的 DSA 索引采集返回能力、限制与代码证据。
- [10-16 至 10-23 阶段变化说明](DSA_index_replay_changes_2026-10-16_to_2026-10-23.md)：间隔内已合入 PR、无变化事项与新增跟踪项。
- [PR #62915 独立说明](DSA_index_replay_PR62915_说明.md)：Profile-based indexer replay 的支持的模型布局、返回通道、运行前提与对 Ascend 适配的含义。
- [核查证据与复查方法](evidence/README.md)：PR 状态核查口径与合入 SHA 清单。

## 本期核心结论（2026-10-23）

- 三项目主线仍都没有完整的 DSA 索引采集返回链路；SGLang capturer 的 `attn_tp_size == 1` 限制仍在。
- vLLM 新增第二条公共实现 Draft：[#62915](https://github.com/vllm-project/vllm/pull/62915) Profile-based indexer replay（10-21 前后创建，57 个文件，含 indexer_capture、aux_output_connector、OpenAI 协议与 DeepSeek 模型接入），与既有 [#49555](https://github.com/vllm-project/vllm/pull/49555)（仍为 Draft）并行，收敛方向未定。
- vLLM-Ascend 合入 [#19220](https://github.com/vllm-project/vllm-ascend/pull/19220) DSpark 双向 SparseFlashMla（逻辑 per-request top-k 索引 + 持久长度 buffer）与 [#19190](https://github.com/vllm-project/vllm-ascend/pull/19190) 层间 O-projection buffer 共享，并经 [#19491](https://github.com/vllm-project/vllm-ascend/pull/19491) 移除旧 DSA-CP 文档；独立的 DSA 索引采集返回实现仍未出现。
- MoE routed-experts 返回族（R3）与 SWA bounded replay 族本期均有合入，但按既有口径不属于 DSA index replay，详见变化说明的边界小节。

- vLLM-Ascend 侧在状态报告第 5 节列出"当前可做与待办事项"：可直接依托的 NPU indexer 双路径、DSpark 稳定 buffer 契约、图与传输基础，以及按前置条件分组的 8 项待办（不等上游可先行 3 项、依赖上游收敛 2 项、同步验证 3 项）。

相关报告：[RL 六项特性汇总](../rl-replay-consistency/README.md) 中 I（Index Replay）分项与本目录对应；该报告按自然周（merged_at）口径核对，本目录按阶段间隔（本地主线差异 + PR 核查）口径，两者不混用。
