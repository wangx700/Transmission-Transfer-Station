# 上一周合入 PR：变化与流程位置映射

统计窗：北京时间 2026-10-12 00:00 至 2026-10-19 00:00（左闭右开）。当前状态核对日期 2026-10-23。GitHub `merged_at` 由 UTC 转 +08:00；不能直接用 UTC 日历日期。每个合入 SHA 通过本地 `git show` 检查文件和关键 diff。

检索：在已获取的官方 main 历史中，以 sampling mask/replay、routed experts、hidden states、logprob、batch invariant/determinism 标题与 sampler/AuxOutput/determinism/docs 路径交叉检索，再用 GitHub PR API 确认合入时间。下表是与本次六项特性直接相关及明确相邻的数据正确性/CI 变更，不宣称覆盖该周所有 vLLM PR。M/I/R/H/L/B 步骤见[流程说明](特性流程说明.md)。

| PR | 合入时间（北京时间） | 流程位置 | 改动与影响边界 | 合入代码证据 |
| --- | --- | --- | --- | --- |
| [#59549](https://github.com/vllm-project/vllm/pull/59549) | 2026-10-12 19:12:31 | L1/L5 Rust 指定 ID | Rust OpenAI 请求接入 logprob_token_ids、校验与转换/返回逻辑，减少 Python/Rust 前端能力差异；不是输出完整词表。 | [提交](https://github.com/vllm-project/vllm/commit/21445806bdf6cfc4fdb9b161643e081bd73cc716) |
| [#52231](https://github.com/vllm-project/vllm/pull/52231) | 2026-10-12 23:53:27 | B2/B3 ROCm | 扩展 ROCm 的确定性 matmul/归约/通信与平台配置；不是直接给 NPU 增加支持。 | [提交](https://github.com/vllm-project/vllm/commit/e0317b28ce3ba4c9bd2ff69d885e3f34688a6d71) |
| [#55717](https://github.com/vllm-project/vllm/pull/55717) | 2026-10-13 01:05:56 | L5 全量分数返回 | 修 completion logprobs=-1 时不应按负数 limit 裁掉候选；仍受引擎与请求能力约束。 | [提交](https://github.com/vllm-project/vllm/commit/c599b5971faf7ed8bff4bd1b14713293c20082d6) |
| [#60524](https://github.com/vllm-project/vllm/pull/60524) | 2026-10-14 03:43:40 | B3 wNa16 Linear | BI 下保留 checkpoint packed layout，解包/反量化权重后用 matmul_batch_invariant，绕过 Marlin 的 M 相关 tile/K-split；有性能/内存成本。 | [提交](https://github.com/vllm-project/vllm/commit/b33cb8123707977aed605937a772bf0bc5d2a92d) |
| [#59985](https://github.com/vllm-project/vllm/pull/59985) | 2026-10-14 03:43:54 | B3 gate GEMM | BI 下 GateLinear 关闭按 token 数选择的专用 GEMM，回到 BI linear，避免 router logits 在进入 top-k 之前已漂移。 | [提交](https://github.com/vllm-project/vllm/commit/68fc395554d2ff16f91457bc031334006339f77c) |
| [#61474](https://github.com/vllm-project/vllm/pull/61474) | 2026-10-14 07:14:19 | B4 CI 基础设施 | 自动分片 batch invariance 等 CI，提升测试可调度性；不视作生产数值特性扩展。 | [提交](https://github.com/vllm-project/vllm/commit/0da66a6e468653cd8294d5fe16831bb3d7367284) |
| [#61110](https://github.com/vllm-project/vllm/pull/61110) | 2026-10-15 00:01:15 | H2/H3 模型内部提取 | GraniteMoe/Shared 接入 EagleModelMixin、SupportsEagle3，在指定层收集 aux hidden states 并随 forward 返回，供 EAGLE3/DFlash 使用；不新增通用 HTTP 激活回传。 | [提交](https://github.com/vllm-project/vllm/commit/9e2b579aabe9f289791867de539f999c188b63cd) |
| [#59518](https://github.com/vllm-project/vllm/pull/59518) | 2026-10-15 23:04:05 | L5 endpoint 候选裁切 | generate 保留引擎候选，由 derender 按 chat/completion 自身规则裁切；Rust 去重复 sampled 项，避免 sampled 不在 top-k 时少返回候选。 | [提交](https://github.com/vllm-project/vllm/commit/3dcfaeb4e8f6b90276c5b5f376010783ac24406c) |
| [#61161](https://github.com/vllm-project/vllm/pull/61161) | 2026-10-16 05:46:38 | 采样/验证 RNG→M/L | block verification 重访未提交位置时刷新接受/草稿随机数，避免跨轮复用导致偏采样；保持 seed/order/split 的设计目标，不等于通用 speculative BI 已完成。 | [提交](https://github.com/vllm-project/vllm/commit/9edcc7fa7ec858e13d4dc10ed968b741e2c95d6a) |
| [#61861](https://github.com/vllm-project/vllm/pull/61861) | 2026-10-16 05:48:20 | B4 CI 基础设施 | 自动分片 AMD 的 DP/RayExecutorV2/batch invariance CI；是测试调度，不是数值算法变更。 | [提交](https://github.com/vllm-project/vllm/commit/9b99ccb56bc90d9f6524f0861af274ea2af02f70) |
| [#61388](https://github.com/vllm-project/vllm/pull/61388) | 2026-10-16 08:21:25 | R3 TP/SP 行恢复 | DP=1 没有 DP metadata 时使用 batch descriptor 还原 token 数；TP shard all-gather 后裁 padding，完整 batch 避免无用 gather，恢复采集行对齐。 | [提交](https://github.com/vllm-project/vllm/commit/a8be1500bad93d3c459e435bf71194aa8062e49d) |
| [#60437](https://github.com/vllm-project/vllm/pull/60437) | 2026-10-16 18:04:16 | L5 parser 流式输出 | 带 reasoning/tool parser 的 streaming derender 逐 chunk resolve logprobs；延续 byte-fallback 上下文，隐藏 reasoning 时遵循不返回分数规则。 | [提交](https://github.com/vllm-project/vllm/commit/024c5188bf5a8a464124b36519c1e7de76648625) |
| [#62048](https://github.com/vllm-project/vllm/pull/62048) | 2026-10-17 02:36:15 | B4 CI | ROCm ngram scheduling accuracy 测试开启 batch invariance；增加回归约束，不新增所有模型的 BI 保证。 | [提交](https://github.com/vllm-project/vllm/commit/307c5844ad2d3003154f3a768540e686e37dddec) |
| [#61347](https://github.com/vllm-project/vllm/pull/61347) | 2026-10-17 10:08:08 | L5/RL 权重版本归属 | OpenAI completion/chat 按请求 opt-in 返回 weight_versions，区间 [start,end) 对齐 output token，流式终止 chunk 携带；帮助排除异步权重变化。 | [提交](https://github.com/vllm-project/vllm/commit/b3a7e2ff6e7d182cd9c230cdb04dd66043b9f413) |
| [#61996](https://github.com/vllm-project/vllm/pull/61996) | 2026-10-17 12:22:12 | M3 过滤性能 | split top-p 的拆分数量兼顾 vocab 长度；MRV2 传 has_topp_only_rows，没有纯 top-p 行时跳过多余 split pipeline；不改变 M5 支持集格式。 | [提交](https://github.com/vllm-project/vllm/commit/7b6310c109135b1b4aba42ed363697987380fd10) |
| [#61118](https://github.com/vllm-project/vllm/pull/61118) | 2026-10-18 02:53:06 | B2 编译配置 | 设置 Inductor batch_invariant，补齐仅开 vLLM 环境变量而编译归约仍未进入 BI 的缺口。 | [提交](https://github.com/vllm-project/vllm/commit/03e8300d2cac77b24f7ab6013a803369ee64c19e) |
| [#62302](https://github.com/vllm-project/vllm/pull/62302) | 2026-10-18 21:11:45 | R1–R6 文档 | 新增 routed experts 使用说明、prefix cache 与输出语义。仅文档，不是本周首次实现 R3。 | [提交](https://github.com/vllm-project/vllm/commit/371745ebd6c12f314402c6c319de2e7ef82c2981) |
| [#62305](https://github.com/vllm-project/vllm/pull/62305) | 2026-10-18 23:26:37 | L1/L5 接口文档 | 记录原生 generate 的请求/响应和 DP rank header，辅助对齐协议；无新的采样算法。 | [提交](https://github.com/vllm-project/vllm/commit/fe8f2140eeb6103e2ac01fbe2ce40dcc9dba19de) |

表注：时间范围仅为上一完整自然周（10-12 至 10-18），不是最近滚动七天，也不包含10-19至10-23新合入内容。共18项，其中运行时/接口与文档、CI分别按流程列明；文档与CI不算作新增模型数值能力。标题日期、commit author date 和 PR merged_at 可能相差一天，上表统一使用后者转换北京时间。

## 六项特性的本周结论

- **Sampling Mask**：未检出新增公共返回协议的合入；#61996 改过滤执行效率，#61161 是相邻的采样概率正确性修复。
- **Index**：#49555 当前仍 Open/Draft；本周未检出完整索引采集返回闭环合入，不把内部 indexer 优化计作完成 replay。
- **R3**：#61388 修 TP 分片行恢复；#62302 补使用和边界说明。已有主线采集链路与本周修复不能混写成“本周新增 R3”。
- **Hidden States**：#61110 只扩模型内部 aux extraction；窗口 API #57736 与 Observation #37002 当前仍 Open。
- **Logprob**：本周集中修前端、候选与流式 parser 语义，以及指定 ID 和权重版本返回；不是这些 PR 首次提供 logprob。
- **Batch 一致性**：ROCm 扩展、GateLinear/wNa16 专用路径和 Inductor 补漏；CI分片单列。speculative BI #52522 仍 Open，不能把 RNG 修复等同整个组合支持。

## 当前待合入项（不计入上周合入）

| PR | 2026-10-23 核查状态 | 对应流程与待完成能力 |
| --- | --- | --- |
| [#49555](https://github.com/vllm-project/vllm/pull/49555) | Open / Draft；merged_at=null | I：ordinary raw indexer selections 的公共采集/返回设计；范围按当前 PR 说明，不沿用旧版广泛适配承诺 |
| [#57736](https://github.com/vllm-project/vllm/pull/57736) | Open；merged_at=null | H：request-local windowed hidden-state capture/export；作者测试不能当主线验证 |
| [#37002](https://github.com/vllm-project/vllm/pull/37002) | Open；merged_at=null | H：Observation Phase 1 抽象/config，不足以自动完成 Phase 2 runner hook |
| [#52522](https://github.com/vllm-project/vllm/pull/52522) | Open；merged_at=null | B：speculative batch invariance，需单独组合验证 |
| [#54955](https://github.com/vllm-project/vllm/pull/54955) | Open / Draft；merged_at=null | B/L：DeepSeek-V4 0-diff RL 预览，DO NOT MERGE；不代表整体进入主线 |

表注：待合入状态是2026-10-23核对时点，不是上周末的历史状态；Open、Draft、关闭未合入和合入应分开统计。上述链接会随远端更新，固定摘录见 evidence。

## 证据等级

报告作者完成源码/diff与公共状态核查，未复跑 PR 作者的模型测试。PR 描述中的性能数字、模型通过率不在本报告中作为独立验收结论。涉及 Ascend 时要追加 vllm-ascend/NPU/HCCL 的独立接口与硬件验证。
