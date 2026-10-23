# Batch 不变性与训推一致性：阶段变化说明

比较时间：2026-10-10 上次分析 → 2026-10-16 本次分析，日历日期间隔 6 天，均以北京时间表述；未保存上次分析的具体时刻，因此不宣称精确间隔 144 小时。范围：SGLang、vLLM、vllm-ascend 的非量化模型；verl / Megatron 作为训练对齐依赖。

本说明回答“这段时间有什么变化、哪些 PR 已合入、哪些 PR 可以新增跟踪”。当前完整支持矩阵与待做工作见[整体报告](./sglang_vllm_ascend_consistency_report.md)。依据为本地提交树和 2026-10-16 已核查的 GitHub 状态；未运行硬件数值测试，测试数据来自 PR 作者时按其原有范围引用。

## 1. 时间区间与比较基准

表注：旧 HEAD 为 10 月 10 日报告保存的本地版本，新 HEAD 为 10 月 16 日刷新后的版本。HEAD 差异表示此次刷新范围，不能全部视为六天内首次开发的内容；verl 旧版本尤其早于比较窗口。

| 仓库 | 上次 HEAD | 本次 HEAD |
| --- | --- | --- |
| SGLang | `1814ece57f` | `4e85e8ed37` |
| vLLM | `5ca26880ad` | `dd9e07a25f` |
| vllm-ascend | `53085cb4e` | `c5766cd5e` |
| verl（训练依赖） | `87c68d21` | `5b550b0c` |

## 2. 这六天的变化结论

表注：左列为上次报告口径，右列为 10 月 16 日核查后的变化。“新建 PR”“本期合入”“本次新增发现”分别表述，不混为新增支持。

| 方向 | 10 月 10 日口径 | 10 月 16 日变化 |
| --- | --- | --- |
| Ascend Qwen3.5 | 未列入 BI 验证清单 | #18449 文档更新已合入：35B-A3B 在 A2/A3 已验证，950PR/950DT 未支持；没有新增运行时代码 |
| Ascend 训推定位工具 | 已有 selected logprob，Ascend trace replay 尚未列入报告 | #15604 本期合入：固定 decode token 历史并保留真实 logprob，具备更直接的逐步对齐工具 |
| MLA / V4 入图 | 有图接口及量化 V4 FULL decode 测试，非量化严格 BI 未验收 | #18750 / #18767 修复 padding metadata 和 workspace 正确性；非量化 V4 整体 BI 验收结论不变 |
| vLLM Router | 通用 BI 已支持，但特殊 Router 可能绕过替换 | #59985 本期合入，BI 下关闭 M 相关专用 GEMM；新增若干非量化 MoE 模型的 PR 级验证证据 |
| V4 0-diff 方案 | 主要跟踪 #54955 / #7050 / #7688 三个预览/集成 PR | 三者仍 Open、head 未变；本次补齐具体拆分 PR，不能表述为整套方案已进入主线 |
| 编译一致性 | 上次未单列 Inductor 风险 | 新增跟踪 #60074，窗口内新建 #61118；归约自动调优可能使 BI 开关不足以保证 compile 前向一致 |
| SGLang GDN | 有确定性模式，特殊状态与分块仍需验收 | 窗口内新建 #44076 / #44194 / #44438，分别补分块绝对对齐、固定 tile 和 P/D backend 校验，仍 Open |
| 生产调度组合 | Ascend 不支持 BI 与分块、缓存、抢占组合 | 限制仍在，本期未发现可将其改成整体支持的依据 |

## 3. 本期合入及相关进展


表注：比较区间为上次保存的 HEAD 到本次 HEAD，在线 PR 状态截至 2026-10-16。“已合入”由本地主线提交确认；“新增跟踪”表示上次报告未列出，并不表示 PR 都在 10 月 10 日之后创建。量化验证结果不计入非量化支持结论。

| 方向 | PR / 改动 | 本期结论及 Ascend 工作 |
| --- | --- | --- |
| Ascend 模型验证扩展 | [#18449](https://github.com/vllm-project/vllm-ascend/pull/18449)，已合入 | 文档新增 Qwen3.5-35B-A3B：A2/A3 支持，950PR/950DT 未支持。此次仅改文档，不能表述为新增运行时实现；分块、缓存、抢占限制仍在 |
| Ascend 固定 token 回放 | [#15604](https://github.com/vllm-project/vllm-ascend/pull/15604)，已合入 | MRV2 支持 `--enable-trace-replay` 和 `trace_decode_token_ids`；空 batch 跳过 kernel。可固定 decode 历史并取得真实模型 logprob，方便定位训推差异；不自动消除差异 |
| Ascend 入图正确性 | [#18750](https://github.com/vllm-project/vllm-ascend/pull/18750)，已合入 | V4.1 `replay_start` 按 FULL 图 padding 后请求数分配，清零补齐行，仅有有效 replay 边界时注入 metadata。原问题复现使用 W8A8，不能据此宣布非量化 V4 一致性通过 |
| Ascend 图 workspace | [#18767](https://github.com/vllm-project/vllm-ascend/pull/18767)，已合入 | 修复 DCP FIA 在 draft capture 中的 workspace 所有权；应纳入图回放与并行组合回归，不等于新增确定性 Attention |
| vLLM Router | [#59985](https://github.com/vllm-project/vllm/pull/59985)，已合入 | BI 下禁用按 token 数选择的专用 GateLinear GEMM，回到 BI linear；PR 报告非量化 GLM-4.5-Air、MiniMax-M2、ERNIE-4.5-21B-A3B 在 GB300 各 22/22 通过。Ascend 核查自身 router 是否绕过 BI 替换，不能直接继承 GPU 验收 |
| vLLM 通用 BI 覆盖 | [#52231](https://github.com/vllm-project/vllm/pull/52231)，已合入 | 扩展 ROCm；其中 `.out` Matmul 注册、softmax dtype、通信固定归约是跨平台核查点。Ascend 需要核对自己的 aten 覆盖与通信实现，不能直接套用 ROCm 配置 |
| V4 专项拆分 | [#60442](https://github.com/vllm-project/vllm/pull/60442)～[#60454](https://github.com/vllm-project/vllm/pull/60454)、[#59925](https://github.com/vllm-project/vllm/pull/59925)、[#59929](https://github.com/vllm-project/vllm/pull/59929)，仍 Open | 相比只跟总预览，已能按算子分工；本次新增跟踪，创建时间早于 10 月 10 日，本期更新了描述/验证信息。原 P2 总包 #59883 已关闭但未合入 |
| SGLang 确定性补漏 | [#44076](https://github.com/sgl-project/sglang/pull/44076)、[#44194](https://github.com/sgl-project/sglang/pull/44194)、[#44438](https://github.com/sgl-project/sglang/pull/44438)，均 Open | 最近六天新建：GDN 分块绝对位置对齐、varlen 输出固定 tile、分别校验 P/D backend。说明“开了确定性模式”仍需检查特殊路径；Ascend 对应做 GDN 和 P/D 路径回归 |

V4.1 另有 [Ascend #17742](https://github.com/vllm-project/vllm-ascend/pull/17742)（encoder SWA bounded replay）、[#17496](https://github.com/vllm-project/vllm-ascend/pull/17496)（decoder bounded replay）合入；SGLang 合入 [#43736](https://github.com/sgl-project/sglang/pull/43736)、[#43812](https://github.com/sgl-project/sglang/pull/43812)、[#44326](https://github.com/sgl-project/sglang/pull/44326) 等 BCG/DP/TRT-LLM replay 支持。**这里的 bounded replay 是模型缓存/状态重算功能，不是固定生成 token 的 trace replay，也不是 batch 不变性验收。**

## 4. 原有重点 PR 在间隔内的状态

表注：比较上次核查与 2026-10-16 当前状态。前三项 head 与旧记录相同；其余项没有本期新提交证据，不把评论或更新时间当作实现进展。

| PR | 上次 → 当前 | 变化判断 |
| --- | --- | --- |
| [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955) | Open → Open，head `4b5c627c23e6` | 总预览没有确认新的实现提交，继续作方案索引 |
| [Megatron #7050](https://github.com/NVIDIA/Megatron-LM/pull/7050) | Open → Open，head `327a23823944` | 训练端预览未合入，无本期实现推进证据 |
| [verl #7688](https://github.com/verl-project/verl/pull/7688) | Open → Open，head `0c849d861753` | recipe 仍是预览集成，不能当正式非量化闭环 |
| [verl #8026](https://github.com/verl-project/verl/pull/8026)、[#8025](https://github.com/verl-project/verl/pull/8025) | Open → Open | actor 前向 / logprob 公式和原始 mismatch 指标仍待合入 |
| [vLLM #52522](https://github.com/vllm-project/vllm/pull/52522) | Open → Open | 推测解码 BI 仍待合入，MTP 不自动获得支持 |

## 5. 当前新增跟踪项


表注：截至 2026-10-16，下表均 Open。“新增”指加入本报告；V4 拆分 PR 大多创建于 10 月 3～7 日。修改内容来自当前 PR 描述，不能当作主线已具备能力。优先级以非量化 Ascend 适配相关度排序。

| 优先级 / 方向 | PR | 修改内容 | Ascend 对应工作与边界 |
| --- | --- | --- | --- |
| P0：V4 sparse prefill | [vLLM #60442](https://github.com/vllm-project/vllm/pull/60442) | 每个请求独立决定 gather 尺寸、分块规划和 Top-K 顺序 | 检查 NPU sparse prefill 是否受同批其他请求长度影响；PR 尚无独立全模型验收 |
| P0：V4 入图 metadata | [vLLM #60443](https://github.com/vllm-project/vllm/pull/60443) | BI 使用预分配的最大 C128 宽度；处理 graph capture 的 boundary flag | 检查固定 capacity、padding、buffer 地址和回放更新；PR 报告 H20 四项定向测试通过，不是全模型 BI |
| P0：V4 确定性选择 | [vLLM #60453](https://github.com/vllm-project/vllm/pull/60453) | score + 请求内 index 的复合键，统一 P/D tie-break 和输出顺序；不支持的 BI DCP/PCP 显式拒绝 | NPU Indexer 要实现等价选择规则；不能认为该方案已覆盖 PCP/DCP |
| P0：V4 decode 索引 | [vLLM #60444](https://github.com/vllm-project/vllm/pull/60444) | C4/C128 压缩 KV 与 SWA 的有序索引、有效长度组装 | 核对 NPU 索引排列与无效槽屏蔽；该 PR 只提供原语，不切换 decode 路由 |
| P0：V4 Sparse MLA decode | [vLLM #60454](https://github.com/vllm-project/vllm/pull/60454) | `VLLM_DS4_DECODE_KERNEL=sparse` 可选路由，gather 后执行 Sparse FlashMLA，paged 仍默认 | 对照 NPU P/D 的布局及归约路径；需要前述索引/metadata 配套，不代表 P/D 已数值等价 |
| P0：V4 HC / reload | [vLLM #59929](https://github.com/vllm-project/vllm/pull/59929) | 显式 mHC pre/post、固定为单 token 等价拆分，reload 后原地刷新首层派生广播；另含 FP8 o-proj | 非量化范围只跟 HC/reload；Ascend 已有 pre/post，不重复实现，检查拆分/归约及权重更新 |
| P1：V4 MoE / DeepEP | [vLLM #59925](https://github.com/vllm-project/vllm/pull/59925) | 稳定 Router、固定 rank dispatch capacity、空 rank 参加 collective、图安全 metadata，shared expert 仅执行一次 | 参考通信和聚合契约；原实现包含 FP8/DeepGEMM，不能整体作为非量化验收；作者明确多 rank 和全模型仍未测 |
| P0：compile 确定性 | [vLLM #60074](https://github.com/vllm-project/vllm/pull/60074)、[#61118](https://github.com/vllm-project/vllm/pull/61118) | 前者约束 Inductor BI/归约和 combo benchmark；后者在确定性初始化中启用 Inductor BI | 两个 PR 有交叠，要跟踪最终收敛方案，避免重复搬运；#60074 提供非量化 Qwen3-0.6B 验证，#61118 的量化模型结果不外推 |
| P1：非量化 MoE 性能 | [vLLM #60959](https://github.com/vllm-project/vllm/pull/60959) | Hopper BF16/FP16 BI MoE 固定 tile，维持与 batch 大小无关 | 参考“先固定数值规则再调性能”的思路；CUDA tile/性能百分比不移植为 NPU 结论 |
| P1：GDN 混合 P/D 与缓存 | [vLLM #55868](https://github.com/vllm-project/vllm/pull/55868) | 缓存命中/分块采用共同对齐网格，仍 WIP 且依赖其他工作；其引用的混合 P/D #49827 已关闭未合入，只作历史参考 | 对照 Qwen3.5 的状态边界和 cache hit。文档单模型支持不意味着这些生产组合可用 |
| P0：SGLang GDN | [#44076](https://github.com/sgl-project/sglang/pull/44076)、[#44194](https://github.com/sgl-project/sglang/pull/44194) | 前者把分块绝对结束位置对齐 64 token，含缓存命中与续块；后者 varlen prefill 输出固定 64 tile | Ascend 使用自身 GDN 实现，需要验证同样的边界问题是否存在，不能直接认定同一个 bug |
| P1：SGLang 配置校验 | [#44438](https://github.com/sgl-project/sglang/pull/44438) | 分别验证显式 prefill/decode backend 的确定性支持，防止覆盖总 backend 后漏校验 | 审核 Ascend 实际 P/D backend 和专用算子；该 PR 仍注明部分后续自动选择路径待补 |
| P2：Mamba2 | [SGLang #43611](https://github.com/sgl-project/sglang/pull/43611) | 多请求 prefill 按 scan chunk 对齐，状态映射回真实序列边界 | 可作为后续混合模型回归参考；作者明确 cache 开启时仍存在 logprob 差异，不能声明完整解决 |
| P2：Ascend GDN 状态恢复 | [#18765](https://github.com/vllm-project/vllm-ascend/pull/18765) | checkpoint + correction records 替代每 draft 位置完整 SSM 快照，提交已接受前缀状态 | 只作为 MTP 状态正确性依赖；当前限单设备、eager MRV1、dense Qwen3.5 等配置，作者明确不承诺跨 batch bitwise 不变性，也未启用引擎图路径 |

[V4 P2 原总包 #59883](https://github.com/vllm-project/vllm/pull/59883) 已关闭且未合入，具体工作改跟 #60442/#60443/#60444/#60453/#60454；关闭不能写成“已支持”。[P1 #59863](https://github.com/vllm-project/vllm/pull/59863) 是 FP8 activation quantizer，保留为依赖背景，不纳入非量化待做主清单。[SGLang #35541](https://github.com/sgl-project/sglang/pull/35541) 已关闭且未合入，不作为有效待合入项。

## 6. 跟踪项的时间性质

表注：日期为 PR 创建日期按北京时间归类。V4 拆分 PR 创建于窗口之前，是本次补充发现的跟踪项；本期描述和测试信息更新不能等同于所有代码都在本期新增。

| 分类 | PR | 应如何理解 |
| --- | --- | --- |
| 窗口内新建 | vLLM #61118（10 月 12 日）；SGLang #43611（10 月 11 日）、#44076（10 月 13 日）、#44194（10 月 14 日）、#44438（10 月 15 日）；Ascend #18765（10 月 15 日） | 新提出的编译/GDN/校验/状态恢复改动，当前仍 Open |
| 窗口开始当日创建 | vLLM #60959（10 月 10 日） | 上次具体分析时刻未保存，归入本次新增跟踪，不断言当时尚未创建 |
| 窗口前已存在，本次补充跟踪 | vLLM #60442/#60443/#60444/#60453/#60454、#59925/#59929、#60074、#55868 | 不能写成“过去六天新建”；应追踪其当前实现、依赖和最终合入 |
| 关闭未合入 | vLLM #59883、#49827；SGLang #35541 | 不列为已支持；#59883 的工作改跟拆分 PR，其余保留历史参考 |

优先跟踪 V4 图 metadata / Top-K / Sparse MLA、Inductor 确定性和 SGLang GDN 修复。Ascend 先做非量化 V4 普通 decode 的 eager/graph、BS1/BSN、同图档/跨图档比较，再扩展训练重算和 reload。固定 token 回放只固定历史，最终仍要验 logits/logprob。
### 本次代码证据与验收边界

- `Ascend 模型验证与调度限制（vllm-ascend/docs/source/user_guide/feature_guide/batch_invariance.md:184）`：限制未解除，Qwen3.5 的硬件矩阵见同文件第 208 行。
- `Ascend trace token 接口（vllm-ascend/vllm_ascend/worker/v2/sample/trace_replay.py:23）` → `MRV2 patch 接入（vllm-ascend/vllm_ascend/patch/worker/patch_v2/patch_triton.py:80）`：固定历史但仍需比较模型真实 logprob。
- `FULL graph padded replay metadata（vllm-ascend/vllm_ascend/models/deepseek_v41/engram/model_state.py:153）`：按 padding 后请求数更新并清零；属于 V4.1，不能当作 V4 稀疏算子 BI 实现。
- `vLLM GateLinear BI guard（vllm/vllm/model_executor/layers/fused_moe/router/gate_linear.py:86）`：禁用 M 相关快速路径 → 回到 BI linear。
- `SGLang backend 校验（sglang/python/sglang/srt/arg_groups/overrides.py:1051）`与`RL 目标入口（sglang/python/sglang/srt/arg_groups/attention_hook.py:585）`：当前主线与 #44438 拟补校验应区分。
- `Ascend DS4 Router（vllm-ascend/vllm_ascend/models/deepseek_v4/model.py:410）`、`HC pre/post（vllm-ascend/vllm_ascend/models/deepseek_v4/model.py:745）`：已有 FP32 输入和显式 HC，不等于这些 kernel 的全部 BI 数值规则已经验收。
- 本次 GitHub 状态、head 和 PR 描述快照：`consistency_refresh_20261016.json（.pr-analysis/consistency_refresh_20261016.json）`。未运行硬件测试，表中性能/通过项均注明来自 PR 作者的验证。

