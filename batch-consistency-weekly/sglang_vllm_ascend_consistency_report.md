# SGLang / vLLM / vllm-ascend 一致性支持与适配汇总

更新日期：2026-10-23（北京时间）。当前状态快照；阶段变化单独见[10 月 16 日至 23 日变化说明](./consistency_changes_20261016_20261023.md)与[10 月 10 日至 16 日变化说明](./consistency_changes_20261010_20261016.md)。范围：非量化模型的 batch 一致性和训推一致性；本次已 fetch 并快进更新相关仓库，结合 GitHub PR 最新状态静态核查，未运行 GPU/NPU 数值测试。PR 作者报告的测试结果与本次实测严格区分。

## 1. 核心结论

- **SGLang**：已有确定性推理、请求 seed 和训练目标匹配入口；部分 GPU backend 支持分块、缓存及图模式。训练侧仍需使用匹配的算子，不能把推理确定性视为通用训推一致性。
- **vLLM**：已合入 GateLinear 的 BI 路径修复及 ROCm 支持；**10-17 起主线在确定性初始化中启用 Inductor BI（#61118），BI 验证名单新增 Gemma-2、Granite (MoE) SWA、Qwen3-0.6B（#61254），cooperative TopK 溢出改精确选择（#59645）**。V4 已有拆分后的 sparse prefill、图 metadata、Top-K、decode、MoE/DeepEP、MHC PR，但这些专项 PR 仍未合入，#60074（pin Inductor kernel 配置）仍 Open。
- **vllm-ascend**：Qwen3.5-35B-A3B 已列入 A2/A3 验证名单；已有 MRV2 trace decode replay、FULL 图 padding metadata 修复；**10-16 至 10-23 主线 BI 面零代码改动，窗口内合入集中在 V4.1 bounded replay 跨 runner/PCP 扩展、PCP 图回放 padding、GDN 状态索引与 MTP 正确性**。DS4 稀疏 Attention/Top-K、HC、MoE 的完整数值契约、动态调度兼容及训练侧共享前向仍待验收。

表注：以下是检查时本地 HEAD，日期为提交日期。“支持”须区分实现入口、文档验证及实际数值验收。本文不承诺跨硬件、跨版本或跨 TP 数量逐 bit 一致。

| 仓库 | 上次 HEAD | 本次 HEAD | 本次提交日期（北京时间） |
| --- | --- | --- | --- |
| SGLang | `4e85e8ed37` | `fa628c5359` | 2026-10-23 |
| vLLM | `dd9e07a25f` | `18343393b4` | 2026-10-23 |
| vllm-ascend | `c5766cd5e` | `db8836752` | 2026-10-23 |
| verl（外部训练依赖） | `5b550b0c` | `bfff4d90` | 2026-10-23 |

本次结论以提交树为准。上期核查时 `vllm_ascend/attention/sfa_v1.py` 的未提交修改现已不在工作区（本窗口主线也未改动该文件），相关图接口结论以主线版本为准。

## 2. 模型支持情况

表注：vLLM/Ascend 的“已验证”来自本地 batch-invariance 文档；SGLang 文档提供 backend 支持及模型示例，没有与 vLLM 完全同口径的模型验收清单。“未确认”不表示模型无法运行。量化模型及其验证结果不纳入本表。

| 模型 | SGLang | vLLM | vllm-ascend | 核心待做工作 / 对应 PR |
| --- | --- | --- | --- | --- |
| Qwen3 Dense / MoE | 确定性文档提供 Qwen3-8B、30B-A3B 示例；训练端需另行对齐 | 1.7B、8B、30B-A3B 已列入验证 | 1.7B、8B、30B-A3B、235B-A22B 已列入；有缩小模型 TP4 CI | 用 Dense 建立训推基线，再补真实 MoE TP/EP；未确认专属非量化模型 PR，跟踪通用训练基础 **verl #8026 / #8025** |
| DeepSeek V3 / R1 / V3.1 | 确定性配置针对 absorbed-MLA 模型校验 backend；不能推导全部组合已验证 | V3、V3-0324、R1、V3.1 已列入验证 | 未列入一致性验证名单 | MLA、router、专家聚合与不同 batch 的 logprob；未确认专属 PR，跟踪 **vLLM #27433、SGLang #10278**（均为 Issue） |
| DeepSeek V3.2 / V4 / V4.1 | 有 DSA/模型支持路径；本次未确认非量化完整训推验收 | 不在上述验证名单；V4 有专项 batch 不变性预览 PR（#54955），未确认主线完整纳入 | DS4/V4.1 已注册，有 indexer、HC、Sparse Attention；未确认完整非量化验收 | 优先 Top-K、P/D、HC、MoE；DS4 跟踪 **vLLM #54955 / Megatron #7050 / verl #7688**，不能自动覆盖 V3.2/V4.1 |
| Qwen3-Next / Qwen3.5 | 有确定性路径，但 GDN 分块和小 batch tile 仍有待合入修复 #44076 / #44194 | Next-80B-A3B 已列入；Qwen3.5 未明确列入该文档清单；混合 P/D、缓存参考 #49827（已关闭未合入）并跟踪 #55868 | **Qwen3.5-35B-A3B 已列入 A2/A3 验证；950PR/950DT 未支持**；不能推广到全系列 | 优先 GDN 状态、分块和缓存回归；Ascend #18449 已合入，#18765 是 eager MTP 状态恢复方案，不承诺 BI |
| Qwen3-VL | 官方 RL 文档指向 VLM true-on-policy 集成；不能视为所有 VL 模型已通过 | 2B/4B 单图和视频已列入 batch 验证 | 未列入一致性验证名单 | 视觉编码器、视觉 token 布局和训练侧共享前向；未确认这两个模型的专属 PR |
| Qwen2.5 / Llama 3 | Llama 有确定性示例；其余具体组合需验收 | 多个规模已列入验证 | 未列入一致性验证名单 | 优先复用现有算子做模型回归；未确认专属 PR |
| GLM-4.5-Air / MiniMax-M2 / ERNIE-4.5-21B-A3B | 本次未确认对应完整非量化验收 | #59985 已合入，PR 提供 GB300 的模型级 BI 测试结果，验证范围受硬件/TP 配置约束 | 本次未确认模型级 BI 验收 | 跟踪 GateLinear 数值路径；训练端是否共享同一前向仍需验证 |

证据：`SGLang 确定性文档（sglang/docs/docs/advanced_features/deterministic_inference.mdx:22）`、`MLA backend 校验（sglang/python/sglang/srt/arg_groups/attention_hook.py:585）`、`vLLM 模型名单（vllm/docs/features/batch_invariance.md:125）`、`Ascend 模型名单（vllm-ascend/docs/source/user_guide/feature_guide/batch_invariance.md:198）`。

Ascend 的 `Qwen3 TP4 测试（vllm-ascend/tests/e2e/pull_request/four_card/rlhf/consistency/test_batch_invariant_tp4.py:32）` 使用两层、dummy 权重，是基础覆盖，不是完整模型训练验收。DS4 部分测试使用量化模型，不能作为本报告非量化支持的证明。

### DeepSeek-V4“专用预览”的含义

这里指 [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955) 提供的 **DeepSeek-V4 专项 batch 不变性适配方案**。“专用”表示 V4 的 Indexer、Sparse MLA、MHC、MoE 等特殊计算路径需要额外约束，通用 Matmul/Attention 的确定性替换不足以证明整条 V4 前向链路满足 batch 不变性。

该方案的接口流程可以概括为：`输入 hidden states → Indexer/Top-K 选择稀疏 KV → Sparse MLA Attention → HC pre/block/post → Router/专家计算与 combine → logits/logprob`。这是相关计算环节的概括；实际层内 Attention 和 MoE 各自有 HC 包装，不表示整个模型只有一次 HC 调用。

重点适配内容包括：

- **Matmul / Router**：固定 split-K 规划和部分结果的累加顺序，避免根据 batch 的 M 维切换数值路径；router 使用统一的 FP32 计算路径。
- **Indexer / Top-K**：固定同分候选的选择规则、索引排列及 prefill/decode 选择语义。
- **Sparse MLA**：对齐 compressed/SWA KV 的组织顺序、prefill/decode 计算路径，以及请求局部的执行规划和 workspace。
- **MHC / HC**：显式拆分 pre/block/post，并约束 token 拆分和归约方式。
- **MoE**：约束专家容量、跨 rank staging、专家输出聚合顺序和 shared expert 的执行方式。

“预览”表示截至 **2026-10-16**，该 PR 仍为 Open，标记 **Preview / DO NOT MERGE**；head `4b5c627c23e6` 与上次核查相同。不能认定非量化 V4 的 batch 不变性或训推一致性已完成验收。训练侧配套 [Megatron-LM #7050](https://github.com/NVIDIA/Megatron-LM/pull/7050)、[verl #7688](https://github.com/verl-project/verl/pull/7688) 也仍 Open，head 与上次记录相同；其中量化路径的结果不作为本报告非量化结论。现在应以拆分 PR 作为具体实施跟踪入口，总预览保留作方案索引。

HC 指 Hyper-Connections，mHC 指 Manifold-Constrained Hyper-Connections，负责多路残差的混合：`多路残差 → HC pre → Attention/MoE → HC post → 多路残差`。一致性重点是混合、归一化和归约顺序，不是 KV Cache。

**对 vllm-ascend 的含义**：已有 DS4 模型与专用算子实现，但仍需逐项核对上述数值规则是否落实到 NPU 路径，并验证不同 batch、prefill/decode、TP/EP 以及权重更新后的 logits/logprob。GPU kernel 无需照搬，需在 NPU 上实现等价的选择、布局、归约和聚合规则。该预览也不能自动证明 V3.2/V4.1 已支持。

## 3. 核心特性支持与缺口

表注：SGLang 的 GPU backend 能力不等同于 Ascend backend 能力；本表不把“模型可运行”计为一致性支持。PR 编号的完整链接见下一节。

| 特性 | SGLang | vLLM | vllm-ascend | 需要做的工作 / 跟踪项 |
| --- | --- | --- | --- | --- |
| 通用 batch-invariant | 有确定性模式和算子路径 | 有集中确定性模块 | 有 AscendC/Triton 注册 | 核查专用算子是否绕过通用替换；跟踪 SGLang #10278、vLLM #27433（Issue） |
| 分块、prefix cache、抢占重算 | 文档列出 FA3/Triton 分块、Radix Cache、图模式支持；FlashInfer 不支持该矩阵中的 Radix Cache；抢占需单独验收 | backend/模型相关，不能统一保证 | **文档明确不支持与 BI 组合** | 请求局部布局、P/D 归约及重算边界；DS4 参考 **#54955** |
| 训推前向与 logprob 对齐 | 有 `rl_on_policy_target`，影响确定性配置和 logprob 公式；训练侧需配套 | 有 logprob、token 回放及 head_dtype；不是任意训练端的完整保证 | **已有 NPU selected-logprob 并接入 V2**；训练侧完整共享未确认 | 共享 NPU 前向、梯度与概率公式；**#8025 度量已合入（10-22）**，跟踪 **#7050 / #8026** |
| DS4 专用算子 | 非量化全链路验收未确认 | 专用方案在 **#54955** 预览中 | 有显式 HC pre/post、FP32 router 输入、indexer/Sparse Attention；等价 BI 契约未确认 | 固定 Top-K tie-break、KV 顺序、HC 归约、专家容量与 combine；跟踪 **#54955 / #7050** |
| 并行与推测解码 | 有相关运行路径；不能推导跨 TP/EP 或任意 speculative 一致 | 推测解码 BI 有 **#52522** | TP4 基础 CI、MTP/DSpark 实现；不等于严格一致 | 固定 RNG、通信聚合、draft/verify、恢复边界；MTP 单独验收 |
| 编译后归约 | 特殊 kernel 仍有 tile 补漏 PR | **#61118 已合入（10-17）**：确定性初始化启用 Inductor BI，另有 ROCm guard #62274；#60074 仍 Open，拟 pin Inductor kernel 配置，与 #61118 互补 | 本次未确认等价编译约束和覆盖验收；torch-ascend 是否具备同配置待查 | 检查 eager 与 compile 的算子分派、tile 和归约；不把 CUDA Graph 开关与编译确定性混为一谈 |
| MLA / Sparse MLA 入图 | backend 与模型相关；BCG replay 功能不等于 BI | V4 图 metadata 和 sparse decode 分别跟踪 #60443 / #60454 | 已有 SFA 图接口和 FULL decode 测试；padding/workspace 修复之外，**10-16~10-23 bounded replay 扩展到跨 runner 与 PCP（#18936/#19034/#19192/#19321）**，尚缺非量化 V4 严格验收 | 固定输入历史，比较 eager/graph、同图档与跨图档的 Attention 输出、logits/logprob |

核心源码入口：

- SGLang：`训练目标触发确定性（sglang/python/sglang/srt/arg_groups/attention_hook.py:585）`、`sampler 的目标匹配公式（sglang/python/sglang/srt/layers/sampler.py:121）`。
- vLLM：`确定性初始化（vllm/vllm/model_executor/determinism/batch_invariant.py:1429）`、`固定 token 回放（vllm/vllm/v1/worker/gpu/sample/trace_replay.py:14）`。
- Ascend：`算子注册（vllm-ascend/vllm_ascend/batch_invariant.py:116）`、`调度限制（vllm-ascend/docs/source/user_guide/feature_guide/batch_invariance.md:184）`、`NPU logprob（vllm-ascend/vllm_ascend/worker/v2/sample/logprob.py:68）`、`HC pre/post（vllm-ascend/vllm_ascend/models/deepseek_v4/model.py:745）`。

## 4. 对应待跟踪 PR

表注：状态核查截至 2026-10-16。Open 代表未合入；Merged 代表其所属仓库已合入，不代表 Ascend 已适配。训练框架 PR 在此作为推理对齐的外部依赖，不改变三个推理框架的比较主体。总预览、训练依赖与专项适配 PR 分别列示；阶段内状态变化见独立变化说明。

| PR | 状态 | 模型 / 特性 | 跟踪内容 |
| --- | --- | --- | --- |
| [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955) | Open；Preview / DO NOT MERGE | DeepSeek-V4 | Matmul/router、确定性 Top-K、Sparse MLA、MHC、MoE、reload；关注后续拆分 PR |
| [Megatron-LM #7050](https://github.com/NVIDIA/Megatron-LM/pull/7050) | Open；Preview / DO NOT MERGE | DS4 训练端 | 共享前向、backward、packed batch、CP/EP、权重同步；无本地仓库，需补核查 |
| [verl #7688](https://github.com/verl-project/verl/pull/7688) | Open | DS4 对齐 recipe | 依赖集成、初始及更新权重后的两步验收 |
| [verl #8026](https://github.com/verl-project/verl/pull/8026) | Open | 通用训练前向 / logprob | actor BI 开关、sampler 公式；NPU 需等价实现 |
| [verl #8025](https://github.com/verl-project/verl/pull/8025) | **Merged，2026-10-22** | 一致性度量 | rollout vs actor raw logprob mismatch 指标已进入主线（mismatch_count 窗口求和、diff_max、one-ULP 掩蔽）；**NPU 接入等价指标为当下动作** |
| [vLLM #52522](https://github.com/vllm-project/vllm/pull/52522) | Open | 推测解码 BI | 随机推测解码、抢占恢复；不能直接推导 MTP 支持 |
| [slime #2262](https://github.com/THUDM/slime/pull/2262) | Merged，2026-08-11 | SGLang / GLM-5 训练对齐参考 | 关注请求/路由顺序、DSA、通信聚合契约；原方案含量化，不作为非量化 0-diff 证明 |

补充路线跟踪：[SGLang #10278](https://github.com/sgl-project/sglang/issues/10278)（现为 closed，早已关闭，非本期变化）、[vLLM #27433](https://github.com/vllm-project/vllm/issues/27433)、[Ascend #5487](https://github.com/vllm-project/vllm-ascend/issues/5487)、[vLLM #48305](https://github.com/vllm-project/vllm/issues/48305)。这些是 Issue/RFC，不应误标为待合入 PR。没有确认模型专属 PR 的条目，保持“待补跟踪项”，不编造对应关系。

### 当前专项 PR 跟踪清单

表注：截至 2026-10-16，下表均 Open。V4 拆分 PR 大多创建于 10 月 3～7 日；此处按当前适配任务组织。修改内容来自当前 PR 描述，不能当作主线已具备能力。优先级以非量化 Ascend 适配相关度排序。

| 优先级 / 方向 | PR | 修改内容 | Ascend 对应工作与边界 |
| --- | --- | --- | --- |
| P0：V4 sparse prefill | [vLLM #60442](https://github.com/vllm-project/vllm/pull/60442) | 每个请求独立决定 gather 尺寸、分块规划和 Top-K 顺序 | 检查 NPU sparse prefill 是否受同批其他请求长度影响；PR 尚无独立全模型验收 |
| P0：V4 入图 metadata | [vLLM #60443](https://github.com/vllm-project/vllm/pull/60443) | BI 使用预分配的最大 C128 宽度；处理 graph capture 的 boundary flag | 检查固定 capacity、padding、buffer 地址和回放更新；PR 报告 H20 四项定向测试通过，不是全模型 BI |
| P0：V4 确定性选择 | [vLLM #60453](https://github.com/vllm-project/vllm/pull/60453) | score + 请求内 index 的复合键，统一 P/D tie-break 和输出顺序；不支持的 BI DCP/PCP 显式拒绝 | NPU Indexer 要实现等价选择规则；不能认为该方案已覆盖 PCP/DCP |
| P0：V4 decode 索引 | [vLLM #60444](https://github.com/vllm-project/vllm/pull/60444) | C4/C128 压缩 KV 与 SWA 的有序索引、有效长度组装 | 核对 NPU 索引排列与无效槽屏蔽；该 PR 只提供原语，不切换 decode 路由 |
| P0：V4 Sparse MLA decode | [vLLM #60454](https://github.com/vllm-project/vllm/pull/60454) | `VLLM_DS4_DECODE_KERNEL=sparse` 可选路由，gather 后执行 Sparse FlashMLA，paged 仍默认 | 对照 NPU P/D 的布局及归约路径；需要前述索引/metadata 配套，不代表 P/D 已数值等价 |
| P0：V4 HC / reload | [vLLM #59929](https://github.com/vllm-project/vllm/pull/59929) | 显式 mHC pre/post、固定为单 token 等价拆分，reload 后原地刷新首层派生广播；另含 FP8 o-proj | 非量化范围只跟 HC/reload；Ascend 已有 pre/post，不重复实现，检查拆分/归约及权重更新 |
| P1：V4 MoE / DeepEP | [vLLM #59925](https://github.com/vllm-project/vllm/pull/59925) | 稳定 Router、固定 rank dispatch capacity、空 rank 参加 collective、图安全 metadata，shared expert 仅执行一次 | 参考通信和聚合契约；原实现包含 FP8/DeepGEMM，不能整体作为非量化验收；作者明确多 rank 和全模型仍未测 |
| P0：compile 确定性 | [vLLM #60074](https://github.com/vllm-project/vllm/pull/60074)（仍 Open）、[#61118](https://github.com/vllm-project/vllm/pull/61118)（**已合入，2026-10-17**） | 后者在确定性初始化中启用 Inductor BI（`inductor_config.batch_invariant=True`，带 hasattr 防御）；前者拟 pin Inductor kernel 配置 | 两者互补而非替代，继续跟踪 #60074 的最终收敛；Inductor 开关仅覆盖 CUDA，NPU 编译等价约束仍需单独实现与验证 |
| P1：非量化 MoE 性能 | [vLLM #60959](https://github.com/vllm-project/vllm/pull/60959) | Hopper BF16/FP16 BI MoE 固定 tile，维持与 batch 大小无关 | 参考“先固定数值规则再调性能”的思路；CUDA tile/性能百分比不移植为 NPU 结论 |
| P1：GDN 混合 P/D 与缓存 | [vLLM #55868](https://github.com/vllm-project/vllm/pull/55868) | 缓存命中/分块采用共同对齐网格，仍 WIP 且依赖其他工作；其引用的混合 P/D #49827 已关闭未合入，只作历史参考 | 对照 Qwen3.5 的状态边界和 cache hit。文档单模型支持不意味着这些生产组合可用 |
| P0：SGLang GDN | [#44076](https://github.com/sgl-project/sglang/pull/44076)、[#44194](https://github.com/sgl-project/sglang/pull/44194) | 前者把分块绝对结束位置对齐 64 token，含缓存命中与续块；后者 varlen prefill 输出固定 64 tile | Ascend 使用自身 GDN 实现，需要验证同样的边界问题是否存在，不能直接认定同一个 bug |
| P1：SGLang 配置校验 | [#44438](https://github.com/sgl-project/sglang/pull/44438) | 分别验证显式 prefill/decode backend 的确定性支持，防止覆盖总 backend 后漏校验 | 审核 Ascend 实际 P/D backend 和专用算子；该 PR 仍注明部分后续自动选择路径待补 |
| P2：Mamba2 | [SGLang #43611](https://github.com/sgl-project/sglang/pull/43611) | 多请求 prefill 按 scan chunk 对齐，状态映射回真实序列边界 | 可作为后续混合模型回归参考；作者明确 cache 开启时仍存在 logprob 差异，不能声明完整解决 |
| P2：Ascend GDN 状态恢复 | [#18765](https://github.com/vllm-project/vllm-ascend/pull/18765) | checkpoint + correction records 替代每 draft 位置完整 SSM 快照，提交已接受前缀状态 | 只作为 MTP 状态正确性依赖；当前限单设备、eager MRV1、dense Qwen3.5 等配置，作者明确不承诺跨 batch bitwise 不变性，也未启用引擎图路径 |

[V4 P2 原总包 #59883](https://github.com/vllm-project/vllm/pull/59883) 已关闭且未合入，具体工作改跟 #60442/#60443/#60444/#60453/#60454；关闭不能写成“已支持”。[P1 #59863](https://github.com/vllm-project/vllm/pull/59863) 是 FP8 activation quantizer，保留为依赖背景，不纳入非量化待做主清单。[SGLang #35541](https://github.com/sgl-project/sglang/pull/35541) 已关闭且未合入，不作为有效待合入项。

### 2026-10-16 → 10-23 新增跟踪项

表注：以下均为 Open（核查时点 2026-10-23，全部为窗口内新建）。内容来自当前 PR diff/描述；详细说明见[阶段变化文档](./consistency_changes_20261016_20261023.md)第 5 节。

| 优先级 | PR | 内容 | 边界 |
| --- | --- | --- | --- |
| P0 | [vLLM #62192](https://github.com/vllm-project/vllm/pull/62192)（关联 [#62219](https://github.com/vllm-project/vllm/pull/62219)） | BI 下 pin row-invariant GEMM 路径；跨 tuned M-bucket invariance 测试 | 检查 NPU GEMM 分桶/自动调优是否引入 M 相关数值路径 |
| P0 | [vLLM #62017](https://github.com/vllm-project/vllm/pull/62017) | Nemotron-H NVFP4 BI serving；含 mamba2 SSD BI 算子、`determinism/scope.py`、`batch_invariant_configs.py` | 模型量化不计入非量化支持；scope/configs 与 Mamba2 BI 基础设施是通用新增 |
| P0 | [vLLM #62193](https://github.com/vllm-project/vllm/pull/62193) | `batch_invariance_exempt()` 作用域：spec drafter 豁免 BI kernel 要求 | 与 #52522 组合成 speculative BI 方案，跟踪收敛 |
| P0 | [Ascend #19442](https://github.com/vllm-project/vllm-ascend/pull/19442) | TP4 BI e2e 新增 Qwen3.5-35B-A3B 4 层切片（3 GDN + 1 FA + MoE top-8，dummy 权重） | 首个混合 GDN+MoE 的 Ascend BI e2e；仍是切片基础覆盖，非完整验收 |
| P1 | [Ascend #19237](https://github.com/vllm-project/vllm-ascend/pull/19237) / [#19238](https://github.com/vllm-project/vllm-ascend/pull/19238) | DSA O-proj 交换对齐 coordinated runtime batch | 通信归约分组随 runtime batch 变化的 BI 风险点 |
| P1 | [SGLang #45063](https://github.com/sgl-project/sglang/pull/45063) | NPU batch-invariant：`torch.use_deterministic_algorithms(True, warn_only=True)` | SGLang NPU 路线尝试，可对照 vllm-ascend 算子注册路线 |
| P1 | [SGLang #44923](https://github.com/sgl-project/sglang/pull/44923) | `filter_batch` 后刷新 `is_all_greedy`/`is_any_greedy` | 动态 batch 贪心标志过期是直接不一致来源；Ascend 调度器自查同类刷新 |
| P2 | [SGLang #45599](https://github.com/sgl-project/sglang/pull/45599)、[#44861](https://github.com/sgl-project/sglang/pull/44861)、[verl #8317](https://github.com/verl-project/verl/pull/8317) | AMD split-K BI matmul；SM12x FP8 确定性；GB200 Megatron Lite + BI rollout 镜像 | 平台覆盖/量化边界/训练基础设施 |

### MLA 入图结合 DeepSeek-V4 的验证建议

**可以验证，首阶段限定为非量化 V4、普通 decode、FULL_DECODE_ONLY。** 通用 MLA 入图通过，不等于 V4 的 Sparse MLA/Indexer/HC/MoE 整体 BI 通过。已有 SFA `get_cudagraph_support()`、`build_for_graph_capture()` 入口，以及 BI=1 的 V4 FULL decode 测试，但原测试使用 W4A8，不能充当本范围的严格验收。

接口流程：`请求/固定 decode token 历史 → scheduler batch 与图档位 → metadata/padding、Indexer/Top-K、KV 更新 → Sparse MLA 图回放 → HC/MoE → logits/selected logprob`。

1. **算子层**：固定请求的 Q/KV/Top-K，改变邻居请求和 batch 位置，比较 eager 与 graph 输出。先隔离 Sparse MLA，再联动 Indexer。
2. **模型层**：固定权重、TP/EP、输入历史；借助 #15604 trace replay，对同一历史计算真实 logprob。比较 BS1/BSN、请求排列、同图档和跨图档，覆盖有 padding 的实际 batch；同时确认实际命中图回放。不要把强制 token 相同当作通过证据。
3. **边界层**：关闭 MTP/DSpark、prefix cache、chunked prefill，避免抢占，先比较逐 token logits/logprob 的 bitwise mismatch、max/mean 误差；再逐项扩展。图 padding、空 rank、workspace 生命周期是本期修复提示的重点。
4. **训推层**：上述通过后，再比较训练重算与 rollout，覆盖 optimizer 更新和 reload；推理侧入图验证本身不构成训推一致性证明。

### 本次代码证据与验收边界

- `Ascend 模型验证与调度限制（vllm-ascend/docs/source/user_guide/feature_guide/batch_invariance.md:184）`：限制未解除，Qwen3.5 的硬件矩阵见同文件第 208 行。
- `Ascend trace token 接口（vllm-ascend/vllm_ascend/worker/v2/sample/trace_replay.py:23）` → `MRV2 patch 接入（vllm-ascend/vllm_ascend/patch/worker/patch_v2/patch_triton.py:80）`：固定历史但仍需比较模型真实 logprob。
- `FULL graph padded replay metadata（vllm-ascend/vllm_ascend/models/deepseek_v41/engram/model_state.py:153）`：按 padding 后请求数更新并清零；属于 V4.1，不能当作 V4 稀疏算子 BI 实现。
- `vLLM GateLinear BI guard（vllm/vllm/model_executor/layers/fused_moe/router/gate_linear.py:86）`：禁用 M 相关快速路径 → 回到 BI linear。
- `SGLang backend 校验（sglang/python/sglang/srt/arg_groups/overrides.py:1051）`与`RL 目标入口（sglang/python/sglang/srt/arg_groups/attention_hook.py:585）`：当前主线与 #44438 拟补校验应区分。
- `Ascend DS4 Router（vllm-ascend/vllm_ascend/models/deepseek_v4/model.py:410）`、`HC pre/post（vllm-ascend/vllm_ascend/models/deepseek_v4/model.py:745）`：已有 FP32 输入和显式 HC，不等于这些 kernel 的全部 BI 数值规则已经验收。
- 本次 GitHub 状态、head 和 PR 描述快照：`consistency_refresh_20261016.json（.pr-analysis/consistency_refresh_20261016.json）`。2026-10-23 刷新：窗口内合入以本地提交树核对（vLLM `dd9e07a25f..18343393b4`、vllm-ascend `c5766cd5e..db8836752`），新跟踪 PR 的 diff 快照在 [window_20261023](./evidence/)。未运行硬件测试，表中性能/通过项均注明来自 PR 作者的验证。

## 5. 建议优先做的三件事

1. **建立 Qwen3 非量化训推基线**：复用已有 NPU 确定性前向与 logprob，补训练梯度路径；比较初始权重、optimizer 更新及 reload 后的逐 token 数值。**verl #8025 的 mismatch 指标已合入，先让 NPU rollout/actor 接入等价指标**。
2. **补 DS4 专项契约**：Top-K/Sparse Attention 优先，再做 HC 与 MoE。已有 HC 显式结构和 logprob 不重复开发；重点核查归约、索引排列和通信顺序。
3. **解锁生产组合**：分块、缓存、抢占、真实 TP/EP、图模式和推测解码逐项验收；随后扩到 Next/GDN 和 VL，记录性能成本。

验收接口流程：

```text
训练权重 → 同步到 rollout → 确定性 forward → token/logprob/所需路由索引
        → 训练侧共享前向重算 → 逐 token 比较 → backward/optimizer
        → reload 更新权重 → 再次比较
```

输出文本一致、seed 固定、BI 开关为 1 均不能代替数值验收。需要分别确认算子 batch 不变性、推理跨 batch 一致性、训练与 rollout 前向一致性；backward 逐 bit 可复现是另一个目标。

### vllm-ascend 待做事项（2026-10-23 刷新）

表注：完整清单与依据见[10 月 16 日至 23 日变化说明](./consistency_changes_20261016_20261023.md)第 7 节。"可直接启动"不依赖未合入的上游 PR；P0 三项互相独立，建议最先做。

| 优先级 | 类别 | 事项 | 对照 |
| --- | --- | --- | --- |
| P0 | 可直接启动 | 接入 verl #8025 等价的 rollout/actor raw logprob mismatch 度量，Qwen3 dense 先跑首轮统计 | verl #8025 已合入 |
| P0 | 可直接启动 | NPU GEMM/算子分派自查：实现选择与 tile 分桶是否依赖 M，补算子级跨 batch 不变性测试 | vLLM #62192 思路；`vllm_ascend/batch_invariant.py` 覆盖面 |
| P0 | 可直接启动 | V4/V4.1 图回放回归矩阵：trace replay 固定历史 → eager vs graph、BS1/BSN、同图档/跨图档、有无 padding，逐 token 比较 | #15604 工具；覆盖 #18936/#19034/#19192/#19321/#19177 |
| P1 | 可直接启动 | Router/Top-K 规则核对（绕过 BI 替换、FP32 统一、tie-break/溢出确定性） | vLLM #59985/#59645/#60453 |
| P1 | 可直接启动 | 通信归约顺序核对（DSA O-proj / HCCL 分组是否随 batch 漂移；空 rank 参加 collective） | Ascend #19237；vLLM #59925 契约 |
| P1 | 可直接启动 | 动态 batch 状态自查（过滤请求后贪心/采样标志刷新）与 GDN 混合模型回归 | SGLang #44923；SGLang GDN 三 PR + Ascend #19378 |
| P1 | 需随上游收敛 | 编译等价约束（torch-ascend 是否有 inductor BI 等价配置；#60074 收敛后对齐） | vLLM #61118 已合入、#60074 Open |
| P1 | 需随上游收敛 | DS4 数值契约清单：按拆分 PR 描述固化 Top-K/索引/HC/专家聚合规则，做成可执行验收表 | #60442~#60454/#59925/#59929（均 Open） |
| P2 | 验收动作 | CI 扩展：调度准确性测试开启 BI（移植 ROCm 做法）；spec/MTP 单独验收 | vLLM #62048/#62893；Ascend #18765/#19220 |
