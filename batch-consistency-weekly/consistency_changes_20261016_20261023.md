# Batch 不变性与训推一致性：阶段变化说明（2026-10-16 → 2026-10-23）

比较时间：2026-10-16 上次分析 → 2026-10-23 本次分析，日历日期间隔 7 天，均以北京时间表述。范围以 **vLLM / vllm-ascend 的非量化模型 batch 一致性**为主，SGLang、verl / Megatron 作为对齐参照与训练依赖。本文回答三个问题：当前两仓库的 batch 一致性状态、10-16 至 10-23 合入了哪些 PR、有哪些需要新增跟踪。完整支持矩阵见[整体报告](./sglang_vllm_ascend_consistency_report.md)，上一阶段变化见[10 月 10 日至 16 日说明](./consistency_changes_20261010_20261016.md)。

依据为本地提交树（已 fetch 快进）与 2026-10-23 核查的 GitHub 状态；PR 合入以本地主线提交树为准，GitHub API 匿名额度中途耗尽后改用提交树与 PR diff 交叉核对。未运行 GPU/NPU 数值测试；测试数据来自 PR 作者时按其原有范围引用。

## 1. 时间区间与比较基准

| 仓库 | 上次 HEAD | 本次 HEAD | 本次提交日期（北京时间） |
| --- | --- | --- | --- |
| vLLM | `dd9e07a25f` | `18343393b4` | 2026-10-23 |
| vllm-ascend | `c5766cd5e` | `db8836752` | 2026-10-23 |
| SGLang | `4e85e8ed37` | `fa628c5359` | 2026-10-23 |
| verl（训练依赖） | `5b550b0c` | `bfff4d90` | 2026-10-23 |

窗口内新增提交数：vLLM 453、vllm-ascend 151、SGLang 561、verl 35。下文只展开与 batch 一致性及 DS4/V4.1 跟踪直接相关的部分，不把 HEAD 差异全部视为本期新增能力。

## 2. 当前 vLLM / vllm-ascend batch 一致性状态

- **vLLM**：`VLLM_BATCH_INVARIANT=1` + 集中确定性模块的主线路径不变。本期实质变化有四项：Inductor BI 开关进入主线（[#61118](https://github.com/vllm-project/vllm/pull/61118) 合入，`override_envs_for_invariance()` 中设置 `inductor_config.batch_invariant=True`，合入时为直接赋值，随后 [#62274](https://github.com/vllm-project/vllm/pull/62274) 补 `hasattr` 防御）；BI 验证模型名单新增 Gemma-2、Granite (MoE) SWA、Qwen3-0.6B（[#61254](https://github.com/vllm-project/vllm/pull/61254)）；cooperative TopK 在候选溢出时由近似改为精确选择（[#59645](https://github.com/vllm-project/vllm/pull/59645)）；mHC TileLang 对非有限 padding 行跳过 Sinkhorn、≤4 token 融合 post+pre（[#62745](https://github.com/vllm-project/vllm/pull/62745)）。**DeepSeek-V4 专项拆分 PR（#60442/#60443/#60444/#60453/#60454、#59925/#59929）全部仍 Open，V4 0-diff 方案未进入主线；#60074（pin Inductor kernel 配置）仍 Open，与已合入的 #61118 是互补关系而非互相取代，最终收敛方案需继续跟踪。**
- **vllm-ascend**：**主线 BI 面本窗口零代码改动**——`vllm_ascend/batch_invariance.py`（算子注册）、`worker/v2/sample/`、`batch_invariance.md`（调度限制与硬件矩阵）、four_card consistency e2e 在窗口内均无 diff。Qwen3.5-35B-A3B 的 A2/A3 验证状态与"分块/缓存/抢占不与 BI 组合"的限制延续上期。窗口内 151 个合入集中在 V4.1 bounded replay 跨 runner/PCP 扩展、PCP 图回放 padding、GDN 状态索引、MTP/DSpark 正确性与 CI；这些是**图与状态正确性地基**，不新增 BI 能力。GDN checkpoint/replay（#18765）仍 Open。
- **verl**：[#8025](https://github.com/verl-project/verl/pull/8025) 已于 10-22 合入——rollout 与 actor 的 raw logprob mismatch 指标（`training/rollout_logprobs_mismatch_count` 窗口内求和、`training/rollout_logprobs_diff_max`）落地，这是此前跟踪的"一致性度量验收入口"。#8026（actor BI 前向开关）仍 Open。
- **SGLang**：窗口内合入 Qwen3.8 Flash-Next 的 BI kernel 修复（[#42087](https://github.com/sgl-project/sglang/pull/42087)）、确定性模式为 spec 拒绝采样 draft 提供 seed（[#33395](https://github.com/sgl-project/sglang/pull/33395)）、AMD 确定性模式 TP>2 拒绝 RCCL all-reduce（[#43809](https://github.com/sgl-project/sglang/pull/43809)）；GDN 三修复（#44076/#44194/#44438）仍 Open。

## 3. 本期合入 PR 说明（10-16 → 10-23）

表注：合入时间以本地主线提交树为准（北京时间），PR 号来自提交主题。"已合入"不代表 Ascend 已适配或非量化验收通过。

### 3.1 vLLM：batch 一致性直接相关

| PR | 合入 | 改动 | 边界与 Ascend 对应工作 |
| --- | --- | --- | --- |
| [#61118](https://github.com/vllm-project/vllm/pull/61118) | 10-17 | 确定性初始化中启用 Inductor `batch_invariant`，补齐"开了环境变量但编译归约未进 BI"的缺口。上期 P0 跟踪项合入 | 仅 CUDA/Inductor 配置；NPU 的等价编译约束（算子分派、tile、归约自动调优）仍需单独实现验证；#60074 仍 Open，勿重复搬运 |
| [#62274](https://github.com/vllm-project/vllm/pull/62274) | 10-18 | ROCm：仅当 torch 提供 `inductor_config.batch_invariant` 时才设置 | 说明该开关是平台条件性的；NPU 版本要检查 torch-ascend 是否具备同配置 |
| [#61254](https://github.com/vllm-project/vllm/pull/61254) | 10-20 | BI 验证名单新增 Gemma-2、Granite (MoE) SWA、Qwen3-0.6B | 文档同时写明 Granite MoE SWA 排除 FLEX_ATTENTION 与 decode-vs-prefill 测试（per-layer SWA KV 窗口差异）——名单扩容不等于全组合通过 |
| [#59645](https://github.com/vllm-project/vllm/pull/59645) | 10-21 | cooperative TopK 候选溢出时改精确基数选择、处理大量并列；新增 per-row 测试 | 与确定性 Top-K 的 tie-break/溢出路径直接相关；NPU Indexer/TopK 需保证"溢出与并列路径也确定"，可对照 #60453 的复合键方案 |
| [#62745](https://github.com/vllm-project/vllm/pull/62745) | 10-21 | mHC TileLang：非有限 padding 行跳过 Sinkhorn；≤4 token 融合 post+pre | padding 行组成随 batch 变化，HC 归约路径变动需跨 batch 回归；Ascend 已有显式 HC pre/post，对照其 padding 行处理 |
| [#61465](https://github.com/vllm-project/vllm/pull/61465) | 10-19 | 要求 cudnn-frontend ≥1.31.0：旧版 cuDNN 可能从 CUDA graph 回放未加载的 kernel | 图回放正确性依赖；NPU 图回放框架的同类"回放未加载/过期 kernel"风险值得自查 |
| [#62126](https://github.com/vllm-project/vllm/pull/62126)、[#62042](https://github.com/vllm-project/vllm/pull/62042) | 10-16/17 | FP32 router GEMM 支持 Step-5 Preview / Muse Spark Open；ll_bf16_gemm 新增 (5120,256) BF16 输出 kernel | Router 数值路径继续向 FP32/统一路径收敛，与 #59985 GateLinear guard 同向；Ascend DS4 router 已是 FP32 输入，继续核对分派 |
| [#62048](https://github.com/vllm-project/vllm/pull/62048)、[#62893](https://github.com/vllm-project/vllm/pull/62893)、[#62280](https://github.com/vllm-project/vllm/pull/62280) | 10-17/18/21 | ROCm CI：ngram、eagle3 调度准确性测试开启 BI；MoRIIO dummy-weight P/D 测试 pin Inductor pointwise 配置 | CI 约束收紧，不是新能力；"调度准确性测试 + BI"的做法可移植到 Ascend CI |
| [#61996](https://github.com/vllm-project/vllm/pull/61996) | 10-17 | split top-p 小 batch 性能：MRV2 传 `has_topp_only_rows`，无纯 top-p 行时跳过多余 split | 采样路径性能改动、不改格式；sampler 被触碰，下次回归需覆盖小 batch top-p 路径 |
| [#61551](https://github.com/vllm-project/vllm/pull/61551) + 回退 #62186 | 10-17 | "train-inference consistency agent skill" 文档当天加入又回退 | 主线净变化为零，不列为进展 |

### 3.2 vLLM：V4 / V4.1 相关（非 BI 闭环）

| PR | 合入 | 改动 | 定位 |
| --- | --- | --- | --- |
| [#59857](https://github.com/vllm-project/vllm/pull/59857) | 10-16 | DeepSeek-V4.1-Flash PCP 支持 | 生产组合扩展，不是 BI |
| [#62183](https://github.com/vllm-project/vllm/pull/62183) | 10-19 | V4.1-Flash APC + encoder 侧 SWA bounded replay | prefix 命中后的状态重建，与 Ascend #18936/#19321 同主题 |
| [#62377](https://github.com/vllm-project/vllm/pull/62377) | 10-19 | DSv3.2/GLM-5 sparse MLA 在 DCP=1 下允许 decode-only full CUDA graph | 入图范围扩大；BI 未随入图自动成立 |
| [#62652](https://github.com/vllm-project/vllm/pull/62652) | 10-21 | V4.1 replay 保留 target-token scoring 行 | replay 数值正确性修复 |
| [#61868](https://github.com/vllm-project/vllm/pull/61868) | 10-21 | V4.1-Flash GB300 specialized decode | 性能/部署路径 |
| [#60834](https://github.com/vllm-project/vllm/pull/60834) | 10-23 | V4 paged MXFP4 sparse indexer dense path（opt-in） | 量化路径，不纳入非量化结论 |
| [#62302](https://github.com/vllm-project/vllm/pull/62302) | 10-18 | routed experts（R3）使用文档 | 仅文档 |

**结论**：V4.1 生产组合覆盖在扩大，但 #54955 拆分的 BI 专项 PR 全部未合入，不能写成"V4/V4.1 batch 一致性落地"。

### 3.3 vllm-ascend 合入

| PR | 合入 | 改动 | 一致性相关定位 |
| --- | --- | --- | --- |
| [#18445](https://github.com/vllm-project/vllm-ascend/pull/18445) | 10-16 | MiniMax-M3 融合稀疏注意力在 piecewise graph 中的边界修复 | 图模式数值正确性 |
| [#16936](https://github.com/vllm-project/vllm-ascend/pull/16936) | 10-18 | GLM-5.3-Flash MRV2 MTP capture 生命周期保持 | MTP + 图回放组合正确性 |
| [#19034](https://github.com/vllm-project/vllm-ascend/pull/19034) | 10-19 | DSV4.1 replay metadata 按 PCP 本地分段分配尺寸 | 上一期 #18750 同方向延续 |
| [#18936](https://github.com/vllm-project/vllm-ascend/pull/18936) | 10-19 | V4.1 decoder bounded replay 扩展到多 model runner | replay 能力扩展，非 BI |
| [#19177](https://github.com/vllm-project/vllm-ascend/pull/19177) | 10-19 | 加载缓存图前先初始化静态 kernel runtime | 图回放正确性依赖 |
| [#19192](https://github.com/vllm-project/vllm-ascend/pull/19192) | 10-20 | 修复 replicated PCP 图回放的 query padding | 与上期 #18750 同主题的 padding 修复 |
| [#19240](https://github.com/vllm-project/vllm-ascend/pull/19240) | 10-20 | 统一 DeepSeek V4 各硬件 TP 上限 | 配置一致性 |
| [#19165](https://github.com/vllm-project/vllm-ascend/pull/19165) | 10-21 | MRV2 在 DP+MTP 下保持 hybrid dummy query 宽度 | 图捕获宽度一致性 |
| [#19378](https://github.com/vllm-project/vllm-ascend/pull/19378) | 10-21 | 修复 GDN recurrent state 索引越过 token 数（+UT） | **GDN 状态正确性**；与 SGLang GDN 分块修复互为参照 |
| [#19220](https://github.com/vllm-project/vllm-ascend/pull/19220) | 10-22 | DSpark 双向 SparseFlashMla | MTP/DSpark + 稀疏 MLA 组合，需纳入回归 |
| [#19321](https://github.com/vllm-project/vllm-ascend/pull/19321) | 10-23 | V4.1 SWA bounded replay 支持 PCP | replay + PCP 组合扩展 |
| [#17725](https://github.com/vllm-project/vllm-ascend/pull/17725) | 10-22 | batch-sharded 与 PCP lmhead 权重切分检查 | 权重切分防御 |
| [#18684](https://github.com/vllm-project/vllm-ascend/pull/18684) | 10-19 | MegaMoE W4A8 MXFP 权重准备修复 | 量化路径，不纳入非量化结论 |
| #18831 / #19074 / #19166 / #19477；#19184 / #19242 | 10-19~22 | DSV4.1/PCP/DSpark nightly CI 与部署文档 | 测试调度与文档 |

**结论**：Ascend 本期合入围绕"入图 + bounded replay + PCP/MTP 组合的正确性地基"，与上期 #18750/#18767 同向推进；BI 主线（算子注册、sample、文档限制、e2e）零改动，调度限制未解除。

### 3.4 SGLang 与 verl

- SGLang 合入：[#42087](https://github.com/sgl-project/sglang/pull/42087)（Qwen3.8 Flash-Next BI kernel 修复，10-20）、[#33395](https://github.com/sgl-project/sglang/pull/33395)（确定性模式为 spec 拒绝采样 draft 提供种子，10-22）、[#43809](https://github.com/sgl-project/sglang/pull/43809)（AMD 确定性模式 TP>2 拒绝 RCCL all-reduce，10-22，与"通信归约顺序需固定"同类问题）、[#44906](https://github.com/sgl-project/sglang/pull/44906)（mHC 小 batch post 并行化，10-19）、[#45579](https://github.com/sgl-project/sglang/pull/45579)（dsv4 request-window history rows 单次计算，10-21）；#45896 关闭未合入。
- verl 合入：[#8025](https://github.com/verl-project/verl/pull/8025)（10-22）——rollout vs actor raw logprob mismatch 指标。细节：`mismatch_count` 在聚合窗口内求和而非取均值；带 one-ULP 概率差掩蔽测试（1e-8 级 logprob 差经 exp 后不可分辨时不计入）。**对 Ascend 的含义**：训推对齐工作应接入等价指标（NPU rollout/actor 逐 token logprob 比较），作为此前"训推层验收"的可执行入口；#8026 的 actor BI 前向开关仍 Open。

## 4. 原有重点 PR 在间隔内的状态

表注：比较上次核查（10-16）与本次核查（2026-10-23）。head 为本次记录值；"无变化"指 head 与上次记录一致，不评论评论/标签变动。

| PR | 上次 → 本次 | 判断 |
| --- | --- | --- |
| [vLLM #54955](https://github.com/vllm-project/vllm/pull/54955) | Open，head `4b5c627c23e6` 不变 | 总预览继续作方案索引 |
| [vLLM #60442](https://github.com/vllm-project/vllm/pull/60442) / [#60443](https://github.com/vllm-project/vllm/pull/60443) / [#60444](https://github.com/vllm-project/vllm/pull/60444) / [#60453](https://github.com/vllm-project/vllm/pull/60453) / [#60454](https://github.com/vllm-project/vllm/pull/60454) | 全部 Open；本次记录 head `ff4d1371ad`/`5a561598f7`/`1fa966cdf1`/`6a86b2b692`/`961f731a36` | V4 拆分仍待合入 |
| [vLLM #59925](https://github.com/vllm-project/vllm/pull/59925) / [#59929](https://github.com/vllm-project/vllm/pull/59929) | Open；#59929 有更新（10-21），head `ab5e5b4013` | MoE/HC 专项仍待合入 |
| [vLLM #60074](https://github.com/vllm-project/vllm/pull/60074) | Open，head `c74b1a4a24` | 与已合入 #61118 互补，跟踪最终收敛 |
| [vLLM #60959](https://github.com/vllm-project/vllm/pull/60959) | Open，无变化 | 性能向，暂缓 |
| [vLLM #55868](https://github.com/vllm-project/vllm/pull/55868) | Open/WIP，有更新（10-20） | GDN 缓存 BI 仍 WIP |
| [vLLM #52522](https://github.com/vllm-project/vllm/pull/52522) | Open，无变化（09-04 后未动） | 与新 #62193 的豁免作用域可能互补 |
| SGLang [#44076](https://github.com/sgl-project/sglang/pull/44076) / [#44194](https://github.com/sgl-project/sglang/pull/44194) | Open，无变化 | GDN 分块修复待合入 |
| SGLang [#44438](https://github.com/sgl-project/sglang/pull/44438) / [#43611](https://github.com/sgl-project/sglang/pull/43611) | Open，有更新（10-21 / 10-19） | 校验/Mamba2 仍待合入 |
| Ascend [#18765](https://github.com/vllm-project/vllm-ascend/pull/18765) | Open，有更新（10-19） | GDN checkpoint/replay 待合入 |
| verl [#8026](https://github.com/verl-project/verl/pull/8026) / [#7688](https://github.com/verl-project/verl/pull/7688) | Open，有更新（10-23 / 10-21） | actor 前向开关与 DS4 recipe 仍待合入 |
| Megatron [#7050](https://github.com/NVIDIA/Megatron-LM/pull/7050) | Open，无变化 | 训练端预览未推进 |
| Issue/RFC | vLLM #27433（更新 10-22）、#48305、Ascend #5487 仍 Open；SGLang #10278 为 closed（早已关闭，非本期变化） | Issue 不计入待合入 PR |

## 5. 需要新增跟踪的项

表注：下表均为 Open（核查时点 2026-10-23）。修改内容来自 PR 当前 diff/描述，不能当作主线已具备能力。优先级以非量化 Ascend 适配相关度排序。

| 优先级 | PR | 内容 | 对 Ascend / 一致性的含义 |
| --- | --- | --- | --- |
| P0 | [vLLM #62192](https://github.com/vllm-project/vllm/pull/62192) | BI 下 pin row-invariant GEMM 路径；新增跨 tuned M-bucket 的 invariance 测试（含 nemotron_h qkv/o_proj 形状） | M 维分桶 GEMM 若按 batch 切换数值路径会破坏 BI；Ascend 需检查自身 GEMM 分桶/自动调优是否引入 M 相关路径。关联 [#62219](https://github.com/vllm-project/vllm/pull/62219)（H100 BI matmul 表新增 Llama-3.1-8B/Qwen3-8B tuned 配置） |
| P0 | [vLLM #62017](https://github.com/vllm-project/vllm/pull/62017) | Nemotron-H Lightning NVFP4 的"BI serving aligned with training"；含 mamba2 SSD BI 算子、`exact_replay`、`determinism/scope.py`、`batch_invariant_configs.py` | 模型是 NVFP4 量化，不计入非量化支持；但其 Mamba2/SSD BI 基础设施与 scope/configs 拆分是通用新增，且与 #62193 同源，跟踪最终形态 |
| P0 | [vLLM #62193](https://github.com/vllm-project/vllm/pull/62193) | `batch_invariance_exempt()` 作用域：spec decode drafter 构建时豁免 BI kernel 要求（新增 `determinism/scope.py` 与豁免测试） | 与 #52522 组合构成 speculative BI 的完整方案（draft 豁免 + verify 保持）；跟踪两 PR 的收敛关系 |
| P0 | [Ascend #19442](https://github.com/vllm-project/vllm-ascend/pull/19442) | TP4 BI e2e 新增 Qwen3.5-35B-A3B 4 层切片（3 GDN + 1 full-attention + 256 experts top-8，dummy 权重，仅语言塔） | 首个混合 GDN+MoE 的 Ascend BI e2e；仍是切片 + 随机权重的基础覆盖，不是完整模型验收，但比现有两层 dense 覆盖前进一步 |
| P1 | [Ascend #19237](https://github.com/vllm-project/vllm-ascend/pull/19237) / [#19238](https://github.com/vllm-project/vllm-ascend/pull/19238) | DSA O-proj 交换对齐 coordinated runtime batch（main / v0.30.0 backport） | o-proj 通信分组若随 runtime batch 变化，归约顺序随 batch 组成漂移——通信路径上的 BI 相关问题，需确认其分组规则是否固定 |
| P1 | [SGLang #45063](https://github.com/sgl-project/sglang/pull/45063) | NPU batch-invariant：启用 `torch.use_deterministic_algorithms(True, warn_only=True)` | SGLang NPU 路线的 BI 尝试，与 vllm-ascend 的算子注册路线不同，可对照其覆盖面与 warn_only 的漏洞 |
| P1 | [SGLang #44923](https://github.com/sgl-project/sglang/pull/44923) | `filter_batch` 移除行后刷新 `is_all_greedy`/`is_any_greedy` | 动态 batch 组成导致贪心标志过期——直接的跨 batch 不一致来源，vllm-ascend 调度器应自查同类标志刷新 |
| P2 | [SGLang #45599](https://github.com/sgl-project/sglang/pull/45599) | AMD gfx950 分段 split-K bf16 BI matmul | BI 平台覆盖扩展；split-K 固定思路与 #54955 matmul 规划一致 |
| P2 | [SGLang #44861](https://github.com/sgl-project/sglang/pull/44861) | modelopt：SM12x FP8 在确定性模式保持 batch-invariant | 量化边界，仅作路线参考 |
| P2 | [verl #8317](https://github.com/verl-project/verl/pull/8317) | GB200 镜像：Megatron Lite + vLLM batch-invariant rollout | 训练侧 BI 基础设施，非 NPU |

时间性质：#62192/#62193/#62017/#62219、Ascend #19442/#19237/#19238、SGLang #45063/#44861/#45599/#44923、verl #8317 均为窗口内（10-16 ~ 10-23）新建；SGLang #44923 虽随检索出现，其修复对象（动态 batch 贪心标志）长期存在。窗口内关闭未合入：SGLang #45896（engram hasher idle DP batch）。

## 6. 与上期"建议优先做的三件事"的对应

1. **训推基线**：verl #8025 指标已合入，当下动作是让 NPU rollout/actor 接入等价的 mismatch 指标并跑首轮统计；#8026 合入后再接 actor BI 开关。
2. **DS4 契约**：拆分 PR 无合入进展；vLLM 主线的 cooperative TopK 精确化（#59645）与 FP32 router GEMM 扩展（#62126/#62042）是可参考的数值规则，继续等 #60442~#60454/#59925/#59929 收敛。
3. **生产组合**：Ascend 图回放/bounded replay/PCP 修复持续累积（#18936/#19034/#19192/#19321），"分块、缓存、抢占不与 BI 组合"的限制未变；建议把本期的 GDN 索引修复（#19378）与 SGLang GDN 三 PR 的边界问题合并成一份混合模型回归清单。

## 7. vllm-ascend 可以做 / 待做的事情

表注：基于 10-16 → 10-23 窗口证据整理。"可直接启动"不依赖尚未合入的上游 PR；"需随上游收敛"先做准备工作，等对应 PR 合入后对齐；"验收动作"用现有主线工具（trace replay #15604、NPU selected-logprob、BI 开关）即可执行。输出文本一致、seed 固定、BI=1 均不能代替数值验收；backward 逐 bit 可复现是另一个目标。

| 优先级 | 类别 | 事项 | 依据 / 对照 |
| --- | --- | --- | --- |
| P0 | 可直接启动 | 接入 verl #8025 等价的训推差异度量：NPU rollout 与 actor 的 raw logprob mismatch（mismatch_count 聚合窗口内求和、diff_max、one-ULP 概率差掩蔽），先在 Qwen3 dense 小模型跑首轮统计 | [verl #8025](https://github.com/verl-project/verl/pull/8025) 已合入（10-22），指标语义可直接复刻 |
| P0 | 可直接启动 | NPU GEMM/算子分派自查：matmul、归约、norm 的实现选择与 tile 分桶是否依赖 M（token 数），补算子级跨 batch 不变性测试 | vLLM [main #62192](https://github.com/vllm-project/vllm/pull/62192) 的 M-bucket invariance 测试思路；Ascend 算子注册 `vllm_ascend/batch_invariant.py` 覆盖面核对 |
| P0 | 可直接启动 | V4/V4.1 图回放回归矩阵：trace replay 固定 decode 历史 → eager vs graph、BS1/BSN、同图档/跨图档、有无 padding，逐 token logits/logprob bitwise 比较 | 工具已具备（#15604 + MRV2 patch）；覆盖本期修复点 #18936/#19034/#19192/#19321/#19177 |
| P1 | 可直接启动 | Router/Top-K 数值规则核对：router 是否绕过 BI 替换、FP32 路径是否统一；TopK 的 tie-break、候选溢出与并列路径是否确定 | vLLM #59985（GateLinear guard）与 #59645（TopK 溢出精确化）是参照规则；DS4 对照 #60453 复合键 |
| P1 | 可直接启动 | 通信归约顺序核对：DSA O-proj / HCCL collective 的分组规则是否随 runtime batch 漂移；空 rank 是否参加 collective | Ascend [#19237](https://github.com/vllm-project/vllm-ascend/pull/19237)（Open）点名的 O-proj 问题；vLLM #59925 的空 rank 契约 |
| P1 | 可直接启动 | 动态 batch 状态自查：scheduler 过滤/移除请求后，贪心与采样标志（is_all_greedy 类状态）是否刷新；权重切分防御沿用已合入的 #17725 检查 | SGLang [#44923](https://github.com/sgl-project/sglang/pull/44923) 是同类 bug 的参照 |
| P1 | 可直接启动 | GDN 混合模型回归：核对 NPU GDN 是否存在 SGLang #44076/#44194 的分块边界/tile 问题（不能直接认定同一 bug）；GDN 状态索引越界修复 #19378 纳入回归 | SGLang GDN 三 PR + Ascend #19378 + Open 的 [#19442](https://github.com/vllm-project/vllm-ascend/pull/19442) |
| P1 | 需随上游收敛 | 编译等价约束：查 torch-ascend 是否具备 inductor batch_invariant 等价配置；#60074 收敛后决定搬运方案，避免与已合入的 #61118 重复；把静态 kernel runtime 初始化纳入 BI × 图模式组合回归 | vLLM #61118 已合入、#60074 仍 Open；Ascend #19177 |
| P1 | 需随上游收敛 | DS4 数值契约落地：等 #60442~#60454/#59925/#59929 收敛前，先按其 PR 描述固化 NPU 侧规则清单（Top-K 复合键与 tie-break、C4/C128 索引与无效槽屏蔽、HC 单 token 等价拆分与 padding 行处理、专家容量/聚合顺序、shared expert 单次执行），做成可执行验收表 | 拆分 PR 描述即契约草案；HC padding 行对照 vLLM #62745 |
| P2 | 需随上游收敛 | spec decode / MTP：等 #62193 + #52522 收敛后对照 draft 豁免方案；MTP 状态恢复验收单独做（#18765 仍 Open，不承诺 BI）；DSpark 双向 SparseFlashMla（#19220 已合入）纳入回归 | vLLM #52522/#62193；Ascend #18765/#19220 |
| P2 | 验收动作 | CI 扩展：把 ROCm"调度准确性测试开启 BI"的做法移植到 Ascend nightly（ngram/eagle3 类测试 + pin 归约/编译配置） | vLLM #62048/#62893/#62280；Ascend #18831/#19074/#19166 已扩 nightly，可叠加 BI 维度 |
| P2 | 验收动作 | 验收边界纪律：#19442 是切片 + dummy 权重的基础覆盖，不是完整模型验收；Ascend 模型名单"已列入"不等于全组合通过；每步扩展记录性能成本 | 本报告第 2、7 节的既有口径 |

建议的启动顺序：先做 P0 三项（度量接入 → 算子分派自查 → 图回放回归矩阵），三者互相独立、都不依赖上游合入，且分别对应"训推对齐入口""算子契约""生产组合正确性"三条线；P1 的通信与 GDN 回归可与其并行。

## 8. 本次代码证据与验收边界

- vLLM 主线 diff：`vllm/model_executor/determinism/batch_invariant.py`（+4，#61118）、`docs/features/batch_invariance.md`（+4，#61254）；`vllm/v1/worker/gpu/sample/` 变更来自 #61996/#52477/#62888。
- vllm-ascend 主线 diff：`vllm_ascend/batch_invariant.py`、`worker/v2/sample/`、`docs/.../batch_invariance.md`、`tests/.../consistency/` 均为空 diff——BI 主线零改动的直接证据。
- 新跟踪 PR 的 diff 快照：`./evidence/`（#62192/#62193/#62017/#19442/#19237/#45063/#44923/#44861/#45599/#8025）。
- GitHub 状态核查时点：2026-10-23 12:00 前后（北京时间）；GitHub API 匿名额度耗尽后，合入状态一律以本地提交树（`git log` 提交主题中的 PR 号 + `git show`）核对，PR 状态（Open/Closed/Merged）在额度内已核对的按 API 记录，其余按 diff 可获取性佐证。
- 未运行 GPU/NPU 数值测试；文中性能与通过率均来自 PR 作者描述，不作为独立验收结论。
