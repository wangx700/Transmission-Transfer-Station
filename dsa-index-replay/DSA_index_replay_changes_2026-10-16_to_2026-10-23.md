# DSA Index Replay 阶段变化说明：2026-10-16 → 2026-10-23

对比间隔：7 个自然日；日期按北京时间。范围：vLLM、vLLM-Ascend、SGLang。依据本地官方 main（已同步至 10-23 HEAD）提交差异及 GitHub PR/RFC 页面核查，未执行硬件验证。

## 1. 相比上一次分析，实际改变了什么

**索引采集返回能力的支持范围仍未改变：vLLM 主线与 Ascend 主线依旧没有完整采集返回链路，SGLang 维持现有 capturer。** 本期最重要的变化是 vLLM 出现第二条、覆盖面更完整的公共实现 Draft：[#62915](https://github.com/vllm-project/vllm/pull/62915) Profile-based indexer replay（10-21 前后创建），在 #49555 之外新建 AuxOutput 采集、存储与 API 返回基础设施。Ascend 侧合入 DSpark 双向 SparseFlashMla（逻辑 per-request top-k 索引）、DeepSeek 层间 O-projection buffer 共享，并移除旧 DSA-CP 文档；vLLM 侧合入首个 CP sparse indexer 实现（MiniMax-M3，ROCm opt-in）及多项 top-k 内核正确性修复。

| 项目 | 10-16 分析基线 | 10-23 分析结果 | 七天内变化 |
|---|---|---|---|
| vLLM | RFC #47280 Open，#49555 Draft；主线无完整返回链路 | RFC 仍 Open，#49555 仍 Draft；新增 #62915 Draft | 出现第二条实现入口；top-k 内核修复、MiniMax-M3 CP indexer 合入 |
| vLLM-Ascend | 已有 NPU indexer，无完整索引采集返回 | 支持状态相同 | DSpark 双向 SparseFlashMla、O-proj buffer 共享合入；旧 DSA-CP 文档移除 |
| SGLang | 已有 CUDA/NPU 索引采集返回，attention TP=1 限制 | 支持状态相同，限制仍在（indexer_topk.py:33 assert） | KVShard 页分片 DSA cache、V4.1 indexer prefill 图等基础工作合入 |

表格注释：基线提交为 vLLM dd9e07a25f、Ascend c5766cd5e、SGLang 4e85e8ed37；本次 HEAD 分别为 19c0e170d2、db8836752、f4a2df3dac，本地三个仓库已同步到上述 HEAD。状态比较限定上述源码与 GitHub 页面核查；"无新增"不代表 indexer 相关代码没有更新。

## 2. 本期已合入 PR 与修改内容

### vLLM（基线 dd9e07a25f → 19c0e170d2）

| PR | 合入日期 | 修改内容 | 对采集适配的影响 |
|---|---|---|---|
| vLLM [#59645](https://github.com/vllm-project/vllm/pull/59645) | 10-21 | cooperative TopK 在候选溢出时改为精确选择，新增溢出内核与按行测试 | 溢出场景 top-k 结果改变；采集预期需按新内核校准 |
| vLLM [#62467](https://github.com/vllm-project/vllm/pull/62467) | 10-19 | persistent_topk 不再选择超出行长度的 float4 尾通道 | 修复越界 top-k 选择，影响索引有效性判断 |
| vLLM [#61938](https://github.com/vllm-project/vllm/pull/61938) | 10-18 | Triton MLA decode int32 溢出与 logits workspace 记账修复 | int32 索引/位移上限需在采集返回编码中保持一致 |
| vLLM [#57832](https://github.com/vllm-project/vllm/pull/57832) | 10-21 | MiniMax-M3 CP sparse lightning indexer（Triton，opt-in，ROCm） | vLLM 首个合入的 CP indexer 实现，可作为 CP 索引坐标与局部行归属参考 |
| vLLM [#62377](https://github.com/vllm-project/vllm/pull/62377) | 10-19 | DSv3.2/GLM-5.2/5.3 sparse MLA 在 DCP=1 时允许 decode-only 全量 CUDA graph | 图执行范围扩大，采集需覆盖 graph 路径生命周期 |
| vLLM [#61943](https://github.com/vllm-project/vllm/pull/61943) | 10-19 | ROCm 用 AITER top-k 替换 DSv3.2/GLM-5 sparse indexer top-k 内核 | 平台内核差异，不改接口；top-k 数值需重验 |

低影响组（本期合入、主要属性能与平台工程）：[#60834](https://github.com/vllm-project/vllm/pull/60834)（ROCm paged MXFP4 sparse indexer dense path）、[#61615](https://github.com/vllm-project/vllm/pull/61615)（SM90 small-head FlashMLA sparse prefill）、[#62029](https://github.com/vllm-project/vllm/pull/62029)（FlashInfer DCP prefill context pass 改走 trtllm-gen）、[#60805](https://github.com/vllm-project/vllm/pull/60805)（HiSparse hot buffer 按 rounded sparse top-k 宽度取整）、[#59857](https://github.com/vllm-project/vllm/pull/59857)（DeepSeek-V4.1-Flash PCP 支持）。

表格注释：窗口为上期基线之后的提交，日期分布 10-15 至 10-23（其中 10-15 的 3 个提交为基线之后合入），日期取主线合入提交口径，与上期报告一致。修改描述依据 PR 与主线提交，未本地复测。

### vLLM-Ascend（基线 c5766cd5e → db8836752）

| PR | 合入日期 | 修改内容 | 对采集适配的影响 |
|---|---|---|---|
| Ascend [#19220](https://github.com/vllm-project/vllm-ascend/pull/19220) | 10-22 | DSpark 双向 SparseFlashMla：draft 注意力改走逻辑 per-request top-k 索引，新增持久 dspark_swa_topk_lengths buffer 与 [T,1] 长度契约 | 采集若覆盖 DSpark draft 路径，需按逻辑索引与 [T,1] 契约对齐坐标 |
| Ascend [#19190](https://github.com/vllm-project/vllm-ascend/pull/19190) | 10-21 | DeepSeek 层间共享 O-projection buffer（配置开关 + UT） | 静态内存预算与 buffer 归属变化；不改 top-k 语义 |
| Ascend [#19262](https://github.com/vllm-project/vllm-ascend/pull/19262) | 10-20 | o_proj 静态交换 buffer 按 decode 容量取值 | 静态 buffer 尺寸口径变化，采集侧 buffer 规划需同步 |
| Ascend [#19079](https://github.com/vllm-project/vllm-ascend/pull/19079) | 10-21 | KPool 主注意力识别为 SFA（utils 路由修正） | 影响 SFA 判定，间接影响采集入口选择 |
| Ascend [#18715](https://github.com/vllm-project/vllm-ascend/pull/18715) | 10-16 | KV Transfer 支持 packed SFA C8 分层传输 | PD/传输坐标与 C8 scale 打包需对齐 |
| Ascend [#19064](https://github.com/vllm-project/vllm-ascend/pull/19064) | 10-17 | 修复 KV 传输中 C8 K-scale 损坏 | 传输正确性依赖，非索引返回新特性 |
| Ascend [#18752](https://github.com/vllm-project/vllm-ascend/pull/18752) | 10-17 | DCP MLA decode 改用 FIA v2（含 ACL graph 测试） | DCP 路径实现变化，采集适配需跟随 |
| Ascend [#19491](https://github.com/vllm-project/vllm-ascend/pull/19491) | 10-23 | 移除旧 DSA-CP 文档，更新教程与配置口径 | CP 能力文档口径变化，影响 #18651 跟踪上下文；不改代码 |

低影响组：[#18445](https://github.com/vllm-project/vllm-ascend/pull/18445)（MiniMax-M3 piecewise graph 稀疏注意力边界修复）、[#18267](https://github.com/vllm-project/vllm-ascend/pull/18267)（MRV2 CP 精度 CI 覆盖 SFA/GQA/MLA）、[#18845](https://github.com/vllm-project/vllm-ascend/pull/18845)（Kimi A5 draft KV cache 与 FP8 MLA 输出修复）、[#19020](https://github.com/vllm-project/vllm-ascend/pull/19020)（sparse KV offload 文档）。

表格注释：Ascend 窗口共 151 个提交（10-16 至 10-23）。#19220/#19190/#19262 均触及 dsa_v1.py，是该文件本期的主要变化来源。

### SGLang（基线 4e85e8ed37 → f4a2df3dac）

| PR | 合入日期 | 修改内容 | 对采集适配的影响 |
|---|---|---|---|
| SGLang [#44280](https://github.com/sgl-project/sglang/pull/44280) | 10-23 | KVShard：页分片 DSA cache 的 attention 与 indexer 支持 | pool/页归属变化；capturer 仍要求 attn_tp_size == 1，限制未解除 |
| SGLang [#45783](https://github.com/sgl-project/sglang/pull/45783) | 10-22 | prefill 图内仅保留两级 V4.1 indexer | indexer 结构简化，采集点位需重新对位 |
| SGLang [#45573](https://github.com/sgl-project/sglang/pull/45573) | 10-21 | 恢复 request-window 行的 sparse prefill | 与返回行覆盖范围相关 |
| SGLang [#44564](https://github.com/sgl-project/sglang/pull/44564) | 10-19 | dsa-eagle-topk 迁移索引 cache 时复制 token keys 与 scales | 缓存迁移正确性依赖 |
| SGLang [#43832](https://github.com/sgl-project/sglang/pull/43832) | 10-17 | DSA skip-topk 流水线交接 owed outputs | PP/MTP 场景交接正确性 |
| SGLang [#42565](https://github.com/sgl-project/sglang/pull/42565) | 10-16 | MTP draft decode 前修正 ragged seed indices | 索引种子正确性 |

低影响组：[#44436](https://github.com/sgl-project/sglang/pull/44436)（NPU 紧凑 DSA indexer 分配提升 KV 容量）、[#38461](https://github.com/sgl-project/sglang/pull/38461)（NPU dsa 模块免 CPU seq lens 拷贝）、[#45251](https://github.com/sgl-project/sglang/pull/45251)（verify 行按单请求合并 full-scan indexer）、[#45909](https://github.com/sgl-project/sglang/pull/45909)（共享 sparse prefill KV 源每 forward 反量化一次）、[#44114](https://github.com/sgl-project/sglang/pull/44114)（QSA FP8 decode scratch）、[#45100](https://github.com/sgl-project/sglang/pull/45100)/[#44305](https://github.com/sgl-project/sglang/pull/44305)（sgl-router kv-indexer 连接稳定性）。

## 3. 持续跟踪与本次新增跟踪

### 继续跟踪

| 跟踪项 | 当前状态 | 本期变化与关注点 |
|---|---|---|
| vLLM [RFC #47280](https://github.com/vllm-project/vllm/issues/47280) | Open（10-23 页面核查） | 本期无合入；公共协议入口，需与 #62915 实现方案对照 |
| vLLM [PR #49555](https://github.com/vllm-project/vllm/pull/49555) | Open、Draft、未合入（10-23 核查） | 本期未合入；"ordinary raw indexer selections"实现入口，与 #62915 并存 |
| vLLM [一致性 RFC #48305](https://github.com/vllm-project/vllm/issues/48305) | 上层背景入口 | 关注 DSA Index Replay 分项；不据此宣称本期更新 |
| vLLM [#60734](https://github.com/vllm-project/vllm/pull/60734) | Open（10-23 核查） | DBO microbatch 隔离仍未合入；capturer 需遵守相同 buffer 归属 |
| vLLM [#59367](https://github.com/vllm-project/vllm/pull/59367) | Open（10-23 核查） | DCP top-k merge workspace 未合入；CP 采集内存预算仍需自行预留 |
| Ascend [#18651](https://github.com/vllm-project/vllm-ascend/pull/18651) | Open（10-23 核查） | A5 FP8 LI C8 + DSA CP 未合入；注意 #19491 已调整 CP 文档口径 |
| Ascend [RFC #10072](https://github.com/vllm-project/vllm-ascend/issues/10072) | Open（10-23 核查） | RL 总路线图；Indexer Replay 表述仍混有 MoE 路由语义，作背景入口 |

表格注释：Open/Draft 状态来自 10-23 GitHub 页面核查（state 字段与 Draft 标识）；"未合入"同时以本地主线 HEAD 检索不到对应合入提交为准。旧 PR #47279 已由 #49555 接替的结论维持。

### 本次新增建议跟踪

| 优先级 | 跟踪项 | 当前状态 | 修改内容与跟踪理由 |
|---|---|---|---|
| 高 | vLLM [#62915](https://github.com/vllm-project/vllm/pull/62915) | Open、Draft；10-21 前后创建，10-23 仍 Draft | Profile-based indexer replay for sparse models：新增 indexer_capture 层、aux_output_connector（store/profile/memory/worker/artifacts）、docs/training/indexer_replay.md、OpenAI completion/chat 协议字段与 deepseek_v2/v32/v4 采集集成，共 57 个文件；是目前覆盖"采集→保存→API 返回"最完整的公共实现，Ascend 适配应以其与 #47280 的收敛结果为准 |
| 高 | Ascend [#19220](https://github.com/vllm-project/vllm-ascend/pull/19220) | 10-22 合入 | DSpark draft 改走逻辑 per-request top-k 索引并新增持久长度 buffer；NPU 采集适配必须覆盖该路径的索引坐标与生命周期 |
| 中 | vLLM [#57832](https://github.com/vllm-project/vllm/pull/57832) | 10-21 合入 | CP sparse lightning indexer 首例；CP 场景索引坐标、局部行与因果长度处理的实现参考 |
| 中 | vLLM [#63237](https://github.com/vllm-project/vllm/pull/63237) | Open；10-22/23 创建 | R3 routed-experts capturer 初始化时序（KV cache profiling 之前）；与 index replay 共用 AuxOutput/capturer 基础设施，列作边界跟踪 |
| 中 | Ascend [#19491](https://github.com/vllm-project/vllm-ascend/pull/19491) | 10-23 合入 | 旧 DSA-CP 文档移除；CP 能力口径变化直接影响 #18651 的跟踪上下文 |

表格注释："新增跟踪"指本报告首次列入，不代表本期新建。#62915 与 #63237 为本期新创建 PR；#19220/#57832/#19491 为本期合入且改变适配前提的条目。#63237 属 MoE routed-experts 返回族，因与 index replay 共用基础设施列入边界跟踪。本次仍未确认 Ascend 独立的 DSA 索引采集返回实现 PR/RFC；候选依据适配相关性筛选，非所有 indexer PR 的穷举。

## 4. 对后续工作的影响

1. #62915 与 #49555 并行推进且方案不同（Profile-based vs ordinary raw selections），Ascend 适配前需先确认公共实现收敛到哪条路径，避免按 Draft 接口提前固化采集协议。
2. Ascend DSpark 逻辑索引与 [T,1] 长度契约、O-proj buffer 共享改变了 NPU 采集的 buffer 与坐标假设；采集设计需同时覆盖 dsa_v1 的普通 top-k 与压缩 cache top-k 两条路径（[dsa_v1.py:2630](https://github.com/vllm-project/vllm-ascend/blob/db8836752/vllm_ascend/attention/dsa_v1.py#L2630)、[dsa_v1.py:2580](https://github.com/vllm-project/vllm-ascend/blob/db8836752/vllm_ascend/attention/dsa_v1.py#L2580)）。
3. vLLM CP indexer（MiniMax-M3）与 DCP=1 图范围扩大后，CP/图执行场景的索引行归属与生命周期验证应提前纳入组合测试，不以相关 PR 合入替代验证。

## 5. 边界说明（本期不属于 DSA index replay 的相邻合入）

- MoE routed-experts 返回族（R3）：[#62302](https://github.com/vllm-project/vllm/pull/62302)（R3 文档）、[#62392](https://github.com/vllm-project/vllm/pull/62392)/[#62489](https://github.com/vllm-project/vllm/pull/62489)（store 修复）、[#62624](https://github.com/vllm-project/vllm/pull/62624)+[#63013](https://github.com/vllm-project/vllm/pull/63013)（平台钩子 10-21 合入、10-22 回退）。这些属于 MoE 路由回放，不作为 index replay 进展引用。
- SWA bounded replay 族：[#62183](https://github.com/vllm-project/vllm/pull/62183)（PCP APC 与 encoder-side SWA bounded replay）、[#62652](https://github.com/vllm-project/vllm/pull/62652)（V4.1 replay 中 target-token scoring 行保留）。SWA bounded replay 在本系列文档中始终排除在范围外。

[当前整体报告](DSA_index_replay_status.md) 另行说明当前能力、限制、代码证据和适配任务。
