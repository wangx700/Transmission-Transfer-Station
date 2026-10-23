# SGLang、vLLM、vLLM-Ascend 非量化 LoRA 支持与工作报告

更新时间：2026-10-23（北京时间）。范围：浮点基础模型与浮点 LoRA 推理；量化专属优化、diffusion LoRA 不纳入主体结论。

本文独立说明截至 2026-10-23 的整体支持情况：模型与模块、通用特性、明确限制、核心待办及代码依据。未运行 GPU/NPU 模型测试；PR 作者的验证不代表本次复现。阶段变化和新增跟踪清单另见 [阶段进展说明](./LoRA阶段进展与PR跟踪_2026-10-16至2026-10-23.md)。

## 1. 核心结论

1. **普通投影与 MoE 已有基础支持**：三个项目均有线性层包装与 adapter 管理；vLLM 有专家 TP 切片修复及 fully-sharded 防护（#61284 已合入），SGLang 有 CSGMV 优化、LoRA 缓存路由隔离，且 plan-driven 新执行栈已进入 main，Ascend 有 NPU 优化包装。具体模型目标模块仍需逐项判断。
2. **DeepSeek V4.1 有两个平行专项方案，均未合入**：vLLM [#57245](https://github.com/vllm-project/vllm/pull/57245)（分组输出、compressor、文本/视觉 adapter）与 SGLang [#45231](https://github.com/sgl-project/sglang/pull/45231)（attention 投影含分组 wo_a 语义、wq_b 消歧与 shared expert）均 Open。Ascend 应对照两方案选定设计后适配，不重复实现。
3. **MLA 已有明确问题与分阶段修复，但主线仍未解决**：[问题 #48974](https://github.com/vllm-project/vllm/issues/48974)、[#56327](https://github.com/vllm-project/vllm/pull/56327)、[#56718](https://github.com/vllm-project/vllm/pull/56718)、临时保护 [#61804](https://github.com/vllm-project/vllm/pull/61804) 均 Open。SGLang #45892 已将 dense MLA correction 收入新栈；`vllm_ascend/lora` 当前无任何 MLA/kv_b 处理（10-23 核验），需先补显式拒绝/告警。
4. **不能混淆分支**：SGLang #44332 已合入的是 `sglang-miles`；#42875/#42882/#42884 已 Closed（被 #45892 单 PR 方案替代）、#42284 仍 Draft；#44862（GLM-5.3-Flash KDA adapters）合入非 main 分支，main 未落地。
5. **Ascend 核心缺口仍在**：V4.1 特殊投影原生 LoRA、MLA 增量、PCP/FusedMC2/EPLB 组合；replay 遇 LoRA 关闭、PCP LoRA NotImplementedError、dynamic EPLB/FusedMC2 断言在 10-23 main 原样保留。10-16 起 [#17681](https://github.com/vllm-project/vllm-ascend/pull/17681) 将 Model Runner V2 设为全配置默认，LoRA+MRV2 链路需回归确认。

## 2. 当前代码基线

本报告依据三个仓库 2026-10-23 12:03 fetch 的 origin/main 固定提交，不包含开放 PR 或其他开发分支。

表注：HEAD 日期是代码提交日期；支持结论限定到对应提交，不能替代模型实测或发行版本承诺。

| 仓库 | main HEAD | HEAD 日期 |
|---|---|---|
| SGLang | `f4a2df3dac40290dd35fff35c0dae4d227cf6ce1` | 2026-10-23 |
| vLLM | `19c0e170d257dc654b500ecccb5e6d3f153366bd` | 2026-10-23 |
| vLLM-Ascend | `db8836752a17abc9a26212d2b6e075b72dd4e2ab` | 2026-10-23 |

## 3. 按模型、模块判断支持与待办

表注：有实现/用例不等于所有 target_modules、并行和图模式都通过。以下"待补"仅在训练 adapter 实际包含该目标时成立；不把模型结构中的 q_lora_rank/o_lora_rank 当作外部 PEFT 支持证据。

| 模型结构 / 代表模型 | SGLang main | vLLM main | Ascend main | 核心工作 |
|---|---|---|---|---|
| 普通 GQA/MHA Dense：Llama、Qwen | 线性/QKV/MLP wrapper 已有 | 普通及 fully-sharded wrapper 已有 | 有上游包装与 NPU 优化包装 | 复用；partial targets、TP/KV 复制与非零 adapter 对照 |
| MoE：Qwen3 MoE、shared experts | 有 MoE wrapper/runner；新栈 #45892 含 plan-driven MoE 引擎 | 有 2D/3D 包装；#61235 切片修复、#61284 fully-sharded 防护已合入 | TP/EP、shared Dense 包装已有 | 不重复包装；关注 DP slot 一致性（#61714/#63077）、fully-sharded/图与融合后端 |
| MLA：DeepSeek V2/V3/R1/V3.2、Kimi K2 等 | 新栈 #45892 含 dense MLA correction；旧 #42875 已 Closed | kv_b 被吸收路径/旧引用绕过，#48974 未解决；#56327/#56718/#61804 仍 Open | 直接 W_UK/W_UV 路径仍需补增量与验证；`vllm_ascend/lora` 无任何 MLA/kv_b 处理 | 优先参考 #56327/#56718；Ascend 先加 kv_b-only 显式拒绝/告警 |
| DSA indexer / GLM-5.3-Flash | 有 indexer LoRA 目标与包装基础；#44862 合入非 main 分支 | #56327 补六路 KDA、MLA A 投影、wk_weights_proj 和视觉映射，仍 Open | 训练名称与融合 slice/执行路径需核查 | 对齐 A/B 名称→slice→forward；先映射后新建 wrapper |
| GDN：Qwen3.5、Qwen3-Next | 有 Qwen3.5 logprob 用例；#44704（hybrid MoE GDN PEFT adapter）Open | 已有映射；不等于支持任意卷积目标 | 已有 TP/fully-sharded/ACLGraph/MTP 组合用例 | 分模型逐模块数值回归；#17834 根因跟随 |
| Step5（Step-5-Preview） | main 无；sglang-miles #44917 port Open | #62498 已合入（attention q/k/v/g、sparse indexer 投影、dense MLP、routed experts；fused 投影独立 TP 几何） | 本次未做专项验证 | 上游已支持不等于 NPU 通过；按业务决定是否纳入 |
| NemotronH_Nano_VL_V2 | #39798 已进入 main，修复语言/视觉筛选 | 多模态各目标需按实现判断 | 本次未做完整专项验证 | 将"语言侧支持"和"tower/connector 支持"分开 |
| BERT/RoBERTa/XLM-R embedding | 本次未做对应专项验证 | #61033 embedding、#61391 token classification、#61279 cross-encoder 已合入 | 具备上游模型基础和 full-linear NPU 接口，未证实全模型通过 | 若业务需要，补 embedding/分类任务 NPU E2E |
| Qwen-VL 系列 | 不从语言侧用例外推整个视觉 tower | #62224（文本 LoRA 应用到 wrapper 的 LM）、#62906（batch 默认 LoRA 按会话）已合入 | Qwen2/2.5/3-VL 视觉相关用例已有 | token counts、encoder cache 与 adapter 版本隔离 |

### 3.1 DeepSeek V4.1：确定工作范围与上游方案

**对应实现方案**： [vLLM #57245](https://github.com/vllm-project/vllm/pull/57245)（Open，分组输出的 LoRA 增量、compressor FP32 输出增量、fully-sharded A gather、文本/视觉注册及按原始图片 batch index 设置 metadata；排除 MTP draft head）与 [SGLang #45231](https://github.com/sgl-project/sglang/pull/45231)（Open，attention 投影含分组 wo_a 与 shared expert，分组 wo_a 语义和 wq_b 消歧有前置依据）。两方案均未合入；Ascend 应对照选定设计并验证 NPU 差异，不能据其作者测试声称 Ascend 已通过。

表注：下列是 Ascend 固定 main 的代码审查结论，不是实测故障结论；mHC/Engram 只在训练范围要求时立项。

| 模块 | 现状与缺口 | 确定需要的工作 | 跟踪方式 |
|---|---|---|---|
| wo_a 分组输出 | ColumnParallelLinear，但 DSA o_proj 各路径直接以 npu_transpose_batchmatmul / quant 变体消费 `wo_a.weight`，普通 forward 包装覆盖不到 | 复用 A/B 管理，在实际输出计算注入按 group/token/slot 的增量；检查 TP | 对照 #57245 与 #45231 选定设计；Ascend 需 NPU 适配和数值验证 |
| compressor.wkv / wgate | 裸 nn.Linear（fp32/bf16），不在上游标准 LinearBase 自动发现范围 | 接入发现、加载、包装；在压缩计算中加增量并保留正确 dtype | 参考 #57245；Ascend 不能直接照搬 GPU fused module 形状 |
| indexer.wk（创建该模块的配置） | nn.Linear；wq_b、weights_proj 则有 Replicated 基础 | 核对训练目标；需要 wk 时显式接入并验证 top-k 前增量 | 纳入 Ascend 模型专项；未确认直接专项 PR |
| 普通 wq_a/wq_b/wkv/wo_b、MLP/专家 | 有通用包装基础，融合/保留引用需检查 | 先做非零 adapter 分组回归，不预设全部需要新 wrapper | 模型 E2E 范围 |
| mHC / Engram gate | 裸 Parameter，不是普通线性模块 | 先确认训练是否适配；再定义低秩语义与执行接口 | 条件项，不列为本期必做 |
| decoder bounded replay | 新代码显式以 LoRA metadata 未对齐为由关闭 replay；#19321 已打通 PCP×replay，但 LoRA 排除不变 | 只有业务需要 replay+LoRA 才扩展裁剪行映射 | #17496 已合入，但不含该组合 |

V4.1 的压缩、分组输出和共享 KV 结构不能直接套用 V3 的 kv_b 修正。接口应拆为"已有 wrapper 管理 A/B"和"特殊计算接入增量"，避免重复存储权重。

## 4. 按特性判断

表注：区分运行后端和边界；不将 H200/Hopper 性能数字直接转换为 Ascend 性能差距。

| 特性 | 当前结论 | 待完成工作 |
|---|---|---|
| 动态加载、同批多 LoRA、基础/adapter 混跑 | 三项目已有基础；vLLM 本期集中修复生命周期（sleep 恢复 #62358、失败激活重试 #61618、config 拒绝 #61884、max_loras #60306、AsyncLLM 重载 #41482） | Ascend 对应路径按阶段说明第 6.2 节自查清单核对 |
| adapter 加载与计算重叠 | SGLang 已有 overlap loader；另两者本次未确认同等完整功能 | 仅在换 adapter 成为瓶颈时扩展 |
| rank_pattern / alpha_pattern | vLLM #59801 已合入；Ascend 依赖上游加载能力 | Ascend 高 rank/kernel/分片组合另验，不把 loader 支持等同于端到端支持 |
| fully-sharded MoE / shared stream / DP | #61284 已合入（fully-sharded 下禁用 shared-expert aux stream）；#61301、#61714、#63077、#14049、#13490 仍 Open | 指定模型、backend、TP/DP/EP 与图模式，不宣称任意组合支持 |
| SGLang plan-driven 执行栈 | #45892 已合入 main（dense/MoE plan+runner、CuTeDSL SM90/SM100 BF16/FP8 与 Marlin NVFP4 provider、MLA correction 集成） | 默认开关与 plans 覆盖待确认；NPU 路径未验证，不据此宣称 Ascend 新栈已支持 |
| Ascend FusedMC2 / dynamic EPLB + LoRA | 仍有显式拒绝（10-23 代码核验） | 融合 FFN 增量接口；专家迁移同步 adapter 布局 |
| Ascend MRV2 PCP + LoRA | 仍明确 NotImplementedError；PCP×replay 已打通（#19321） | 对齐分片 token、adapter metadata、图捕获；可与 replay+LoRA 共用行映射设计 |
| V4.1 decoder bounded replay + LoRA | LoRA 导致 replay 被关闭并提示（10-23 代码核验） | 该组合未支持；不代表模型所有 LoRA 都被拒绝 |
| MRV2 默认化 | 10-16 起 #17681 全配置默认 Model Runner V2 | LoRA+MRV2（metadata、graph 捕获、ACL 图）回归确认 |
| SGLang embedding 新词表 / input-scattered | 新增词表仍拒绝；embedding/LM head 的 TP input-scattered 受限；#44924（partial embedding slot 清理）Open | 补词表及正确分片；不应称 embedding 全部未支持 |
| VL tower/connector | 框架与部分模型用例已有；#62224/#62906 修复语言侧应用与 batch 默认选择 | 逐图/逐模态映射、encoder cache 跳过行与失败恢复 |
| 投机解码 / MTP | 已有部分模型用例；#63186（drafter 不应被 config 拒绝）Open；#62999 Closed 未合入 | 验证 logits 行级 LoRA 映射、接受/拒绝后状态恢复 |
| 运行时 merge/unmerge | SGLang #44354 仍 Draft，初期只含 Dense 投影 | 不计为 main 支持；评估切换 drain、缓存清空和权重副本成本 |

## 5. 核心 PR、RFC 与问题单

表注：以下为当前核心工作对应的公开 PR，状态核验于 2026-10-23；完整阶段跟踪表在独立进展说明中。Open 不计为已支持。

| 工作方向 | PR | 当前状态与范围 |
|---|---|---|
| V4.1 特殊投影与文本/视觉 | [vLLM #57245](https://github.com/vllm-project/vllm/pull/57245) | Open/main；分组输出、compressor、图像 metadata |
| V4.1 attention + shared expert | [SGLang #45231](https://github.com/sgl-project/sglang/pull/45231) | Open/main；分组 wo_a、wq_b 消歧、shared expert，与 #57245 平行 |
| GLM-5.3 与 MLA prefill | [vLLM #56327](https://github.com/vllm-project/vllm/pull/56327)、[#56718](https://github.com/vllm-project/vllm/pull/56718) | Open/main；组合修复，完整模型 parity 未确认 |
| MLA 未生效的显式保护 | [vLLM #61804](https://github.com/vllm-project/vllm/pull/61804) | Open/main；拒绝或告警，不是增量修复 |
| SGLang LoRA 执行栈 | [SGLang #45892](https://github.com/sgl-project/sglang/pull/45892) | **已合入 main（10-23）**；21 部分堆叠栈单 PR 替代，后续跟踪其 follow-up |
| lm_head prompt logprobs | [vLLM #62049](https://github.com/vllm-project/vllm/pull/62049) | Open/main；修复 #51594，对应 RFC #48297 logprobs 工作流 |
| DP slot 一致性 | [vLLM #61714](https://github.com/vllm-project/vllm/pull/61714)、[#63077](https://github.com/vllm-project/vllm/pull/63077) | Open/main；idle DP 保留 adapter 与 MoE slot 重映射为同族后续 |
| packed/QKK 映射 | [vLLM #61267](https://github.com/vllm-project/vllm/pull/61267)、[#62034](https://github.com/vllm-project/vllm/pull/62034) | Open/main；attention_k_eq_v 的 K 与 V slice 两半 |
| GDN/TP 小 slice | [Ascend #17834](https://github.com/vllm-project/vllm-ascend/pull/17834) | Open；需结合上游根因复现，不直接采纳 workaround |
| MoE fully-sharded 与图 | [Ascend #14049](https://github.com/vllm-project/vllm-ascend/pull/14049)、[#13490](https://github.com/vllm-project/vllm-ascend/pull/13490) | Open/main；不是任意组合已支持 |

表注：下列本次重新核验为 Open；RFC 是需求/设计跟踪，不是实现状态。没有关联专项 PR 时不虚构编号。

| 入口 | 用途 |
|---|---|
| vLLM [RFC #48297](https://github.com/vllm-project/vllm/issues/48297) | LoRA Correctness and Lifecycle for RL Training：集中跟踪 MLA、DP、logprobs、投机解码、重载缓存与 adapter-only 同步；建议作为主跟踪入口 |
| vLLM [Issue #48974](https://github.com/vllm-project/vllm/issues/48974) | MLA kv_b decode 吸收 + prefill 旧引用；跟踪 prefill/decode 两条应用路径 |
| Ascend [RFC #13220](https://github.com/vllm-project/vllm-ascend/issues/13220) | MoE 架构对齐，涉及 LoRA metadata 与组件边界；不是 MC2/EPLB LoRA 已实现证明 |
| Ascend [贡献任务 #14208](https://github.com/vllm-project/vllm-ascend/issues/14208) | DS/Kimi/GLM/MiniMax/Step LoRA 层、权重加载、多 LoRA 切换；用于沟通任务范围与认领，内容路径可能早于当前重构 |

历史 #396/#2929 已关闭，不作为当前开放待办。router-driven adapter mixture 与每请求单 adapter 的 multi-LoRA 是不同需求，未纳入本期核心实现。

## 6. 现在可以确定的核心工作

表注：P0 为正确性与目标模型阻塞；"代码未接通"不等于本次已跑出错误。开发前先核对训练 adapter 的实际 keys/target_modules，避免开发未训练参数的适配。vLLM-Ascend 逐项待办与自查清单细化见 [阶段进展说明第 6 节](./LoRA阶段进展与PR跟踪_2026-10-16至2026-10-23.md)。

| 优先级 / 工作包 | 已确认未完成 | 建议实施内容 | 验收 |
|---|---|---|---|
| P0：V4.1 原生 LoRA | Ascend wo_a、compressor 与部分 indexer.wk 尚缺完整标准接入；#57245 与 #45231 均未合入 | 对照两方案选定设计；NPU 分组输出增量、nn.Linear 接入、dtype/TP/图像 metadata；不重复独立实现上游同方案 | 分模块非零 adapter 与 merge/reference logits 对照；普通/TP、prefill/decode |
| P0：MLA 完整增量 | 主线问题仍 Open；`vllm_ascend/lora` 无任何 MLA/kv_b 处理 | 先加 kv_b-only adapter 显式拒绝/告警（对齐 #61804）；再补 Q/V 增量及 prefill 实时 wrapper/缓存行映射 | kv_b-only、混合 adapter、cached prefill、替换 adapter 后版本一致 |
| P1：GDN/packed/分布式组合 | TP 小 slice、partial targets、DP slot/图模式不能一概视为安全；MRV2 已默认化 | 先吸收或复现已有 PR；Qwen3.5/Next、MoE 逐组合验证；PCP/MC2/EPLB/replay 按业务选择一个扩展；LoRA+MRV2 回归 | TP2/4、rank 边界、非零 A/B、graph/eager、不同 adapter 不串用 |

接口流程：

```text
adapter config + A/B
  → target names / packed mapping
  → wrapper 选择、TP/EP 分片、set_lora
  → token / logits row / image / replay row → adapter slot/version
  → 普通投影增量 或 MLA/分组输出/融合专家的专用增量
  → cache/recurrent state 隔离 → logits 与更新生命周期验收
```

暂不固定为必做：mHC/Engram gate 专用低秩语义、单请求多 adapter 混合、全部量化后端、所有模型的全部组合。先根据训练与部署需求决定范围。

## 7. 固定提交代码证据与验证边界

- SGLang：[legacy MLA Q/V 修正](https://github.com/sgl-project/sglang/blob/4e85e8ed3738ac9d7686946fb6db7242bc66a2a9/python/sglang/srt/lora/deepseek_mla_correction.py#L62)、[模块筛选/词表限制](https://github.com/sgl-project/sglang/blob/4e85e8ed3738ac9d7686946fb6db7242bc66a2a9/python/sglang/srt/lora/lora_manager.py)、[embedding input-scattered 限制](https://github.com/sgl-project/sglang/blob/4e85e8ed3738ac9d7686946fb6db7242bc66a2a9/python/sglang/srt/lora/layers.py#L123)。
- vLLM：[通用 wrapper 选择与模块发现](https://github.com/vllm-project/vllm/blob/dd9e07a25febb05f1a44f2ef9e6c56803ee1a9b6/vllm/lora/utils.py#L105)、[sharded apply 合约](https://github.com/vllm-project/vllm/blob/dd9e07a25febb05f1a44f2ef9e6c56803ee1a9b6/vllm/lora/layers/column_parallel_linear.py#L24)、[rank/alpha pattern](https://github.com/vllm-project/vllm/blob/dd9e07a25febb05f1a44f2ef9e6c56803ee1a9b6/vllm/lora/peft_helper.py)、[V4.1 compressor 直接权重路径](https://github.com/vllm-project/vllm/blob/dd9e07a25febb05f1a44f2ef9e6c56803ee1a9b6/vllm/models/deepseek_v41/attention.py)。
- Ascend：[V4.1 wo_a 直接读取说明](https://github.com/vllm-project/vllm-ascend/blob/c5766cd5e3912858a61f250f8201da9bcc3a6cff/vllm_ascend/models/deepseek_v41/model.py#L437)、[compressor nn.Linear](https://github.com/vllm-project/vllm-ascend/blob/c5766cd5e3912858a61f250f8201da9bcc3a6cff/vllm_ascend/models/deepseek_v41/compressor.py#L21)、[indexer](https://github.com/vllm-project/vllm-ascend/blob/c5766cd5e3912858a61f250f8201da9bcc3a6cff/vllm_ascend/models/deepseek_v41/indexer.py#L75)。
- Ascend 组合限制：[MoE MC2/EPLB](https://github.com/vllm-project/vllm-ascend/blob/c5766cd5e3912858a61f250f8201da9bcc3a6cff/vllm_ascend/lora/fused_moe.py#L170)、[PCP](https://github.com/vllm-project/vllm-ascend/blob/c5766cd5e3912858a61f250f8201da9bcc3a6cff/vllm_ascend/worker/v2/pcp_manager.py#L238)、[V4.1 replay](https://github.com/vllm-project/vllm-ascend/blob/c5766cd5e3912858a61f250f8201da9bcc3a6cff/vllm_ascend/models/deepseek_v41/decoder_replay_layers.py#L122)、[full-linear NPU 接口](https://github.com/vllm-project/vllm-ascend/blob/c5766cd5e3912858a61f250f8201da9bcc3a6cff/vllm_ascend/lora/punica_npu.py#L650)。

表注：以上为 10-16 报告快照的固定 commit 链接。10-23 复核已重新核验 Ascend 侧锚点在 `db8836752` 全部不变（wo_a 直接消费注释、compressor/indexer nn.Linear、三项组合限制、punica_npu 负索引跳过语义），并确认 `vllm_ascend/lora` 无 MLA/kv_b 处理；SGLang/vLLM 锚点保留 10-16 快照链接，引用时以对应基线提交为准。

验证边界：本期（10-16 → 10-23）完成三仓库 fetch 与提交扫描、GitHub 页面状态核验、Ascend 关键代码路径复核；没有安装依赖、没有运行模型 E2E/性能测试、没有提交代码或创建 PR/RFC。不能从"PR 作者说通过"推导本地新依赖组合已验证。
