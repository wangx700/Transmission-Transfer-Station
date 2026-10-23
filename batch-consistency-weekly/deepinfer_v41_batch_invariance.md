# DeepInfer / DeepSeek-V4.1：Batch 不变性流程、算子与 Ascend 复用说明

整理日期：2026-10-23（北京时间）。分析基线为 DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b`，不是对当日最新主线的声明。本文是固定版本源码调研，不属于周报时间窗内的新增提交统计；未运行 GPU/NPU 数值或性能测试。

## 1. 结论与版本

DeepInfer 的 NVIDIA batch 不变性通过算子库确定性模式、Attention split-KV 控制、特定融合分支选择及 MoE combine 归约约束共同实现。Ascend 已有部分算子适配，但完整功能仍被框架禁止，不能通过删除断言就认定支持。

固定子模块版本：DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`，FlashMLA `2e5429fc5653bab6e081f09477126f731882a6a9`，DeepEP `93eb6eb238127e96c6d7a4a625a6dad158348509`，DeepGEMM-Ascend `3a35326271b069b8b5bf1a754435acd6cc91575b`，DeepEP-Ascend `3b25377d04b24fc6154698ded78a2bcb2c59afff`。

“同一输入重复运行确定性”“不同 batch 下模型数值不变”“最终随机生成相同”是不同要求。最后一项还需要相同序列种子、采样参数和历史；固定生成 token 只能对齐历史，不能单独证明数值相同。

## 2. 主干接口流程与算子

### 与 vLLM 主线的支持口径

2026-10-23（北京时间）补充核查：DeepInfer 官方 README 将 DeepSeek-V4.1-Flash 列为支持模型，并明确 BI 仅 NVIDIA 支持；这是框架声明与实现证据，本文未独立复现。Ascend 仍不属于完整支持范围。

vLLM 已支持 V4.1 模型推理，但不能等同于已经完整支持 V4.1 的 batch 不变性。核查时其官方 BI 已验证模型名单没有 V4/V4.1；V4 专项预览 #54955、request-local sparse prefill #60442、确定性 sparse TopK #60453 和 Inductor 配置固定 #60074 均为 Open、未合入。因此准确表述是“vLLM 主线尚无本次可核实的 V4.1 完整 BI 支持/验收结论，专项适配仍在推进”，而不是“vLLM 没有 BI 功能”或“不能推理 V4.1”。V4 专项 PR 的存在也不能自动证明其方案已覆盖 V4.1。

来源：[DeepInfer README](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md)、[vLLM BI 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/batch_invariance.md)、[V4.1 推理跟踪](https://github.com/vllm-project/vllm/issues/56400)、[专项预览 #54955](https://github.com/vllm-project/vllm/pull/54955)、[#60442](https://github.com/vllm-project/vllm/pull/60442)、[#60453](https://github.com/vllm-project/vllm/pull/60453)、[#60074](https://github.com/vllm-project/vllm/pull/60074)。PR 状态由 GitHub API 的 `state` 与 `merged` 字段核对；动态页面后续可能变化。

```text
DeepseekModel.forward(input_metadata, kv_caches, ...)
  → Token Embedding；准备 Engram 与跨层共享状态
  → 逐层 DeepseekDecoderLayer.forward()
      → attn_norm_forward：mHC Pre/Post Mix、Prenorm GEMM、RMSNorm；特定层插入 Engram
      → V4Attention.forward()
          → Q / SWA KV / 压缩 KV 投影
          → 压缩状态更新、Norm、RoPE、量化与 KV cache 读写
          → 稀疏层 Indexer：MQA logits → TopK KV 选择
          → Prefill attention 或 paged Decode attention → 输出投影
      → ffn_norm_forward：mHC 与 FFN 归一化
      → DeepseekMoE.forward()
          → Gate / TopK / EPLB 逻辑到物理专家映射
          → EP Dispatch → Grouped GEMM → SwiGLU/量化 → Grouped GEMM
          → Shared Expert → EP Combine
  → 最终 mHC 收束与 Output Norm
  → LM Head / Sampler → token、logprob
```

这是逻辑流程；MegaMoE、MegaMHC 和 Attention 融合路径会合并其中步骤，具体层功能取决于配置。

证据：[模型 forward](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/deepseek.py#L154)、[DecoderLayer](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/decoder_layer.py#L103)、[Attention](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/attn.py#L752)、[Indexer](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/index_attn.py#L516)、[MoE](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/mlp.py#L565)。

表注：下表的“风险”是定位方向，只有后文给出明确证据的项才视为已确认机制；不能据此认定所有列出的 Ascend 算子存在问题。M 指当前算子处理的 token 行数，K 指归约维度。

| 环节 | 算子 | Batch 变化可能影响的内容 |
| --- | --- | --- |
| mHC | Prenorm GEMM、平方和、Sinkhorn、Mix、RMSNorm | K 分片、部分和累加顺序 |
| 投影与输出 | BF16 / FP8 / FP4 GEMM、LM Head | M 相关的算法、tile 和归约策略选择 |
| KV 路径 | Compress、Norm、RoPE、量化、Store/Load | 分块与状态边界、量化归约及融合路径 |
| Indexer | MQA logits、TopK | 打分差异可能改变 KV 入选集合；并列值规则也应验收 |
| Attention | Sparse Prefill、Paged Decode、融合 kernel | Split-KV 和 softmax 部分结果归并 |
| MoE | Gate、Grouped GEMM、SwiGLU、Dispatch/Combine | 专家 token 数、专家物理位置与归约分组 |
| 采样 | 概率变换、随机采样 | 输入 logits 一致性，以及随机数是否绑定序列身份 |

浮点加法不满足结合律。Batch 改变并行任务量，若进而改变分片和归约树，同一 token 也可能产生不同数值；TopK、专家路由和采样会进一步放大差异。

## 3. 当前实现采取了什么措施

### 3.1 Worker 与 GEMM

接口：`Worker.init_model → configure_worker_kernels → deep_gemm/tile_kernels.use_deterministic_algorithms(True)`。NPU 在此首先断言禁止 `DEEPINFER_BATCH_INVARIANT`；即使不开 BI，NPU 也启用这两个库的确定性模式以减少 CI 数值不稳定。

[runtime_config.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/runtime_config.py#L54)：

```python
if PLATFORM.is_npu:
    assert not envs.DEEPINFER_BATCH_INVARIANT
if envs.DEEPINFER_BATCH_INVARIANT or PLATFORM.is_npu:
    deep_gemm.use_deterministic_algorithms(True)
    tile_kernels.use_deterministic_algorithms(True)
```

NVIDIA DeepGEMM 中，确定性模式会绕过特定 BF16 cuBLASLt 分支、令 MegaGate 不切 K、令 MegaMHC 使用固定 split 数，而不是把所有算子一律改为单分片。证据：[BF16 GEMM](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/apis/gemm.hpp#L518)、[MegaGate](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/jit_kernels/heuristics/mega_gate.hpp#L110)、[MegaMHC](https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/csrc/apis/mega_mhc.hpp#L15)。这些 CUDA 实现不能直接作为 Ascend 同名接口已完成适配的证据。

### 3.2 Attention

接口：`V4Attention._decode_attention → decode_mla_mqa_sparse(batch_invariant=...) → flash_mla_with_kvcache(enable_batch_invariant=...) → sparse_decode.cpp`。

NVIDIA 分支用 `!enable_batch_invariant && !(topk + extra_topk <= 640)` 决定是否启用 split-KV；BI 开启后关闭。Ascend 分支本来就固定 `enable_split_kv=false`，所以“Ascend 没有关闭 split-KV”不是其完整 BI 缺失的理由。[后端传参](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/attention_backend.py#L125)、[FlashMLA 分支](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/sparse_decode.cpp#L219)。

另外，BI 会撤销 dense packed-KV 层在 PROMPT 引擎上的融合放行条件；并非关闭所有融合。Fused decode 提前返回、没有透传 BI 参数，但它底层本身不支持 split-KV，不能据未传参就认定遗漏。[融合选择](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/v4_modules/attn.py#L87)、[融合内核](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp#L378)。

### 3.3 MoE combine 与明确的 Ascend 冲突

接口：`userbuffer.allocate_ep → EPBuffer(...) → dispatch → 专家计算 → combine`。

框架传入：

```python
allow_multiple_reduction = not (
    envs.DEEPINFER_BATCH_INVARIANT
    and envs.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL > 0
)
```

即 BI 与周期性专家重排同时开启时，要求单次归约。DeepEP 的参数说明指出关闭多次归约后只在 combine epilogue 归约一次，但可能增加传输量；结合专家重排条件，可推断其作用是避免物理专家布局变化影响归约分组。[框架代码](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/parallel_utils/userbuffer.py#L71)、[参数契约](https://github.com/deepseek-ai/DeepEP/blob/93eb6eb238127e96c6d7a4a625a6dad158348509/deep_ep/buffers/ep.py#L220)。

DeepEP-Ascend 则明确 `assert allow_multiple_reduction, 'Ascend requires allow_multiple_reduction=True'`。[代码](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/buffers/ep.py#L207)。因此即便绕过 Worker 断言，该配置组合仍会失败。它不证明关闭专家重排后一定通过，也不证明所有其他配置都必然数值失败。

## 4. Ascend 当前可以复用什么

这里的“直接复用”限于匹配当前 DeepInfer、依赖版本和硬件环境时调用已有实现；迁入 vllm-ascend 仍需对齐 ABI、布局、shape、流和执行图接口，不表示可以直接复制后宣称整模型 BI。

表注：状态依据上文固定提交的源码及测试代码；没有本次真机通过率或性能数据。“测试存在”表示有可复用的验收设计，不表示本次已执行通过。

| 能力 | 可复用内容与接口流程 | 复用等级与限制 |
| --- | --- | --- |
| HC prenorm 确定性计算 | `tf32_hc_prenorm_gemm → get_config(deterministic=True) → 固定 K 分片策略 → 按 split_id 分阶段累加` | 已有 Ascend 实现；可在原栈调用，迁移时适配形状/布局与全局开关 |
| HC prenorm 算子测试 | 重复运行逐位比较；从完整 batch 抽取行单独计算再逐位比较 D 和平方和 | 可复用测试思路及 shape 集；不是整层/整模型验收 |
| 按序列身份采样 | `SamplingParams.seq_seed → prepare_sample → SamplingKernelInputs → Ascend random_sampling_tl`；随机量由 seed、position、sample_case 决定 | 已有 Ascend kernel；迁移时保持 seed、位置、概率变换语义，不能仅换 kernel |
| EP dispatch 确定性排序 | `EPBuffer(deterministic=True) → dispatch → finalize_dispatch → handle.deterministic_sort` | 已有排序能力；要显式启用并处理 overlap hook，不能代替 combine 的归约保证 |
| Attention 非 split-KV 路径 | `flash_mla_with_kvcache → Ascend sparse decode → enable_split_kv=false` | 已有路径，可作为数值基线；其余数值、padding 与 Prefill/Decode 一致性仍需测试 |
| 模型级固定轨迹对比 | 批量 generate → 保存 token → 单条 golden_token_ids 回放 → 比较真实 logprobs/TopK | 可复用方法；当前 NPU BI 开关被禁止，原测试不能直接当作 Ascend 已通过 |
| TileLang 后端分派 | `ops.py → PLATFORM.tilelang_ops_backend → asc/*_tl` | 可复用接口分层与逐算子替换方式；不是每个导出算子都已证明 BI |

### 4.1 HC prenorm：已经解决的具体问题

普通模式的 `num_k_splits_by_workload` 根据 `num_token_blocks` 变化；源码明确说明这会改变每行浮点归约分组、破坏 BI。确定性模式将工作量上限改为 16，再结合 hidden 大小与核数约束得到实际分片，移除对 M 的依赖。同时 `num_store_phases=num_k_splits`，每阶段只允许对应 split 写回并同步，固定累加顺序。

证据：[配置与原因注释](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/deep_gemm/tilelang_ops/hc_prenorm.py#L13)、[阶段化写回](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/deep_gemm/tilelang_ops/hc_prenorm.py#L295)、[开关传递](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/deep_gemm/tilelang_ops/hc_prenorm.py#L392)、[逐位测试](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/tests/test_hyperconnection.py#L101)。

注意两点：确定性路径仍使用原子加，通过阶段同步约束顺序，不能概括为“删除原子加”；框架 `mhc_ops.py` 中 Ascend 的公开 `n_splits=1` 表示后端内部已归并，只输出一份结果，不表示 kernel 内部没有 K 分片。[公开接口](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/deep_gemm/tilelang_ops/hc_prenorm.py#L363)。

### 4.2 采样与 EP 排序的复用边界

Ascend 采样实现逐行读取 `seq_seeds[bx]`、`sample_positions[bx]` 与 `sample_case` 调用 `draw_random`。可复用的是“随机数绑定序列和位置”的语义，不能把 batch 行号、请求到达次序当作随机身份。[Ascend kernel](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/asc/random_sampling.py#L207)、[C++ 输入准备](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/prepare_sample.cpp#L161)、[采样重排/单行测试](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/random_sampling.py#L310)。固定输出还要求 logits 一致；官方可复现输出配置要求固定 seq_seed、开启 BI 并关闭 MTP varlen，不能把该完整保证外推到 Ascend。[官方配置](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/README.md)。

DeepEP-Ascend 排序按源 token 整理 expanded 接收布局，并更新缓存 handle；启用 deterministic 时在 finalize 或 overlap hook 中执行。DeepInfer 当前创建 EPBuffer 的位置没有传入 `deterministic=True`，所以“库里存在”不等于“框架此处已启用”。[排序实现](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/buffers/ep.py#L100)、[调用位置](https://github.com/deepseek-ai/DeepEP-Ascend/blob/3b25377d04b24fc6154698ded78a2bcb2c59afff/deep_ep/buffers/ep.py#L485)。

后端分派证据：[ops.py](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/layers/tilelang_ops/ops.py)。迁移时可保留统一算子接口，在 Ascend 后端实现并逐项验证，避免把 CUDA kernel 当作 NPU 实现。

## 5. Ascend 尚缺什么，怎样落地

已经确认的缺口是 Worker 明确禁止完整 BI，以及 BI+专家重排需要的单次 combine 归约接口不被 DeepEP-Ascend 接受。普通 GEMM、Grouped GEMM、Indexer、量化和融合算子需要逐项做跨 M/布局测试，不能仅凭存在确定性开关就宣称通过，也不能在没有实测时认定它们全有问题。TileKernels 在本仓库中只有依赖与调用证据，内部完整覆盖不在本次核查范围。

建议先复用 HC prenorm 与采样算子级测试；再对齐 EP 单次归约契约，并分别验收“同一专家布局”和“专家重排后”两种情况。用固定权重、输入历史、并行配置，从 HC 输出、Q/KV、Indexer logits/indices、Attention 输出、Gate 路由、专家输出、Combine、最终 logits 逐级找首个差异。跨 batch 检查至少包含 BS1/BSN、请求重排、padding、eager/graph、同图档/跨图档；Prefill/Decode、压缩状态与跨层缓存另作组合覆盖。基础主干通过后再扩展 MTP/varlen 和 Engram 相关路径。

移除 Worker 断言应在这些契约和回归通过之后，不能作为第一步功能修复。相关 vllm-ascend 待做项见[整体报告](sglang_vllm_ascend_consistency_report.md)；本文不代表已经完成移植。

## 6. 测试语义与证据边界

[模型级测试](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/tests/models/test_batch_invariant.py#L52) 先批量运行，再将输出 token 作为单条运行的 `golden_token_ids`，严格比较 token、logprobs、top_token_ids、top_logprobs。其价值是对齐上下文后验证真实数值，不是证明两次自由随机生成相同，更不是 Ascend 已通过的证据。

本报告中的“改动点”指该固定版本确定性分支相对普通分支的代码行为。公开主仓库基线只有初始发布和文档更新，本文没有将这些分支虚构为周报期间的新补丁。无硬件测试、无性能测量，不对端到端逐位一致性作独立验收结论。
