# DSA Index Replay 当前整体支持与适配报告

状态日期：2026-10-23（北京时间）。范围：vLLM、vLLM-Ascend、SGLang；不涉及训练框架实现。依据官方 main（本地已同步至 10-23 HEAD）和 GitHub PR/RFC，未执行硬件验证。

## 1. 核心结论

- **SGLang**：已有 CUDA/NPU 索引采集、请求级保存和原生接口返回，可作为实现参考；attention TP 和 API 覆盖仍有限制。
- **vLLM**：已有 DSA indexer，但公共索引采集与返回能力仍在 Draft PR 阶段；10-21 前后新增 [#62915](https://github.com/vllm-project/vllm/pull/62915) Profile-based indexer replay Draft，与 #49555 并行，当前主线未检出完整链路。
- **vLLM-Ascend**：已有 NPU 索引计算基础，需同时跟进 vLLM 公共能力并补后端采集、runner 输出与图执行适配。

本报告中的 index replay 指推理记录实际选中的稀疏 KV 索引并返回，供训练回放。它不等于 MoE router replay、图执行 replay 或 SWA bounded replay；仅有返回接口也不能证明训练回放闭环完成。

## 2. 当前支持情况


| 能力 | vLLM | vLLM-Ascend | SGLang |
|---|---|---|---|
| DSA indexer / 稀疏 Attention | 已有 | 已有 NPU 实现 | 已有 CUDA/NPU 实现 |
| 逐层索引采集、请求保存与返回 | 主线未检出完整链路；#49555/#62915 均为 Draft | 主线未检出完整链路 | 已有 capturer 和原生 /generate 返回 |
| NPU 采集与异步输出 | 需硬件插件适配 | 待实现 | #39060 已于 09-28 合入，非本期新增 |
| attention TP > 1 的采集 | 返回特性尚未合入 | 待实现与验证 | capturer 仍要求 attn_tp_size == 1 |
| OpenAI 接口返回索引 | #49555/#62915 提案能力，非主线能力 | 待协议与后端接入 | 当前 OpenAI 目录未检出对应字段；现有测试说明只覆盖 /generate |

表格注释：本次已同步官方 main：vLLM 19c0e170d2（提交日期 10-23）、Ascend db8836752（10-23）、SGLang f4a2df3dac（10-23）。状态截至 2026-10-23；“未检出”限定本次源码范围。索引采集返回不等于完整训练回放。

## 3. 核心 PR / RFC 入口

| 项目 | 跟踪入口 | 当前作用和状态 |
|---|---|---|
| vLLM | [RFC #47280](https://github.com/vllm-project/vllm/issues/47280)、[PR #49555](https://github.com/vllm-project/vllm/pull/49555)、[PR #62915](https://github.com/vllm-project/vllm/pull/62915) | RFC Open；#49555 Open/Draft 未合入；#62915 Open/Draft（10-21 前后创建，Profile-based indexer replay，含采集存储与 API 协议） |
| vLLM | [RFC #48305](https://github.com/vllm-project/vllm/issues/48305) | 训练推理一致性背景，DSA Index Replay 分项导航 |
| SGLang | [#24392](https://github.com/sgl-project/sglang/pull/24392)、[#39060](https://github.com/sgl-project/sglang/pull/39060) | 已合入，分别提供基础采集返回和 NPU 接入参考 |
| Ascend | [RFC #10072](https://github.com/vllm-project/vllm-ascend/issues/10072) | Open，RL 总路线图；提及 Indexer Replay 但混有 MoE 路由语义，不是独立 DSA 实现方案 |

表格注释：状态截至 2026-10-23；#48305 是背景导航。#49555 与 #62915 方案不同（ordinary raw selections vs profile-based），收敛结果未定。尚未确认 Ascend 独立的 DSA 采集返回实现 PR/RFC。具体阶段增量及候选 PR 见独立变化说明。

## 4. 需要推进的三项核心工作


1. **公共接口与协议**：跟进 #47280/#49555 与新增的 #62915（Profile-based indexer replay），明确两条 Draft 的收敛方向，以及开关、输出、token 范围、层映射、int32 编码、无效值，还有 V3.2 token/V4 压缩 KV 坐标。上游未合入时先做 Ascend，也需承接公共链路。
2. **NPU 采集与输出**：在实际使用 top-k 的位置接入，覆盖 prefill/decode/共享索引与 #19220 引入的 DSpark 逻辑索引路径；补 NPUModelRunner 静态 buffer、异步 D2H、事件、请求保存和 ACLGraph 生命周期，参考 SGLang #39060，并适应 #16731 算子切换与 #19190 buffer 共享后的归属。
3. **正确性与成本**：先无 CP/MTP 最小闭环，再扩展图执行、异步、prefix cache、CP/MTP、PD。验证 token 移位、局部/全局坐标、DBO 隔离、有效行与内存，不以相关 PR 合入替代组合场景验证。

接口流程：

```text
请求开启采集 → indexer 生成或共享复用 top-k → capturer 记录
→ runner 拷出 → scheduler 按请求/token 保存 → API 返回
→ 外部训练端回放（本报告范围外）
```


## 5. vLLM-Ascend：当前可做与待办事项

### 当前已具备、可直接依托的能力

| 能力 | 主线位置/依据 | 与采集返回的关系 |
|---|---|---|
| DSA/SFA indexer 与 top-k 计算 | 普通路径 [sfa_v1.py#L2078](https://github.com/vllm-project/vllm-ascend/blob/db8836752/vllm_ascend/attention/sfa_v1.py#L2078)、压缩 cache 路径 [dsa_v1.py#L2580](https://github.com/vllm-project/vllm-ascend/blob/db8836752/vllm_ascend/attention/dsa_v1.py#L2580)；MTP skip_topk 层共享复用 | 采集接入点；两条路径与共享复用场景都需覆盖 |
| DSpark 逻辑索引 + 稳定 buffer 契约 | [#19220](https://github.com/vllm-project/vllm-ascend/pull/19220)；[dsa_v1.py#L634](https://github.com/vllm-project/vllm-ascend/blob/db8836752/vllm_ascend/attention/dsa_v1.py#L634) 注释明确 ACL graph capture 冻结地址、算子回放必须从稳定 buffer 读 | 采集 buffer 可直接沿用 caller-provided 稳定 buffer 模式；[T,1] 长度契约需对齐 |
| CANN 原生 LI 算子 | [#16731](https://github.com/vllm-project/vllm-ascend/pull/16731) 已切换 `torch_npu.npu_quant_lightning_indexer` | 采集基于当前原生算子输出，不依赖已删除绑定 |
| 图执行与并行基础 | ACL graph capture/warmup；DCP decode FIA v2（[#18752](https://github.com/vllm-project/vllm-ascend/pull/18752)）；CP UT（test_dsa_cp/test_sfa_cp 等，[#18267](https://github.com/vllm-project/vllm-ascend/pull/18267) MRV2 CP 精度覆盖） | 图执行与 CP 场景组合验证的基础 |
| 传输与 offload | packed SFA C8 分层传输（[#18715](https://github.com/vllm-project/vllm-ascend/pull/18715)）、C8 K-scale 修复（[#19064](https://github.com/vllm-project/vllm-ascend/pull/19064)）、sparse KV offload 文档（[#19020](https://github.com/vllm-project/vllm-ascend/pull/19020)） | PD/传输坐标与 scale 正确性依赖 |
| 工具与测试 | `benchmarks/prepare_indexer_indices.py`（NPU graph 内索引后处理，含 int32 sentinel 无效值口径）、`quantize_indexer_query.py`；`tests/ut/attention` 的 test_indexer/test_dsa_*/test_sfa_* | 无效值编码口径与后处理性能可先期复用 |

### 待办事项（按前置条件分组）

现在就能启动（不等上游）：

1. 采集协议原型：按 #47280 语义起草开关、输出、token 范围、层映射、int32 编码与无效值（对齐 `prepare_indexer_indices` 的 sentinel 口径）；Draft 收敛前不固化对外字段。
2. 静态 buffer 与内存设计：61 层×2048 top-k×int32 约 488 KiB/token 的量级评估，纳入 #19190 共享归属；沿用 dsa_v1 稳定 buffer 模式。
3. 单卡最小闭环：无 CP/MTP 的 prefill+decode 采集→异步 D2H→请求级保存，先内部验证正确性（当前主线无此链路，属待实现）。

依赖上游收敛：

4. 公共链路与 API 对齐：#49555/#62915 收敛后接入调度器保存与接口返回；对齐 #60734 的 DBO microbatch 隔离语义。
5. CP 场景：等 #18651（A5 FP8 LI C8 + DSA CP）与 #19491 新文档口径后，再定 CP 局部行、因果长度与空分片的采集坐标。

与实现同步推进：

6. 组合场景验证：ACL graph 生命周期、异步、prefix cache、MTP（DSpark 路径与 skip_topk 共享复用）、PD（#18715 C8 分层传输坐标）。
7. 成本复核：32768 缓存 token 约 15.25 GiB（不含临时拷贝/编码）量级复核，并与 #61938 修复后的 int32 上限对齐。
8. 真机验证：本系列结论均未做 NPU 数值验证，采集实现接入后需补齐。

## 6. 当前限制与代码证据


代码证据固定到 10-23 核对所用主线提交：SGLang `f4a2df3dac`、vLLM-Ascend `db8836752`。

- SGLang attention TP=1 限制仍在：[indexer_topk.py#L33](https://github.com/sgl-project/sglang/blob/f4a2df3dac/python/sglang/srt/state_capturer/indexer_topk.py#L33)。KV pool 分片不代表该限制解除。
- SGLang CUDA/NPU 创建入口：[indexer_topk.py#L91](https://github.com/sgl-project/sglang/blob/f4a2df3dac/python/sglang/srt/state_capturer/indexer_topk.py#L91)（`create_indexer_capturer`，10-23 HEAD 上由 99 行移至 91 行）；V4 NPU 采集：[ascend_dsv4_backend.py#L906](https://github.com/sgl-project/sglang/blob/f4a2df3dac/python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py#L906)。
- 返回行数为 [start_len, seqlen-1)，最后新 token 可能尚未 forward：[base.py#L173](https://github.com/sgl-project/sglang/blob/f4a2df3dac/python/sglang/srt/state_capturer/base.py#L173)。
- Ascend 当前采集基础位置：[sfa_v1.py#L2078](https://github.com/vllm-project/vllm-ascend/blob/db8836752/vllm_ascend/attention/sfa_v1.py#L2078)、[dsa_v1.py#L2580](https://github.com/vllm-project/vllm-ascend/blob/db8836752/vllm_ascend/attention/dsa_v1.py#L2580)（`_maybe_update_compressed_caches_and_select_topk`，10-23 HEAD 上由 2328 行移至 2580 行）。
- 内存量级：61 层×2048 top-k×int32 约 488 KiB/token；32768 个缓存 token 约 15.25 GiB，不含临时拷贝/编码。SGLang [#37285](https://github.com/sgl-project/sglang/pull/37285) 优化 CUDA host-cache 分配，不消除数据量，不能直接认为 NPU 同样受益。
- 固定索引不能消除权重版本、量化和算子精度差异；采集返回测试不能证明完整训练一致性。


[10-16 至 10-23 阶段变化说明](DSA_index_replay_changes_2026-10-16_to_2026-10-23.md) 单独列出间隔内已合入 PR、修改内容、无变化事项和新增跟踪项。
