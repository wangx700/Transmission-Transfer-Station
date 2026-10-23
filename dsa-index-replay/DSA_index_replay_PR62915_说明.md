# vLLM PR #62915 说明：Profile-based indexer replay for sparse models

整理日期：2026-10-23（北京时间）。依据 PR 页面与其完整 diff（57 个文件，含新增设计文档 `docs/training/indexer_replay.md`）逐节提取；PR 处于 Open/Draft，内容可能继续变化。未执行任何推理验证。

## 1. 基本信息与定位

- PR：[#62915](https://github.com/vllm-project/vllm/pull/62915) `[Feature][AuxOutput] Add Profile-based indexer replay for sparse models`，作者 Ronald1995（与 [#49555](https://github.com/vllm-project/vllm/pull/49555) 同一作者），10-21 前后创建，10-23 核查时仍为 Draft。
- 定位：在 AuxOutput 框架上新增**稀疏注意力 indexer 选择的采集、存储与 API 返回**，并提供带语义元数据（Profile）的逐模型编码/解码体系。目前是覆盖"采集→保存→返回"最完整的公共实现。
- 与 #49555 的关系：设计文档明确 DeepSeek V4 的 ordinary raw C4A Replay 复用 #49555 打下的基础（固定宽度 int32、寻址压缩 KV 行、原样返回，无需 Profile/codec）；#62915 的 Profile 扩展为既有 V4 adapter 增加可选语义元数据，并为其余模型族提供各自布局。
- 与 RFC [#47280](https://github.com/vllm-project/vllm/issues/47280) 的关系：属于该 RFC "return indexer topk" 的一条实现路线；#49555（ordinary raw selections）与 #62915（profile-based）并行，收敛未定。

## 2. 总体结构与开关

- 新增 `vllm/model_executor/layers/indexer_capture.py`（采集层）与 `vllm/distributed/aux_output_connector/`（connector/store/profile/memory/worker/artifacts），与 R3 routed-experts 返回共用请求 ID、generation、增量 KV 兼容 hash 与 SHM 状态机；两者可独立或同时开启，indexer-only 不绑定 MoE router。
- 配置（`vllm/config/aux_output.py`）：`enable_return_indexer_topk`、`indexer_output_format: raw|packed`、`indexer_encoding: raw|compact`；启动示例：

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve deepseek-ai/DeepSeek-V3.2 --enable-return-indexer-topk
--aux-output-config '{"enable_return_indexer_topk":true,"indexer_encoding":"compact","indexer_output_format":"packed"}'
```

- 多轮去重：`SamplingParams.indexer_topk_prompt_start`（跳过已返回的 prompt 位置），与 `routed_experts_prompt_start` 相互独立。

## 3. 支持的模型与索引布局

设计文档给出 8 类布局（`K`=有效 topk，`P`=Pool 大小，`R`=QSA 压缩比，`G`=全局 KV group 数）：

| 模型 / 后端 | 持久化 source 布局 | 解码后的选择 |
|---|---|---|
| DeepSeek V3.2 / GLM-5.2（NVIDIA/AMD） | `K` int32 | 逻辑 token ID，保留原始顺序，`-1` padding |
| DeepSeek V4 / V4.1（NVIDIA/AMD） | 每个实际 source `K` int32 | 压缩 cache 坐标；source ratio 与消费层映射显式给出 |
| GLM-5.3 KPool（NVIDIA/AMD） | `K/P` packed Pool/validity 字 + `P-1` literal 尾项 | 精确展开历史 + 因果尾部（含短上下文 padding） |
| GLM-5.3 未启用 Pool 压缩 | 原始 int32 | 逻辑 token ID |
| Qwen4Exp NVIDIA（QSA） | `K/R` Block ID、query 位置、可见 Block 数 | 展开 token ID、因果尾、padding，末列为 valid-count |
| Qwen4Exp AMD | 原始 `K+R-1` token ID | 后端实际选择，无末尾计数列 |
| Step5 NVIDIA | `G*(K+2)` int32 | 分 KV group 的历史 Region + 当前 Region；末列为选择宽度元数据（`row[:row[-1]]` 还原变宽选择） |
| HY-V4 NVIDIA / Dots3 / GLM4-MoE-Lite sparse MLA | 实际生产层的原始 token ID | 显式 source 映射；普通层与 SWA 层不作为 source |

- MiniMax-M3 的 per-head Block/indexer 实现已审计，但**本版无 adapter**。
- 未注册 source 的模型/后端在 worker 初始化时直接失败（fail-fast），不静默跳过。
- 模型侧接入点：`deepseek_v2.py`、`deepseek_v32/{attention.py, amd/rocm.py}`、`deepseek_v4/attention.py`、`deepseek_v41/{attention.py, nvidia/model_state.py}`、`glm5next/common/attention.py`、`hy_v4/nvidia/attention.py`、`qwen4_exp/{nvidia,amd}/indexer_qsa.py`、`step5/nvidia/{model, sparse_attention, flashinfer_sparse}.py`、`dots3_note/nvidia/model.py`、通用 `mla.py`。

## 4. 支持的场景与返回通道

| 场景 | 支持情况 |
|---|---|
| 离线 LLM API | `outputs[0].outputs[0].indexer_topk`（int32 NumPy，首维=逻辑 token 位置，第二维=source，按 `profile["sources"]` 排序）+ `indexer_profile` |
| OpenAI Completions / Chat Completions | 每个 choice 返回 `indexer_topk` base64 `.npy`；流式仅在最后一个 chunk 携带；属 vLLM 扩展字段（对照 2026-10-20 OpenAI OpenAPI 2.3.0 不存在） |
| scale-out token-in-token-out | 同样返回 base64 `.npy` |
| packed 输出格式 | 返回 `indexer_payload`（离线 bytes / HTTP base64）+ `indexer_profile`，API server 不展开数据；用 `decode_indexer(rows, profile)` 按 source codec 解码 |
| raw/compact 编码 | `indexer_encoding="raw"` 关闭 compact Pool/QSA codec；所有布局保留选择顺序 |
| 返回范围 | `[indexer_topk_prompt_start, prompt_len + accepted_output_len - 1)`；最后采样 token 无 forward 选择 |

## 5. 运行前提与限制

- 必需：`VLLM_USE_V2_MODEL_RUNNER=1`（Model Runner V2）、generate runner、prefix caching。
- 沿用 AuxOutput 既有限制：PP、DCP/PCP、adaptive verification、remote PD connector 不支持。
- Rust raw engine client 保留新 IPC 字段；Rust HTTP 投递未实现。
- 跨主机 SHM、Mooncake、新 PD 协议不在本版范围。
- SHM 存储：共享 immutable first-writer 行为，hash 重用可能保留该逻辑位置的早期有效选择；驱逐/缺失对象显式报错而非补零。每 output worker 最小持久容量 `KV_blocks × block_size × (R3 + Indexer bytes/token)`；启动按 RAM 与容器 cgroup 较小值预算，容量不足时给出 `--num-gpu-blocks-override` 等修正建议。
- 压缩收益示例（格式尺寸，非性能实测）：`K=2048` 时 GLM KPool `P=4` 存储 2,060 B/source/token（vs 原始 8,204）；QSA NVIDIA `R=4` 存储 2,056（vs 8,208）。

## 6. 测试与验证状态

- CPU 级测试 `tests/model_executor/test_indexer_capture.py` 覆盖：raw 布局保序/padding/source 映射、Pool 编码短因果行与乱序 Pool、QSA 因果尾与末列计数、Step5 prefill/decode 布局一致性、profile 身份（codec 语义 + 消费层）、共享 buffer 复用时快照存活。
- PR 自述：本版未跑模型评测与硬件端到端；CPU-only 测试不代表支持 CPU 推理。

## 7. 对 Ascend 适配的含义

1. 上游已给出"逐模型 source 注册 + codec/profile"的框架；Ascend 若跟进，需在 `dsa_v1`/`sfa_v1` 两条 top-k 路径上注册 NPU source，并声明自己的布局（普通 int32、压缩 cache 坐标、KPool 类编码、无效值口径）与消费层映射。
2. V4/V4.1 的压缩 KV 坐标语义与 Ascend V4 压缩 cache 路径（`_maybe_update_compressed_caches_and_select_topk`）直接相关，可对照其 source ratio/consumer mapping 的表达方式。
3. 协议字段与 profile 结构可作为 Ascend 内部原型的对照物，但仍需等 #49555/#62915/#47280 收敛后再固化对外接口。
