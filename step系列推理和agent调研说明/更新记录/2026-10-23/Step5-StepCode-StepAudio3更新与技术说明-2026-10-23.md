# Step 5 / Step-Code / StepAudio 3 与昇腾适配：2026-10-05 → 10-23 阶段更新

按四块组织：**社区更新点 → 昇腾需要关注的点 → 当前任务项进展 → 人力投入矩阵与日常活动**。前两块按 **RL 场景推理侧** 与 **Agent** 两条主线展开。

> **区间**：2026-10-05 00:00 → 2026-10-23 14:30（北京时间）。本文件为该区间统一版，取代此前"10-20 正文 + 10-23 增补"的两段式写法；`更新记录/2026-10-20/` 保留为当日归档，不再滚动更新。
> **固定版本**：Step-Code `81e79bc4b6`（10-23 14:28:19 +08）；`stepfun-ai/vllm` 的 `step-5-preview-tracking` = `e177c621f2`（10-10 16:48:44，区间内**未再变化**）；`stepfun-ai/Step-5-Preview` 文档仓 `acf39aef5d`（10-15）；HF `stepfun-ai/Step-5-Preview` 权重 `lastModified` = 2026-10-15T01:28:10Z。
> **同期生态基线**（来自本仓其他跟踪目录，10-23 核对）：vLLM `19c0e170d2`（部分专项 `432b1003f8ea`）、vLLM-Ascend `db8836752`、SGLang `f4a2df3dac`。
> **本项目**：工作区 `C:\Users\w00899129\step5`。文中"本项目路径"均相对该工作区，除标注「未找到」外已在本地检出中逐条实测存在；这些私有检出**不随本仓库分发**。
> **方法边界**：公开 API/仓库/文档核查 + 本项目本地源码实测。**未**做真机推理、性能复测、昇腾算子数值验证；未修改任何业务代码。PR 号可直接拼 `https://github.com/<org>/<repo>/pull/<号>` 访问。
> **主线定位**：全文按两条主线展开——**RL 场景的推理侧**（以 Step 5 的引擎改动与 RL 专属关注点为载体，见 1.2 与 2.6）与 **Agent 侧**（Step-Code 会话、子 Agent 编排、上下文/前缀治理）。Step 5 Preview 官方技术文档与 StepAudio 3 只作口径与边界记录（见 1.4），不构成主线结论；batch 一致性 / DSA 索引回放 / LoRA 等生态线不在本文记录，由本仓各跟踪目录（`batch-consistency-weekly/`、`dsa-index-replay/`、`lora-progress-tracking/`、`rl-replay-consistency/`）维护。

---

# 一、社区更新点

## 1.1 状态复核（10-23 实测）

| 对象 | 10-20 时结论 | 10-23 实测 | 增量 |
|---|---|---|---|
| Step-Code `main` | `be88ee77`（10-20 19:49） | `81e79bc4b6`（10-23 14:28:19） | **+16 提交**（13 非 merge） |
| `stepfun-ai/vllm` `step-5-preview-tracking` | `e177c621`，10-10 H200/SM90（PR #30） | `e177c621f2`（**未变**） | 0 |
| `stepfun-ai/Step-5-Preview` 文档仓 | 10-13～10-15 更新许可、vLLM/SGLang 部署、技术博客入口 | `acf39aef5d`（**未变**） | 0 |
| HF `stepfun-ai/Step-5-Preview` 权重 | FP8 可访问，`lastModified` 10-15 09:28:10 | 未变；downloads 2467；tags 含 `fp8`、`step5_vl` | 0 |
| Step-Cookbook | 10-09/10-10 英文 README、10-12 合并 | 未变 | 0 |
| StepAudio 3（Realtime 论文） | 10-12 提交 v3 | 未变；仍无独立三代推理仓；`feat/step-audio-support` 分支 HEAD 停在 2026-03-10 | 0 |

> 口径提示：`stepfun-ai/vllm` 的 `main` HEAD 是 `b372ad3e90`（2026-05-28），比 `step-5-preview-tracking` 的 10-10 **更旧**——判断 Step5 分支新旧必须按分支看。

## 1.2 RL 推理侧：Step 5 在 `step-5-preview-tracking` 上的推理引擎改动

这个分支本身就是 **Step5 专属实现**（代码全部在 `vllm/models/step5/**`，不含通用功能），区间内新增提交仅 `e177c621` / PR #30（10-10 16:48:44）。先给 Step5-Preview 的结构要点，再看这条提交改了它的哪一环：

- **结构**：稀疏 MoE，95 层 = 92 主干 + 3 层 MTP；352 专家；FP8 block-wise 量化；上下文 1M。
- **注意力**：SWA 层（局部滑窗）+ 稀疏全局层（DSA），DSA 带 summary cache sidecar，按 region 打分选 top-k 后读 KV。
- **rollout 相关能力**：MTP ≤ 3（多 token 验证）、CUDA graph 可捕获、reasoning/tool parser。

以下 5 条按行号在 `upstream-step5-reference`（= `e177c621` 同版本）逐行可对上，并标出各自落在 Step5 rollout 推理的哪一环：

| # | 改动（行号可对上） | RL 推理环节 |
|---|---|---|
| 1 | 后端准入扩大：`nvidia/sparse_attention.py:933` 的 `supports_compute_capability` 从 SM103 扩为 SM90、SM103；`model.py:994` 报错文案同步 | rollout 可用性前提（决定能否在该卡上跑） |
| 2 | 按设备选择打分内核：`nvidia/flashinfer_sparse.py:46-49`，SM90 走新增 `paged_mqa_logits_sm90`，其他路径继续用 FlashInfer `fp8_paged_mqa_logits` | long-context rollout 的 prefill/decode 选核 |
| 3 | 新增 Hopper 摘要页打分：`nvidia/ops/sparse_indexer_scores.py:274`；FP8 query + 打包 summary KV + 长度/页表，**超有效长度区域置 −inf** | rollout 注意力可见性（无效区不可被选中） |
| 4 | top-k 确定性与布局修复：`nvidia/ops/sparse_indexer.py` 的 `select_regions`；仅 DeepSelect 保证未排序选择的确定性顺序，其他后端强制排序；要求精确 FP32 排序 | **replay 关键**：同输入同选择的确定性前提 |
| 5 | 硬件覆盖与参考一致性检查：`tests/kernels/core/test_step5_sparse_decode_attention.py` 覆盖 top-k=2/7/512、打包 scales、query pairs、page boundary | 一致性回归基线（仅核查测试代码，未在 H200 上运行） |

接口流程：

```text
/v1/chat/completions
  → Step5ForCausalLM / Step5Model → Step5DecoderLayer
      ├─ SWA 层：局部滑窗注意力
      └─ 稀疏全局层：QKV 与 indexer 投影
           → 更新 summary cache → 对历史区域打分
              SM90 → paged_mqa_logits_sm90；其他受支持路径 → FlashInfer 打分
           → select_regions：历史 top-k + 当前区域
           → 读取所选 KV，执行 prefill/decode attention 与 gate
  → MoE / 残差 → logits → step5 reasoning/tool parser → 标准响应
  → Agent 宿主执行工具并回传
```

## 1.3 Agent 侧：Step-Code 提交与逻辑

Step-Code 是本区间**唯一持续更新**的对象。下表按合并时间（北京时间）列出 10-05 → 10-23 的**全部合入 PR**，不按 10-20 分段。

| 时间（+08） | PR | 标题（原文） | 与本项目相关性 |
|---|---|---|---|
| 10-08 10:35 | #211 | fix: add StepCode app attribution headers | — |
| 10-08 15:57 | #212 | fix(skills): discover ~/.claude/skills, .claude/skills and ~/.stepcode/skills | — |
| 10-08 16:24 | #195 | fix(coding-agent): recover command failures and preserve session state | — |
| 10-08 16:25 | #203 | fix(coding-agent): make Ultracode mode discoverable | — |
| 10-08 16:29 | #222 | feat: add remote Step Plan login | — |
| 10-08 16:36 | #217 | feat: add then_run verification to edit_file and write_file | — |
| 10-09 14:24 | #223 | fix: 修复 StepPage 自动准备、插件误报和终端 PATH 持久化 | — |
| 10-09 15:10 | #226 | fix: scope session listing and continue to the header cwd | — |
| 10-09 16:14 | #225 | fix(tui): refresh working status when tools finish | — |
| 10-09 20:30 | #227 | fix(subagent): return the chain's final result and mark unstarted steps | — |
| 10-09 20:46 | #229 | fix(subagent): stop background lanes from flickering and flooding the transcript | — |
| 10-10 10:43 | #230 | fix(subagent): wake an idle parent when a background lane settles | — |
| 10-10 14:35 | #233 | fix(subagent): deliver background lane results promptly and in full | — |
| 10-10 16:30 | #234 | feat(cli): keep /effort as the only thinking-level command and save on Enter | — |
| 10-11 12:02 | #235 | feat: add Full Access and simplify permission tiers | — |
| 10-11 12:04 | #232 | feat(compaction): use model-managed CLM by default | **强相关** |
| 10-12 12:05 | #241 | fix: stop repeating Ultracode capability reminders | — |
| 10-12 12:11 | #244 | fix(skills): readable, actionable SKILL.md frontmatter diagnostics | — |
| 10-12 12:28 | #243 | fix(shell): discover native Git Bash on Windows instead of WSL | — |
| 10-12 15:04 | #245 | feat: persist Ultracode mode per session | — |
| 10-12 15:29 | #246 | fix(skills): quiet startup skill conflicts for duplicated shared skill dirs | — |
| 10-14 18:02 | #249 | fix: harden workflow lifecycle, concurrency, and retries with structured errors | — |
| 10-14 18:06 | #248 | fix: route then_run through run_command | — |
| 10-15 11:22 | #250 | docs: add repository structure and contributor guidelines | — |
| 10-15 14:04 | #252 | fix: always use system messages for StepFun chat models | — |
| 10-15 17:21 | #254 | chore: define fixed internal pull request reviews | — |
| 10-16 11:35 | #251 | feat(usage): add Overview, Account, and Status panels | — |
| 10-16 11:39 | #242 | feat: add tool availability and standalone task review | — |
| 10-16 11:42 | #255 | fix: preserve context budgets and summary tool identity | **强相关** |
| 10-19 16:38 | #262 | remove | — |
| 10-19 19:14 | #261 | feat(step): store and refresh an account session beside the API key | — |
| 10-19 19:38 | #266 | feat(coding-agent): add SQLite session search index | — |
| 10-19 20:09 | #265 | docs: remove residual upstream branding across all Markdown docs | — |
| 10-19 21:07 | #267 | fix(context): retain live-context notices across append-only turns | **强相关** |
| 10-20 14:49 | #259 | feat(cli): show recent models in the model selector | — |
| 10-20 14:49 | #260 | feat(coding-agent): connect terminal sessions to Lark with mirrored approvals | — |
| 10-20 15:36 | #271 | fix(step): resolve update archive root for flat Windows zips | — |
| 10-20 16:47 | #268 | feat(cli): add /btw for one-shot side questions | — |
| 10-20 17:13 | #273 | docs(contributing): add project contribution guide | — |
| 10-20 19:49 | #274 | feat(tui): show terminal tab status and simplify tool errors | — |
| 10-21 11:35 | #275 | fix(cli): abbreviate run_command titles in Step tool rows | — |
| 10-21 14:01 | #270 | feat(web): improve conversation, inspectors, performance metrics and account settings (part 1) | — |
| 10-21 17:35 | #282 | feat(subagent): let the model choose foreground or background lanes | — |
| 10-21 17:35 | #283 | fix(workflow): reject phase() misuse instead of silently running no agents | — |
| 10-21 17:40 | #284 | fix(compaction): retry incomplete streams and report effective failure budgets | 方法同向 |
| 10-21 17:41 | #279 | fix(step): stop dropping bracket-prefixed paths in narrowed find_files searches | — |
| 10-21 17:42 | #277 | fix(step): include installer output in update failure messages | — |
| 10-21 17:42 | #278 | fix(step): resolve search rows structurally instead of re-parsing rendered text | — |
| 10-22 17:05 | #293 | fix(coding-agent): preserve task-state prompt cache prefixes | **强相关** |
| 10-23 13:32 | #289 | feat(cli): add -C/--cwd flag to set the working directory explicitly | — |
| 10-23 13:33 | #285 | fix(cli): save model selection as default on Enter | — |
| 10-23 13:42 | #296 | fix(coding-agent): keep the tool block stable across plan mode | 方法同向 |
| 10-23 14:23 | #298 | fix(coding-agent): settle extension command turns before returning | — |
| 10-23 14:28 | #297 | feat(coding-agent): make a feedback report investigable | — |

表注：共 54 个 PR，对应区间内 107 条提交（30 merge / 77 非 merge）；相关性仅标注前缀/上下文与压缩两类，其余为 CLI/TUI/会话/仓库治理，均与 Step5 推理无关；PR 号拼 `https://github.com/stepfun-ai/Step-Code/pull/<号>` 访问；10-05→10-20 窗口的逐提交 SHA 清单见 `../2026-10-20/Step-Code-commits-20261005-20261020.tsv`。
**请求流程**：

```text
终端输入 / 会话恢复 → Step 登录与凭证 → fetchStepModels（GET /models）
  → 按 model_type 过滤，解析上下文/视觉/reasoning_effort 元数据 → 用户选择模型与 /effort
  → 工作上下文投影 / 必要时 CLM 维护 → OpenAI 兼容 Chat Completions
  → 返回文本或 tool_calls → 宿主执行工具 / 子 Agent → 结果回传 → 下一轮
```

代码证据（固定 `be88ee77071188975475bcb5ef4e1edd0913d8ad`）：`packages/coding-agent/src/step/defaults.ts`（`STEP_DEFAULT_MODEL = "step-5-preview"`）、`packages/providers/src/step-provider/index.ts`（`fetchStepModels`、`normalizeStepModel`：固定 endpoint / OpenAI completions 方言 / `supportsDeveloperRole: false`）、`packages/coding-agent/src/features/subagent/execute.ts`（链式结果突出最终输出）。

**CLM 与推理成本**：CLM 管理下一次请求的工作上下文视图，原始历史保留；默认维护最多两次请求、90 秒、8,192 输出 tokens；接受自动编辑需至少节省 1,024 tokens 且达 5%。因此收益要同时看维护调用开销、后续输入下降与**前缀缓存受损**，不能推断"默认 CLM 必然更便宜"。10-19 notices 修复与 10-22 #293 都在做"只追加即保持前缀"。

**飞书边界**：`/connect lark` 把消息与审批映射到当前终端会话，第一个合法审批生效；**不启动第二套 Agent 循环**，不是后台常驻服务，也不新增语音处理能力。

## 1.4 非主线：文档口径与边界

- **Step 5 Preview 技术文档**（10-15 *Inside Step 5 Preview*）：92 层窄深设计、SWA + Sparse GQA 降长上下文成本、**训推数值对齐与异步轨迹重放降 RL 偏差**、MTP/FP8/EP/KV 管理提升 rollout 效率、压缩进训练。这是本文两条主线的背景说明，不单独构成结论。**口径差异必须保留**：官方 README 写最大输出可至 1M（输入输出共享总上下文预算），API 模型页仍列 64k；线上按 endpoint 实际限制。
- **StepAudio 3（不在本文主线内）**：区间内只有 Realtime 论文 v3 修订（10-12），仍无独立三代推理仓；旧仓名与 TTS-3B 的"3B"不能作为三代已开源证据。Step-Code provider 过滤语音类型，**语音驱动编程属可选的自行桥接方案，不是仓库已实现能力**；本文不展开。

# 二、昇腾需要关注的点

## 2.1 结论级判断（5 条）

1. **SM90 结论不可外推**：`e177c621` 只让 Hopper/H200 用上该分支原生稀疏路径；本项目目标硬件 A2/910B（另有 950DT 验证），不能据此认为昇腾已支持。
2. **行号口径**：`e177c621` 来自公开分支 09-22～10-10 的 `cute_dsl → flashinfer` 重组之后；本项目 CUDA 检出 `water18@5d7f9124`（09-20）停在重组**之前**且不含该提交。**做 NPU 对标要读 `upstream-step5-reference`，不能拿 `water18` 对行号**（详见 3.1 第 3 项）。
3. **cann-ops 硬依赖且已 churn 一次**：`IndexerRegionScore` / `GenericBlockSparseAttention` 来自 CANN ≥9.2.0-beta.2 外部包，#65 已因 `(score, ids)` 返回变更适配过一次 → 必须 pin 版本 + 签名前检。
4. **DSA 摘要 sidecar 与 KV 迁移互斥**：summary 缓存随物理页复用，要求 `prefix_match_unit` 为 8 的倍数、命中重算 1 block，且与 KV-connector/offload **不兼容（显式 raise）**；做 PD 分离/前缀缓存方案时这是硬约束。
5. **MC2 dispatch 有运行期超时风险**：10-20 现场出现 `MoeDistributeDispatchV3` kernel 超时（AIVEC timeout → `Stream Synchronize failed, retCode=0x30`）。上游材料只把 MoE 当结构，未覆盖该风险，需自建看护（3.2 第 10 项）。

## 2.2 可复用 / 需自建

**可直接落地（可用）**

- **top-k 确定性 / 平票规则**（上游 1.2 第 4 条）：本项目 `vllm/vllm-ascend-step4/vllm/envs.py:292` 声明 `VLLM_STEP5_DSA_FORCE_STABLE_TOPK: bool = True`（**默认已 True**），读取逻辑在同文件 `:1974-1975`（缺省按 `"true"` 解析），实际使用点 `vllm/models/step5/sparse_attention.py:1234 / 1703 / 1705 / 3171 / 4071` 与 `vllm/models/step5/kernels.py:219`；默认值断言 `tests/test_envs.py:66-75`（删环境变量后必须为 `True`）。NPU 侧确定性来源：`vllm_ascend/models/step5/indexer_region_score.py:82/88`（`mask_mode=3` 因果 + `sort_mode=1` 升序）与 `generic_block_sparse_attention.py:58` 的 `_append_current_region_kernel`（Triton 版在 `models/step5/dsa_kernels.py:2267`，升序追加、不挤掉选中项）。→ 动作见 2.4 P0-1。
- **测试矩阵平移**：上游 top-k=2/7/512、page boundary、打包 scales 三张表正是 `indexer_region_score` + GBSA 最易出契约错的维度；本项目落点 `vllm/vllm-ascend-step4/tests/kernels/core/test_step5_*.py`、`tests/v1/attention/test_step5_{mtp_state_machine,sparse_summary_cache}.py`、`vllm-ascend/.../tests/ut/models/test_step5_model.py`、`tests/e2e/nightly/.../test_step5_decode_sparse_meta.py`。
- **上下文治理方法论**：以"只追加保前缀"（#293 + 10-19 notices）为验收模板，在本项目记 `prefix_match_unit`、命中 token、重算 block、sidecar 是否重建（`kv_pool/ascend_store/pool_scheduler.py:121`、`pool_worker.py:172`；同一文件在上游干净镜像 `_ml/` 里是 `:146`/`:262`——**行号随检出不同，引用时必须带检出名**）。对齐校验在 `vllm/models/step5/sparse_attention.py:1242-1251`（`prefix_match_unit` 必须是 sparse region block 的整数倍，否则直接报错）。

**部分可用（语义可迁移，实现不可移植）**

- **Hopper 摘要打分核**：`paged_mqa_logits_sm90` 不能移植（FlashInfer/FP8 打包 + warp 级打分），但"超界置 −inf/不可见"语义在本项目已等价——`indexer_region_score.py` 用 `actual_kv_lengths`(seq_len) 驱动因果上界（算子自推每行可见 region 上界 `floor((ctx+j)/8)`，"当前 region 永不占槽"）。→ 只移植测试构造。
- **按设备选择打分实现**：NPU 侧单一路径（aclnn `IndexerRegionScore`），无设备二分；把"可替换"留在适配层边界（`models/step5/dsa_attention.py` 之上），不要在 `summary_cache.py` 内写死调用点。
- **后端准入写法**：本项目无 SM 门禁，等价门禁在 `models/step5/dsa_attention.py` 与 `vllm_ascend/ascend_config.py`（`step5_prefill_mnd_slot_allocator` 默认 `default`，`:253`/`:389`）；照抄"失败给精确原因 + 可用集合"的写法。

**不可用**

| 上游内容 | 原因 |
|---|---|
| H200/SM90 硬件能力本身 | 目标硬件是 A2/910B；SM90 结论不可外推 |
| FlashInfer / CUTLASS CuTeDSL 依赖清单 | NPU 侧是 aclnn 单算子 + Triton-ascend，依赖树不同 |
| StepAudio 3 全套（Realtime v3 / Gen / Music / TTS） | 本项目无音频理解/生成路径；Step-Code provider 过滤语音 |
| Step-Code 飞书连接、Step Plan 登录、TUI 面板、`/btw` | 产品/交互层，与 Step5 推理无关 |
| "CLM 默认更便宜" | 已否证：维护调用开销 + 前缀缓存受损 |

## 2.3 已完成：上游"不能只改 registry"的迁移矩阵

| 上游环节（`e177c621`） | 本项目 NPU 实现位 |
|---|---|
| 摘要缓存更新 | `models/step5/summary_kernels.py`（Triton-ascend 重实现）+ `summary_cache.py`（fp8 分页 sidecar） |
| 打分 + top-k + 排序 | `models/step5/indexer_region_score.py`（aclnn `IndexerRegionScore`） |
| 稀疏 KV 访问 | `models/step5/generic_block_sparse_attention.py`（aclnn `GenericBlockSparseAttention`，`PA_BBND` 直通、`is_consistent_topk=False`） |
| SWA 层 | `AscendFlashAttentionBackend` → `flash_attn_npu_3.flash_attn_with_kvcache(window_size=(512,0))` |
| MoE 路由 / FP8 专家 | `device/device_op.py`（`npu_moe_gating_top_k`）、`quantization/methods/w8a8/fp8_block.py`、`ops/grouped_matmul_v5.py`、`ops/fused_moe/moe_mlp.py` |

## 2.4 昇腾侧待办（按优先级，含路径与验收）

**P0-1 跨平台 top-k / ids 对齐用例**
落点 `vllm-ascend/vllm-ascend-step4/tests/ut/models/test_step5_model.py`（新增参数化 case）。固定一组构造 scores（平票、`top_k=2/7/512`、page boundary、超有效长度），分别喂 aclnn `IndexerRegionScore` 与 CuTeDSL 选择器。验收：`ids` 逐元素一致、超界列不可见、`-1` 尾填充一致。

**P0-2 接入 verl #8025 等价的训推差异度量**
NPU rollout/actor raw logprob mismatch（count 求和、diff_max、one-ULP 掩蔽），Qwen3 dense 先出首轮统计。验收：输出可比的 mismatch 指标并可回归。

**P0-3 GEMM / 算子分派自查**
实现选择与 tile 分桶是否依赖 M（token 数）；补算子级跨 batch 不变性测试（对照 vLLM #62192 思路）。

**P0-4 订正文档与代码不一致**
落点 `vllm-ascend/.../vllm_ascend/patch/__init__.py:189`：`patch/platform/patch_step5_vl_config.py` **在本地检出中不存在**；`step5_vl` 已由 vLLM 侧原生注册（`vllm/transformers_utils/config.py:126`、`configs/step5.py:214`、`model_executor/models/registry.py:595`）。验收：注释清单与实际文件一一对应 + CI 存在性断言。

**P1-1 MoE dispatch 超时看护**
落点 `ops/fused_moe/token_dispatcher.py:232`（`npu_moe_distribute_dispatch_v2`）、`ops/fused_moe/moe_comm_method.py`。先核 `torch_npu.__version__` 与 v2→V3 算子映射；大 EP（352 专家 + MC2）压测加超时判据与 plog 采集。验收：给出触发条件组合，或明确排除版本映射嫌疑。

**P1-2 V4/V4.1 图回放回归矩阵**
trace replay 固定 decode 历史 → eager vs graph、BS1/BSN、同图档/跨图档、有无 padding，逐 token logits/logprob bitwise 比较（覆盖 #18936/#19034/#19192/#19321/#19177 修复点）。

**P1-3 前缀缓存与上下文治理可解释性验证**
落点 `pool_scheduler.py:121`、`models/step5/summary_cache.py`；对照"只追加"与"改写上下文"两种请求模式。验收：命中率与重算成本的量化差异。

**P2-1 1M 上下文压测**
落点 `vllm/.../transformers_utils/configs/step5.py`（`original_max_position_embeddings=1048576`）与工作区脚本 `run_step5_parallel.sh` / `test_step5_parallel.py`；128K→512K→1M 递增，看 CSA 槽位容量（`worker/step5_csa_slot_manager.py`）、sidecar 显存、prefix 重算；注明 1M 输出 vs API 64k 口径。

**P2-2 cann-ops 版本锁定**
`requirements*` 与镜像 Dockerfile 中 pin `IndexerRegionScore` / `GenericBlockSparseAttention` 接口版本 + 签名前检。

**P2-3 LoRA 与 Step5 组合评估（新增）**
vLLM 侧 Step5 LoRA 已合入 #62498，而本项目 `vllm_ascend/lora` 无任何 MLA 处理、replay 遇 LoRA 关闭、PCP LoRA 仍 `NotImplementedError`。验收：给出"Step5 + LoRA 在 A2 是否可行"的明确结论（可行/需立项/不支持）。

---

## 2.5 本项目自有 fork 的增量（对标基线，非任务进展）

本节只陈述已实测的代码事实，用于判断上游对标口径；任务状态与人力见第三、四块（本期暂空）。

| 本地路径 | 分支 / HEAD | 日期 | 内容 |
|---|---|---|---|
| `vllm/vllm-ascend-step4` | `water18` / `5d7f9124db` | 2026-09-20 | vLLM 侧：`vllm/models/step5/**`（CuTeDSL SM90 核）等 |
| `vllm-ascend/vllm-ascend-step4` | `main` / `80d265f13e` | 2026-10-14 | 插件侧：`vllm_ascend/models/step5/**`、`worker/step5_csa_slot_manager.py` |
| `_ml` | `main` / `775f7d1` | 2026-10-15 | 上游 `vllm-project/vllm-ascend` 干净镜像 |
| `upstream-step5-reference` | `e177c62` | 2026-10-10 | 上游 `stepfun-ai/vllm` partial clone，只含 `vllm/models/step5/` |

- 前两个 checkout 的 `origin` **是同一个私有 fork** `github.com/earthmanylf/vllm-ascend-step4`，只是检出分支不同（不是两个独立仓库）。
- vllm-ascend 区间内主要工程动作是 **#50（aclnn 替换）→ #55（接口拆分）→ #65（接口 churn 修复）** 这条链。
- 本地对象库**没有** `e177c621`（`git cat-file` 失败）；公开分支在 09-21～10-10 有 26 个 Step5 提交（`Sync with upstream (#10)` 另带入上游主线提交，`compare` 的 `total_commits` 为 2220，不能当 Step5 提交数引用）。两棵树里 `water18` 用 `cute_dsl`，公开分支用 `flashinfer`，两棵树不可混用行号。

---

## 2.6 RL 场景专属关注点（与通用推理适配不同）

| 关注点 | 为什么 RL 场景更敏感 | 昇腾侧落点 / 动作 |
|---|---|---|
| **确定性 / 可重放** | rollout 与训练要按同一轨迹对齐，采样与 top-k 的非确定性会被误判成模型问题 | `VLLM_STEP5_DSA_FORCE_STABLE_TOPK`（默认 True）、aclnn `IndexerRegionScore` 的 `sort_mode=1`、GBSA `_append_current_region_kernel`；动作见 2.4 P0-1 |
| **logprob / 数值一致性** | PPO/GRPO 的 ratio、KL 与 teacher scoring 依赖 raw/processed 语义一致 | 接入 verl #8025 等价的 mismatch 指标（count 求和、diff_max、one-ULP 掩蔽）；动作见 2.4 P0-2 |
| **batch 不变性** | 同请求因 batch 大小、同批伙伴与顺序变化产生漂移，RL 侧的奖励就会被噪声污染 | GEMM/算子分派自查（tile 分桶是否依赖 M）+ 图回放回归矩阵；动作见 2.4 P0-3 / P1-2 |
| **replay 的数据闭环** | 拿到数组只是第一步，坐标、层映射、token 位移、权重版本任一处错位即失效 | I（索引）与 H（隐状态）回传仍未合入，按"未合入"单独推进；本仓 `dsa-index-replay/`、`rl-replay-consistency/` 维护口径 |
| **权重更新路径** | 训练侧更新权重后推理侧要热更新且不能回滚到旧分片 | Sparse HCCL（PR #18915）已验证的分片更新路径；Step5 LoRA 组合待评估（2.4 P2-3） |

---

# 三、当前任务项进展

> **待补（本期暂空）**。任务项进展不在本版填写；各工作流当前状态与证据入口已在仓库对应目录中维护：`batch-consistency-weekly/`、`dsa-index-replay/`、`lora-progress-tracking/`、`rl-replay-consistency/`、`vllm-ascend-pr18915-sparse-hccl/`、`qwen3-30b-megatron-hf-sparse-sync/`、`replay-flowcharts/`。待确认后按「状态 / 本期产出 / 证据入口 / 下一步」四列补齐。

---

# 四、人力投入矩阵与日常活动

> **待补（本期暂空）**。人员、归属与工时台账未纳入本次材料，人力投入矩阵与日常活动节奏待补充后填写；现阶段不得引用任何未确认的人数与 FTE 数字。

---

# 五、口径订正（相对 10-20 归档版）

1. **`patch/platform/patch_step5_vl_config.py` 不存在**：只出现在 `patch/__init__.py:189` 的清单注释里；`step5_vl` 现由 vLLM 侧原生注册。
2. **"两个仓库"改口径**：`vllm/vllm-ascend-step4`（water18）与 `vllm-ascend/vllm-ascend-step4`（main）同属 fork `earthmanylf/vllm-ascend-step4`，是同一仓库的两个分支检出。
3. **`water18` 不含 `e177c621`**：行号只对 `upstream-step5-reference` 有效。
4. **提交清单时区混用**：`Step-Code-commits-20261005-20261020.tsv` 共 91 条（27 merge / 64 非 merge；10-20 正文写 63，实际 64），其中 **5 条为 `Z`（UTC）**、86 条为 `+08:00`；按字符串排序会把那 5 条排到最前（如 `2026-10-09T21:18:37Z` 实为北京时间 10-10 05:18）。
5. **MoE 缺运行期风险**：10-20 归档只把 MoE 当结构，未覆盖 MC2 dispatch 超时；本文补入 2.1 第 5 条与 P1-1。
6. **`step5_prefill_mnd_slot_allocator` 定位精确化**：默认 `default`（`ascend_config.py:253/389`），`runner` 为 opt-in（#44）。

---

# 六、证据与复现

**上游（10-23 实测）**

```text
Step-Code 增量：
  GET https://api.github.com/repos/stepfun-ai/Step-Code/commits?since=2026-10-20T11:49:54Z
  → 16 条新增（13 非 merge），HEAD = 81e79bc4b6（10-23 14:28:19 +08）
step-5-preview-tracking：
  GET https://api.github.com/repos/stepfun-ai/vllm/commits?sha=step-5-preview-tracking
  → HEAD = e177c621f2（10-10 16:48:44），区间内未变化
e177c621 目录结构（确认无 cute_dsl）：
  GET …/contents/vllm/models/step5/nvidia/ops?ref=e177c621f2b8ae1be06190bc34eef84646e54f11
上游测试用例（本地检出无 tests 目录）：
  https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/tests/kernels/core/test_step5_sparse_decode_attention.py
```

**本项目（本机实测）**

```text
git -C vllm/vllm-ascend-step4          log -1   → 5d7f9124db 2026-09-20 (water18)
git -C vllm-ascend/vllm-ascend-step4   log -1   → 80d265f13e 2026-10-14 (main)
git -C upstream-step5-reference        log -1   → e177c62   2026-10-10
git -C vllm/vllm-ascend-step4 cat-file -t e177c621f2b8ae1be06190bc34eef84646e54f11  → 不存在
git -C vllm/vllm-ascend-step4 remote -v         → earthmanylf/vllm-ascend-step4.git
```

**文件证据清单**

- 路径映射（机器可读）：`本项目路径映射-20261023.tsv`（同目录）
- Step-Code 提交清单：`../2026-10-20/Step-Code-commits-20261005-20261020.tsv`
- 关键提交 patch：`../2026-10-20/vllm-step5-e177c621-commit.json`
- 故障原文：`附件-MoeDistributeDispatchV3超时摘录-20261020.txt`（同目录；工作区原路径 `diagnostics/20261020_191341/error_context.txt`，未入库）
- 本项目结构基线：`Step5_模型结构分析报告.md`（**仅工作区，未入库**；§11 CUDA 现状 / §12 vllm-ascend 现状 / §12.4 三条算子路线 / §13 打通方案）
- 本文其余"本项目路径"均指工作区 `C:\Users\w00899129\step5` 下私有检出（`vllm/`、`vllm-ascend/`、`upstream-step5-reference/`），**不进本仓库**

**边界与未验证项**

1. 未做真机复测：所有"可用/部分可用"判断来自源码与接口契约，非昇腾/GPU 数值或性能验证。
2. 未验证 v2→V3 算子映射（`token_dispatcher.py:232` 调 `_v2`，故障日志为 V3）。
3. 未做 `e177c621` 与 `water18` 的逐提交 diff，只证明"两棵树不同 + 本地缺该对象"。
4. 未审计上游合流：Step5 能力只在 `step-5-preview-tracking`，未逐 commit 审计是否/何时进 vLLM main。
5. 未做 cann-ops 版本矩阵：`IndexerRegionScore` / `GenericBlockSparseAttention` 在 ≥9.2.0-beta.2 上的签名差异未逐一验证（#65 已 churn 一次）。
6. 第三、四块本期暂空；其余工作流数据引自本仓各跟踪目录 10-23 核对结论，本文未重新核对其实验数据，人力相关字段未获取来源。
