# Step 5 Preview、Step-Code 与 StepAudio 3 更新及技术说明（2026-10-05 → 2026-10-23）

> **区间**：2026-10-05 00:00 → 2026-10-23 14:30，北京时间。这是本系列的统一版，取代此前"10-20 正文 + 10-23 增补"的两段式写法；10-20 目录保留为当日归档，不再滚动更新。
> **固定版本**：Step-Code `81e79bc4b6`（10-23 14:28:19 +08）；`stepfun-ai/vllm` 的 `step-5-preview-tracking` = `e177c621f2`（10-10 16:48:44，区间内**未再变化**）；`stepfun-ai/Step-5-Preview` 文档仓 `acf39aef5d`（10-15）；HF `stepfun-ai/Step-5-Preview` 权重 `lastModified` = 2026-10-15T01:28:10Z。
> **本项目**：工作区 `C:\Users\w00899129\step5`。文中"本项目路径"均相对该工作区，除标注「未找到」外已在本地检出中逐条实测存在；这些私有检出**不随本仓库分发**。
> **方法边界**：公开 API/仓库/文档核查 + 本项目本地源码实测。**未**做真机推理、性能复测、昇腾算子数值验证；未修改任何业务代码。
> **链接口径**：正文给出 PR 号即可直接拼 `https://github.com/stepfun-ai/Step-Code/pull/<号>` 访问；vLLM 提交以 SHA 引用。

---

## 1. 先看结论

1. **推理侧区间内的实质代码增量只有一条：H200/SM90 Sparse Attention 支持**（`e177c621` / PR #30，10-10）。10-10 之后到 10-23 该 tracking 分支**没有新提交**，所以"10-20 → 10-23 推理性有新东西"不成立。MTP、视觉入口、稀疏注意力主体是该分支既有能力，不能算本期新增。
2. **持续更新的是 Step-Code**：10-05→10-20 有 91 条提交（27 merge / 64 非 merge），10-20→10-23 再增 16 条（13 非 merge）。主题始终是上下文管理、后台子 Agent、编排容错与会话产品能力，**对 Step5 推理性能与精度无直接影响**。
3. **10-15 的官方仓、权重与技术博客更新，使"等待官方权重"的旧结论过时**。现在应引用 `stepfun-ai` 官方资源，不能沿用本地旧版 `TypeSafeAI` 镜像 README 的许可、参数限制或部署结论。
4. **StepAudio 3 与 Step-Code 不能画等号**：Step-Code provider 明确过滤语音类型；区间内 StepAudio 侧只有 Realtime 论文 v3 修订（10-12），**没有**官方三代推理仓，也不证明已接入 Step-Code。
5. **上游这些内容对昇腾落地分三类**（详见 §7）：可直接落地的是 top-k 确定性、测试矩阵平移、上下文/前缀稳定方法论；部分可用的是打分核语义、设备分派写法、能力门禁写法；不可用的是 H200 硬件结论、FlashInfer/CuTeDSL 依赖、StepAudio 全套与 Step-Code 产品交互层。
6. **本项目自身增量不在上游，而在自有 fork 与一次故障里**：vllm-ascend 侧 `main@80d265f1`（10-14，含 #50 aclnn 替换、#65 接口适配）；vLLM 侧 `water18@5d7f9124`（09-20，**不含** `e177c621`）；10-20 的 MC2 `MoeDistributeDispatchV3` 超时。**这条结论决定 §4 的行号只能对 `upstream-step5-reference` 生效**（见 §8.3）。

---

## 2. 仓库与文档状态复核（10-23 实测）

表注："未查到"仅指所核查公开仓库、分支与文档，不推断内部研发状态。仓库 push 时间不能代替特定分支的提交时间。

| 对象 | 10-20 时的结论 | 10-23 实测 | 含义 |
|---|---|---|---|
| Step-Code `main` | 更新到 `be88ee77`（10-20 19:49） | `81e79bc4b6`（10-23 14:28:19） | 区间内持续更新，增量 16 条（见 §3.2） |
| `stepfun-ai/vllm` `step-5-preview-tracking` | `e177c621`，10-10 H200 SM90（PR #30） | `e177c621f2`（**未变**） | 推理侧 10-10 后无新提交 |
| `stepfun-ai/Step-5-Preview` 文档仓 | 10-13～10-15 更新许可、vLLM/SGLang 部署、技术博客入口 | `acf39aef5d`（**未变**） | 以文档与资源入口为主；checkout 中未见 README 所述 `inference/` 目录 |
| HF `stepfun-ai/Step-5-Preview` | FP8 官方权重可访问，`lastModified` 10-15 09:28:10 | `2026-10-15T01:28:10Z`（**未变**）；downloads 2467；tags 含 `fp8`、`step5_vl` | 权重未刷新 |
| Step-Cookbook | 10-09/10-10 英文 README、10-12 合并 | 未变 | 9-29 的 API 教程不属于本期新增 |
| StepAudio 3 Realtime 论文 | 10-12 20:05:18 提交 v3 | 未变；仍无独立三代推理仓；`feat/step-audio-support` 分支 HEAD 停在 2026-03-10 | 论文修订 ≠ 模型发布 ≠ 已集成 |

> 口径提示：`stepfun-ai/vllm` 的 `main` HEAD 是 `b372ad3e90`（2026-05-28），比 `step-5-preview-tracking` 的 10-10 **更旧**——判断 Step5 分支新旧必须按分支看。

---

## 3. Step-Code 本期修改及逻辑

日期以进入 `main` 的合并/提交时间为准；部分分支内作者时间早于合并日。"意义"为代码与文档支持的工程解释，不代表已完成业务性能评测。

### 3.1 10-05 → 10-20

| 日期 | 主要变化 | 修改位置、逻辑及意义 |
|---|---|---|
| 10-08 | 远程 Step Plan 登录、`then_run` 动作衔接 | #222、#217：改进远程终端认证与动作执行衔接 |
| 10-09～10-10 | 后台子 Agent 结果、父会话唤醒和通知修复 | #227、#230、#233：链式任务返回最后结果，后台结束唤醒父会话，最终通知携带完整结果 |
| 10-10～10-11 | `/effort` 统一；Full Access；CLM 成为默认压缩方式 | #234、#235、#232：改变推理强度入口、权限层级与长会话上下文维护 |
| 10-12 | Ultracode 会话状态持久化；Windows Git Bash 与 Skills 诊断修复 | #245、#241、#243：避免会话恢复后编排模式丢失 |
| 10-14 | workflow 生命周期、并发、重试与恢复修复；`then_run` 走 `run_command` | #249、#248：取消时先排空子任务，共享并发准入，区分限流与额度耗尽 |
| 10-15～10-16 | StepFun 使用 system role；usage/status；独立任务 review 与工具可用性 | #252、#251、#242：修复协议兼容，补充消耗观测与任务检查 |
| 10-19 | 稳定 live-context notices；SQLite 会话搜索；账户 session 刷新 | #267、#266、#261：保护追加式请求前缀稳定，支持会话检索与认证续期 |
| 10-20 | 飞书连接与镜像审批、最近模型、`/btw`、终端状态与 Windows 更新修复 | #260、#259、#268、#274、#271：扩展同一会话的交互渠道 |

### 3.2 10-20 → 10-23（新增 16 条，13 非 merge）

| 时间（+08） | SHA | 主题 | 与本项目相关性 |
|---|---|---|---|
| 10-21 10:57 | `1ccfa0819d` | `fix(cli)`：命令预览按终端宽度自适应 | 无（UI） |
| 10-21 11:35 | `7ae808a7b9` | Merge #275 | — |
| 10-21 17:32 | `143a9d232a` | `fix(compaction)`：未完整流重试 + 持久化安全失败诊断 | 方法同向：长上下文压缩的失败面与重试 |
| 10-21 17:35 | `118348c05c` | `feat(subagent)`：模型自选前台/后台泳道（#282） | 无（Agent 编排） |
| 10-21 17:35 | `993e714243` | `fix(workflow)`：`phase()` 误用直接报错（#283） | 无 |
| 10-21 17:40/17:41 | `9cc8064` / `8d98d4b` | Merge #284 / #279 | — |
| 10-21 17:42 | `84a16c90ad` | `fix(step)`：搜索结果按结构解析而非重解析渲染文本（#278） | 无（TUI） |
| 10-21 17:42 | `dff413f436` | `fix(step)`：更新失败信息包含安装器输出（#277） | 无 |
| 10-22 16:44 / 17:05 | `50aad0a962` / `3de15c1659` | `fix(coding-agent)`：**保留 task-state prompt cache 前缀**（#293） | **强相关**：追加式请求保持前缀稳定，是 prefix cache 命中率的前置条件（见 §7.1 U6） |
| 10-23 13:32 | `b0a129753d` | `feat(cli)`：`-C/--cwd` 显式工作目录（#289） | 无 |
| 10-23 13:33 | `29b02e1ac1` | `fix(cli)`：Enter 保存模型选择为默认（#285） | 无 |
| 10-23 13:42 | `a749d471de` | `fix`：plan mode 下工具块保持稳定（#296） | 无（同属"稳定前缀"一类） |
| 10-23 14:23 | `a12325d322` | `fix`：扩展命令轮次先结算再返回（#298） | 无 |
| 10-23 14:28 | `81e79bc4b6` | `feat`：反馈报告可调查（#297） | 无 |

**判读**：#293 与 §3.4 的 CLM 结论同向——**"上下文改写/压缩"与"前缀缓存命中"是互斥成本项**，Step-Code 两段窗口都在做"只追加就保持旧提示位置、内容与时间戳"。本项目侧的对应约束见 §7.1 U6。

### 3.3 Step 5 请求接口流程

```text
终端输入 / 会话恢复
  → Step 登录配置与凭证
  → fetchStepModels：GET 当前 endpoint 的 /models
  → 按 model_type 过滤，解析上下文、视觉和 reasoning_effort 元数据
  → 用户选择模型和 /effort
  → 工作上下文投影 / 必要时 CLM 维护
  → OpenAI-compatible Chat Completions 请求
  → 模型返回文本或 tool_calls
  → 宿主执行工具 / 子 Agent，结果回传
  → 下一轮模型请求，直到任务结束
```

代码证据（固定到 `be88ee77071188975475bcb5ef4e1edd0913d8ad`，10-20 分支状态；10-23 的 16 条提交未改动此流程）：

- `packages/coding-agent/src/step/defaults.ts`：`STEP_DEFAULT_MODEL = "step-5-preview"`。默认名称存在，不等于登录失败时仍有离线可用模型清单。
- `packages/providers/src/step-provider/index.ts`：`fetchStepModels`；仅保留大语言模型、路由模型；有类型元数据时过滤语音，无类型元数据的旧接口有兼容逻辑。
- 同文件 `normalizeStepModel`：固定当前 endpoint、OpenAI completions 方言与 `supportsDeveloperRole: false`，防止旧配置覆盖后错发 developer role。
- `packages/coding-agent/src/features/subagent/execute.ts`：链式结果突出最终输出，中间细节进入结构化结果。

### 3.4 CLM 为什么与推理成本有关

CLM 管理的是**下一次请求的工作上下文视图**，原始会话历史保留。触发维护后，模型选择替换旧正文，宿主验证来源与版本、保护消息与节省量，再接受新投影；拒绝、超时等情况下可回退原生摘要压缩。

默认维护最多两次请求、90 秒、8,192 输出 tokens；接受自动编辑需至少节省 1,024 tokens 且达到 5%。因此收益必须同时考虑维护调用开销、后续输入下降及**前缀缓存受损**，不能直接推断"默认 CLM 必然更便宜"；它也不把旧文本的 KV 无损搬到改写文本。

10-19 的 notices 修复让仅追加消息的请求维持旧提示的位置、内容与时间戳，有利于保持前缀稳定；真实缓存命中率仍须测量。

### 3.5 飞书集成边界

`/connect lark` 把消息和审批映射到当前终端会话；终端和飞书中第一个合法审批生效。它**不启动第二套 Agent 循环**，也不是后台常驻服务。不要把新增飞书连接解释成新增语音处理能力。

---

## 4. vLLM tracking：本期真正改了什么

核对分支：`step-5-preview-tracking`。区间内新增提交仅 `e177c621` / PR #30，时间 2026-10-10 16:48:44（北京时间）。

### 4.1 H200/SM90 支持不是仅改设备白名单

以下 5 条的行号在 `upstream-step5-reference`（= `e177c621` 同版本）上可逐行对上：

1. **后端准入扩大**：`nvidia/sparse_attention.py:933` 的 `supports_compute_capability` 从 SM103 扩为 SM90、SM103；`model.py:994` 的报错文案同步（"requires SM90 or SM103"）。
2. **按设备选择打分内核**：`nvidia/flashinfer_sparse.py:46-49`，SM90 走新增的 `paged_mqa_logits_sm90`，其他路径继续用 FlashInfer 的 `fp8_paged_mqa_logits`。
3. **新增 Hopper 摘要页打分**：`nvidia/ops/sparse_indexer_scores.py:274`，输入 FP8 query、打包 summary KV、head weights、长度与页表；还原 FP8 keys 与 FP32 scales 后算加权分数；**超出有效长度的区域置 −inf**，避免被选中。
4. **top-k 的确定性与布局修复**：`nvidia/ops/sparse_indexer.py` 的 `select_regions`；只有 DeepSelect 保证未排序选择的确定性顺序，其他后端强制排序；DeepSelect 按 32 字节对齐、per_row 写连续行，避免错误复用对齐 stride；代码明确要求精确 FP32 排序。
5. **增加硬件覆盖与参考一致性检查**：`tests/kernels/core/test_step5_sparse_decode_attention.py` 覆盖 top-k=2/7/512、打包 scales、query pairs 与 page boundary；SWA、KV layout 集成测试放宽到 SM90/SM103。**仅核查测试代码，未在 H200 上运行**。

> 注意：本项目 `upstream-step5-reference/` 是只检出 `vllm/models/step5/` 的 partial clone（`blob:none` + shallow），**没有 tests 目录**；测试文件需按 §12 的 URL 读。

### 4.2 当前分支接口流程

```text
/v1/chat/completions
  → Step5ForCausalLM / Step5Model
  → Step5DecoderLayer
      ├─ SWA 层：局部滑窗注意力
      └─ 稀疏全局层：QKV 与 indexer 投影
           → 更新 summary cache
           → 对历史区域打分
              SM90 → paged_mqa_logits_sm90
              其他受支持路径 → FlashInfer 打分
           → select_regions：历史 top-k + 当前区域
           → 读取所选 KV，执行 prefill/decode attention 与 gate
  → MoE / 残差 → logits
  → step5 reasoning/tool parser → 标准响应
  → Agent 宿主执行工具并回传
```

主模型接口见 `nvidia/model.py`；attention 分流见 `nvidia/sparse_attention.py`。均匀短 query 批次可进入 decode 路径，包含 MTP 验证的多 token 行，并支持 CUDA graph 捕获；MTP 模块见 `nvidia/mtp.py`。

### 4.3 对部署的实际含义

- 本期增量让 Hopper/H200 可以使用该分支的原生稀疏路径；**不能外推**任意 CUDA GPU、昇腾或其他后端已支持。
- 当前实现包含 NVIDIA/CUDA/Triton/FlashInfer 依赖。向昇腾迁移需逐项替代摘要缓存、打分、top-k、稀疏 KV 访问和 SWA，**不能仅修改模型 registry**——本项目已完成这套替代，反查表见 §7.4。
- tracking 分支存在代码，不证明同名功能已进入 vLLM 上游正式版本；本系列未完成上游逐 commit 合入审计。
- 官方 README 提供专用 vLLM 镜像与 `step5` reasoning/tool parser、MTP 参数；部署必须确认镜像/源码支持这些接口。
- **行号口径（10-23 新增）**：`e177c621` 的公开分支在 09-22～10-10 经历了一次以 `cute_dsl → flashinfer` 为主的重组，本项目 CUDA 检出 `water18@5d7f9124`（09-20）停在该重组**之前**，因此上表行号只对 `upstream-step5-reference` 有效，对 `water18` 无效（见 §8.3）。

---

## 5. Step 5 Preview 官方技术文档要读什么

10-15 的 *Inside Step 5 Preview* 将架构、稳定性、数据与 Agentic RL 连成系统说明。重点是：92 层窄深设计；SWA 与 Sparse GQA 配合降低长上下文成本；训推数值对齐及异步轨迹重放降低 RL 偏差；MTP、FP8、EP 和 KV 管理联合提升 rollout 效率；压缩进入训练以支持超长任务。

这些是模型与训练基础设施的技术披露；**不能**把 rollout 系统收益直接当成 Step-Code 本地命令提速，也不能把训练中的 compaction 等同于 Step-Code CLM 的完全相同实现。

必须保留文档口径差异：官方 GitHub README 写最大输出可至 1M，API 模型页仍列最大输出 64k；前者还注明输入输出共享总上下文预算。**线上集成按 endpoint 的实际限制，不能以开源部署说明覆盖托管 API 的限制。**

---

## 6. StepAudio 3：对应技术与集成关系

### 6.1 区间内确认的是论文修订，未确认官方三代推理仓

Realtime 论文 9-12 首发、9-19 v2、**10-12 v3**。官方组织仓还包括 Step-Audio、Step-Audio2、Step-Audio-R1、StepAudio-Skills 与旧实时 Demo；截至 10-23 仍未发现独立的 StepAudio 3 完整推理仓。旧仓名称、TTS-3B 的"3B"均不能作为三代已开源的证据。

### 6.2 Realtime 技术流程

```text
持续用户音频
  → 音频编码与语义/声学理解
  → 共享对话上下文
      ├─ 全双工轮次控制：继续听、停顿、附和、打断
      ├─ 私有推理与语音生成并行
      └─ Voice Agent 路由、澄清与异步工具执行
  → 工具结果进入对话 → 流式语音响应
```

技术重点是"持续听说 + 思考 + 行动"的并行协调，而非 ASR→文字 LLM→TTS 串接；论文还说明 Adaptive Thinking 与 MTP 用于控制推理响应成本。**与 Step-Code 的关系**：当前 provider 输入类型是 text/image 并按 model_type 过滤实时语音；要做"语音驱动编程"需自行设计 Realtime 前台与执行端桥接（任务 ID、异步状态、取消、权限），属可选集成方案，不是仓库已实现的事实。

### 6.3 文档入口与阅读目的

| 文档 | 主要阅读目的 |
|---|---|
| StepAudio 3 Realtime v3（arxiv 2609.14005） | 全双工、边想边说、异步工具与 Voice Agent；本期有版本修订 |
| StepAudio 3 Gen（arxiv 2609.12945） | 通用音频生成，区别于实时会话执行链 |
| StepAudio 3 Music（arxiv 2609.16034） | 音乐生成专门技术，区别于 Coding Agent |
| TTS Cookbook | `/audio/speech` 合成调用、voice/speed/发音控制；标注 9-29 发布，属于区间之前教程 |
| Cookbook 总入口 | Realtime 最小对话与 VAD 参数示例、Step 5 工具调用教程 |

---

## 7. 上游内容在本项目的可用性拆解

分类口径：**可用**＝本项目有对应实现位且可立即落地；**部分可用**＝语义可迁移、实现不可移植；**不可用**＝本项目无对应路径。

### 7.1 可用（可直接落地）

**U4 — top-k 确定性 / 平票规则（对应 §4.1 第 4 条）**

- 上游结论：只有 DeepSelect 保证未排序选择的确定性顺序，其他后端强制排序；要求精确 FP32 排序，拒绝可能在临界分数选错的 backend。
- 本项目对应位（已实测）：
  - CUDA 侧开关：`vllm/vllm-ascend-step4/vllm/envs.py` → `VLLM_STEP5_DSA_FORCE_STABLE_TOPK: bool = True`，**默认已是 True**（注释写明 streaming selector 在 SM90 上不确定、稳定选择是正确性优先默认）。
  - 默认值回归断言：`vllm/vllm-ascend-step4/tests/test_envs.py:66`。
  - NPU 侧确定性来源：`vllm-ascend/vllm-ascend-step4/vllm_ascend/models/step5/indexer_region_score.py`（aclnn `IndexerRegionScore`，`top_k` + `sort_mode=1` 升序、`mask_mode=3` 因果）；`generic_block_sparse_attention.py:58` 的 `_append_current_region_kernel`（升序追加、不挤掉选中项）。
- **动作**：写成跨平台对齐用例——同一份 scores 分别喂 aclnn 选择器与 CuTeDSL 选择器，断言 `ids` 序列逐元素一致（§10 P0-1）。

**U5 — 测试覆盖思路（对应 §4.1 第 5 条）**

- 上游用例维度：top-k=2/7/512、打包 scales、query pairs、page boundary；SWA 与 KV layout 放宽到 SM90/SM103。
- 本项目对应位：`vllm/vllm-ascend-step4/tests/kernels/core/test_step5_{decode_batch_invariance,optimus_qknorm_rope_cache,prefill_paged_logits_validation,prefill_union_stream_lifetime}.py`、`tests/v1/attention/test_step5_{mtp_state_machine,sparse_summary_cache}.py`、`vllm-ascend/vllm-ascend-step4/tests/ut/models/test_step5_model.py`、`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_decode_sparse_meta.py`。
- **动作**：把三张用例表（top-k 边界、page boundary、打包 scales）平移成 aclnn 侧参数化用例——这三个维度正是本项目 `indexer_region_score` + GBSA 最容易出契约错的边界。

**U6 — 长会话上下文治理（对应 §3.4 与 §3.2 的 #293）**

- 结论：CLM 改写工作上下文能降后续输入，但**损害前缀缓存**；10-19 notices 修复与 10-22 前缀保持都在做"只追加即保持前缀"。
- 本项目对应位：`vllm-ascend/vllm-ascend-step4/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:121`、`pool_worker.py:172`（读 `vllm_config.cache_config.prefix_match_unit`）；`vllm_ascend/models/step5/summary_cache.py`、`summary_kernels.py`（DSA 摘要 sidecar 随物理页复用，需 `prefix_match_unit` 为 8 的倍数、命中重算 1 block，且**与 KV-connector/offload 不兼容，显式 raise**）。
- **动作**：以"只追加保前缀"为验收模板，写一条"命中率可解释"的验证：记录 `prefix_match_unit`、命中 token、重算 block、sidecar 是否重建；不接受只看单轮 token 数的结论。

### 7.2 部分可用（语义可迁移，实现不可移植）

**U3 — Hopper 摘要页打分核（§4.1 第 3 条）**：实现层不可移植（FlashInfer/FP8 打包 + warp 级打分）；但"超界置 −inf/不可见"的语义在本项目已有等价物——`indexer_region_score.py` 用 `actual_kv_lengths`(seq_len) 驱动因果上界（算子自推每行可见 region 上界 `floor((ctx+j)/8)`，"当前 region 永不占槽"）。**动作**：不移植核，移植测试构造（有效长度边界 + 打包 scales），验证"超界必须不可见"。

**U2 — 按设备选择打分实现（§4.1 第 2 条）**：本项目 NPU 侧只有单一路径（aclnn `IndexerRegionScore`），无设备二分。**动作**：把"打分实现可替换"留在适配层边界——`vllm_ascend/models/step5/dsa_attention.py`（`AscendStep5DSABackend` / `AscendStep5DSAImpl`）之上，不在 `summary_cache.py` 内部写死调用点，便于换核或跟随 cann-ops 版本（§10 P2-2 的缓释手段）。

**U1 — 后端准入（§4.1 第 1 条）**：本项目没有 SM 版本门禁（NPU 不适用）。等价的"能力门禁"落在 `vllm_ascend/models/step5/dsa_attention.py` 与 `vllm_ascend/ascend_config.py`（`step5_prefill_mnd_slot_allocator` 等开关，`:253`/`:389`）。**动作**：照抄其写法——"门禁失败给出精确原因 + 具体可用集合"，补到 cann-ops 版本检查（`IndexerRegionScore`/GBSA 来自 ≥9.2.0-beta.2 外部包，已 churn 一次，见 #65）。

### 7.3 不可用

| 上游内容 | 不可用原因 |
|---|---|
| H200/SM90 硬件能力本身 | 本项目目标硬件是 A2/910B（另有 950DT 验证）；SM90 结论无法外推 |
| FlashInfer / CUTLASS CuTeDSL 依赖清单 | NPU 侧对应实现是 aclnn 单算子 + Triton-ascend，依赖树不同 |
| StepAudio 3 全套（Realtime v3 / Gen / Music / TTS） | 本项目无音频理解/生成路径，Step-Code provider 也过滤语音类型 |
| Step-Code 飞书连接、Step Plan 登录、TUI 面板、`/btw` | 属产品/交互层，与 Step5 推理无关 |
| "CLM 默认更便宜" | 已否证（维护调用开销 + 前缀缓存受损）；"前缀稳定"才是收益前提 |

### 7.4 反向可用：§4.3 的"迁移矩阵"，本项目已逐项完成

| 上游环节（`e177c621`） | 本项目 NPU 实现位 |
|---|---|
| 摘要缓存更新 | `vllm_ascend/models/step5/summary_kernels.py`（Triton-ascend 重实现）+ `summary_cache.py`（fp8 分页 sidecar） |
| 打分 + top-k + 排序 | `vllm_ascend/models/step5/indexer_region_score.py`（aclnn `IndexerRegionScore`） |
| 稀疏 KV 访问 | `vllm_ascend/models/step5/generic_block_sparse_attention.py`（aclnn `GenericBlockSparseAttention`，`PA_BBND` 直通、`is_consistent_topk=False`） |
| SWA 层 | `AscendFlashAttentionBackend` → `flash_attn_npu_3.flash_attn_with_kvcache(window_size=(512,0))` |
| MoE 路由 / FP8 专家 | `device/device_op.py`（`npu_moe_gating_top_k`）、`quantization/methods/w8a8/fp8_block.py`、`ops/grouped_matmul_v5.py`、`ops/fused_moe/moe_mlp.py` |

用法：这份映射即"对标上游 tracking 分支的移植台账"，按季度核对 E-W 漂移。

---

## 8. 本项目自身增量（10-05 → 10-23，代码级）

### 8.1 检出与同源关系

| 本地路径 | 分支 / HEAD | 日期 | 内容 |
|---|---|---|---|
| `vllm/vllm-ascend-step4` | `water18` / `5d7f9124db` | 2026-09-20 | **vLLM 侧**：`vllm/models/step5/**`（CuTeDSL SM90 核）、`model_executor/models/step5_vl.py`、`transformers_utils/configs/step5.py` |
| `vllm-ascend/vllm-ascend-step4` | `main` / `80d265f13e` | 2026-10-14 | **vllm-ascend 插件侧**：`vllm_ascend/models/step5/**`、`worker/step5_csa_slot_manager.py` 等 |
| `_ml` | `main` / `775f7d1` | 2026-10-15 | 上游 `vllm-project/vllm-ascend` 干净镜像（无 Step5 适配） |
| `upstream-step5-reference` | `e177c62` | 2026-10-10 | 上游 `stepfun-ai/vllm` 的 partial clone，只检出 `vllm/models/step5/` |

实测：前两个 checkout 的 `origin` **是同一个私有 fork** `github.com/earthmanylf/vllm-ascend-step4`，只是检出分支不同（不是两个独立仓库）。

### 8.2 vllm-ascend 侧 10-05 → 10-14 的 Step5 提交

| 日期 | SHA | 主题 | 落点 |
|---|---|---|---|
| 10-14 | `79c6aca27d` | `fix(step5)`：适配 `indexer_score_topk` 的 `(score, ids)` 返回（#65） | `models/step5/indexer_region_score.py` |
| 10-14 | `80d265f13e` | `[BugFix] Docker fix`（#66） | 镜像/依赖 |
| 10-13 | `22ead721f4` | `[Feat] Support Step5 ViT`（#54） | `models/step5/vl.py` |
| 10-12 | `1cb6581387` | `feat(model)`：fused_indexer_qkz_norm_rope 接受 split index_q/k/z（#55） | `models/step5/fused_qknorm_rope.py` |
| 10-12 | `eadc44d092` | `[Perf][Model]` 用 aclnn 自定义算子替换 DSA indexer 选择与稀疏注意力（#50） | `indexer_region_score.py` + `generic_block_sparse_attention.py` |
| 10-10 | `cf7a81f93c` | `feat(step5)`：runner 管理的 prefill CSA slot 改为 opt-in（#44） | `ascend_config.py`、`worker/model_runner_v1.py` |

结论：本项目区间内的主要工程动作是 **#50（替换为 aclnn 算子）→ #55（接口拆分）→ #65（接口 churn 修复）** 这条链，与 §4.3"不能只改 registry"的判断吻合；cann-ops 硬依赖风险已被 #65 验证过一次。

### 8.3 vLLM 侧 `water18` 与公开分支已分叉（决定行号口径）

实测：

1. `water18` HEAD = `5d7f9124db`（2026-09-20），该 SHA **出现在公开 `step-5-preview-tracking` 的历史里**（同一提交 `chore(step5): clean kernel license metadata`）。
2. 本地 `vllm/vllm-ascend-step4` 对象库中**没有** `e177c621f2b8ae1be06190bc34eef84646e54f11`（`git cat-file` 失败）。
3. 公开分支在 `5d7f9124` 之后经历了 09-22～10-10 的重组（`mrv2 and code reorg (#14)`、`coder improve (#15)`、`FP8 and BF16 megamoe (#17)`、`Step5 attention optimization (#18)`、`MTP (#19)`、`done (#22)`、`#23`、`#30`）；到 `e177c621` 时 `vllm/models/step5/nvidia/` 下**已无 `ops/cute_dsl/`**，改为 `flashinfer_sparse.py` + `ops/{sparse_indexer,sparse_indexer_scores,swa_decode,summary_update,sparse_prefill_pages}.py`（按 `ref=e177c621` 逐目录实测）。
4. `water18` 树里则是 `nvidia/ops/cute_dsl/**`（`topk_selector_sm90*`、`decode_sm90_gqa`、`csa_compact_*` 等）。

含义：

- §4.1 描述的"SM90 打分核选择 / FlashInfer 回退 / `supports_compute_capability`"这套代码**在本项目 CUDA 检出（water18）里不存在**；它只在 `upstream-step5-reference`（= `e177c621`）里可逐行对上。
- **NPU 对标取代码基线要选 `upstream-step5-reference`**；`water18` 才是与本项目 vLLM v0.27 基座配套的运行时。两者**不能混用行号**。
- 本地 `water18` 在本线落后公开分支 **26 个提交**（09-21～10-10；`compare 5d7f9124db...e177c621` 的这些 Step5 提交都在区间内。同一区间另有一次 `Sync with upstream (#10)` 合并带入上游 vLLM 主线提交，`compare` 的 `total_commits` 为 2220，**不能当作 Step5 提交数引用**）。是否回补取决于这些优化对 A2 的参考价值，目前主要是"稳定 top-k"与"摘要打分语义"，其余为 NVIDIA 专属。

### 8.4 10-20 的 MC2 超时（本项目唯一线上故障证据）

证据文件：`2026-10-23/附件-MoeDistributeDispatchV3超时摘录-20261020.txt`（工作区原路径 `diagnostics/20261020_191341/error_context.txt`）。设备 177.2.25.1，容器 `codex-step5-225-yyy-cann92`，时间 2026-10-20 19:13:41。

- 明确故障：`KERNEL_AIVEC` 任务超时 → 故障 kernel `MoeDistributeDispatchV3_e92e3b86..._1024_mix_aiv`（device 4，stream 61）；随后 `Stream Synchronize failed, retCode=0x30 [vector core timeout]`。
- 已排除：HCCL `FindTaskInfo not found` 是"该 task 不是 HCCL task"的后续查询失败；`UDMA ungrant segment failed, ret=-22` 晚于超时约 6.5 秒，属清理噪声，**不能作为根因**。
- 本项目对应调用点（已实测）：`vllm-ascend/vllm-ascend-step4/vllm_ascend/ops/fused_moe/token_dispatcher.py:232` → `torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)`；MC2 参数装配在同文件 `get_dispatch_mc2_kwargs`，通信方法选择见 `ops/fused_moe/moe_comm_method.py`。
- **待核项（不下结论）**：日志 kernel 名是 `MoeDistributeDispatchV3`，代码调的是 `_v2` 包装；可能是 torch_npu/CANN 内部版本映射，也可能是容器内 torch_npu 版本与 requirements 不一致。须在 177.2.25.1 上核对 `torch_npu.__version__` 与算子映射（§10 P1-1）。
- 与上游材料的关系：§4/§6 只把 MoE 作为结构对待（352 专家、FP8/BF16 混布），**没有覆盖 MC2 dispatch 的运行期超时**；本项目需自建看护项。

---

## 9. 本项目路径映射总表

机器可读版本见同目录 `本项目路径映射-20261023.tsv`。

| # | 上游对象 | 上游定位（`e177c621` 实测） | 本项目路径 | 判定 |
|---|---|---|---|---|
| 1 | 后端准入 | `nvidia/sparse_attention.py:933`、`model.py:994` | `upstream-step5-reference/…/{sparse_attention.py,model.py}`；NPU 等价位 `models/step5/dsa_attention.py`、`vllm_ascend/ascend_config.py` | 部分可用 |
| 2 | 打分实现分派 | `nvidia/flashinfer_sparse.py:46-49` | NPU 单路径 `vllm_ascend/models/step5/indexer_region_score.py`；CUDA 私有实现 `vllm/…/step5/nvidia/ops/cute_dsl/sparse_gqa/indexer_ops/**` | 部分可用 |
| 3 | Hopper 摘要打分核 | `nvidia/ops/sparse_indexer_scores.py:274` | 无实现位；语义等价位 `indexer_region_score.py` | 部分可用（仅语义/测试） |
| 4 | top-k 确定性 | `nvidia/ops/sparse_indexer.py`（`select_regions`） | `vllm/…/vllm/envs.py`（`VLLM_STEP5_DSA_FORCE_STABLE_TOPK`）、`tests/test_envs.py:66`；NPU `indexer_region_score.py`、`generic_block_sparse_attention.py:58` | **可用** |
| 5 | 测试矩阵 | `tests/kernels/core/test_step5_sparse_decode_attention.py` 等 | `vllm/…/tests/kernels/core/test_step5_*.py`、`tests/v1/attention/test_step5_*.py`、`vllm-ascend/…/tests/ut/models/test_step5_model.py` | **可用** |
| 6 | 摘要缓存（CSA） | `nvidia/sparse_summary_cache.py`、`nvidia/ops/summary_update.py`；`…/cute_dsl/…/csa_compact_update_sm90_gqa.py`（water18） | `models/step5/summary_kernels.py`、`summary_cache.py`、`worker/step5_csa_slot_manager.py` | 部分可用 |
| 7 | 稀疏精确注意力 | `nvidia/ops/{sparse_decode_attention,sparse_prefill,sparse_prefill_pages}.py` | `models/step5/generic_block_sparse_attention.py` | 部分可用 |
| 8 | SWA 层 | `nvidia/ops/swa_decode.py` | `AscendFlashAttentionBackend`（FA3 `window_size=(512,0)`） | 部分可用 |
| 9 | MTP | `nvidia/mtp.py`；`step5/{sparse_attention_mtp,mtp_state_machine}.py`（water18） | `models/step5/mtp.py`、`vllm/…/step5/{mtp,sparse_attention_mtp,mtp_state_machine}.py` | 部分可用 |
| 10 | MoE / megamoe | `nvidia/mega_moe.py`、`model.py:726` | `ops/fused_moe/{token_dispatcher,moe_comm_method,moe_mlp,routed_experts}.py`、`ops/grouped_matmul_v5.py`、`device/device_op.py` | 部分可用（含风险） |
| 11 | 模型入口 / 配置 | `nvidia/model.py`、`step5_vl.py`、`configs/step5.py` | `vllm/…/{models/step5/model.py,model_executor/models/step5_vl.py,transformers_utils/configs/step5.py}`；NPU 覆写 `models/step5/{model,vl}.py` | **可用** |
| 12 | Step-Code 上下文治理 | `packages/coding-agent/docs/context-management.md`；#293 | 无对应代码；方法论映射到 `summary_cache.py` + `pool_scheduler.py:121` | 仅方法论 |
| 13 | StepAudio 3 | arxiv 2609.14005 v3 等 | 无 | 不可用 |

---

## 10. 可执行清单（按优先级，含路径与验收）

**P0-1｜跨平台 top-k / ids 对齐用例（源自 U4、U5）**
落点 `vllm-ascend/vllm-ascend-step4/tests/ut/models/test_step5_model.py`（新增参数化 case），必要时 `vllm/vllm-ascend-step4/tests/v1/attention/`。固定一组构造 scores（含平票、`top_k=2/7/512`、page boundary、超有效长度），分别喂 aclnn `IndexerRegionScore` 与 CuTeDSL 选择器。验收：`ids` 逐元素一致、超界列不可见、`-1` 尾填充位置一致。

**P0-2｜订正文档与代码不一致（源自 §11 第 1 条）**
落点 `vllm-ascend/vllm-ascend-step4/vllm_ascend/patch/__init__.py:189`。该 patch 文件在本地检出中不存在；`step5_vl` 已由 vLLM 侧原生注册（`vllm/transformers_utils/config.py:126`、`configs/step5.py:214`、`model_executor/models/registry.py:595`）。验收：注释清单与 `patch/platform/*` 实际文件集合一一对应，并加"清单↔文件存在性"CI 断言。

**P1-1｜MoE dispatch 超时看护（源自 §8.4）**
落点 `ops/fused_moe/token_dispatcher.py:232`、`ops/fused_moe/moe_comm_method.py`。先核对 torch_npu 版本与 v2→V3 算子映射；在 Step5 大 EP（352 专家 + MC2）压测里加 dispatch 超时判据与 plog 采集；把 `MoeDistributeDispatchV3` 写入风险清单。验收：给出该 kernel 超时的触发条件组合，或明确排除版本映射嫌疑。

**P1-2｜前缀缓存与"上下文治理"可解释性验证（源自 U6 + #293）**
落点 `kv_pool/ascend_store/pool_scheduler.py:121`、`models/step5/summary_cache.py`。固定模型/effort，记录 `prefix_match_unit`、命中 token、重算 block、sidecar 是否重建，对照"只追加"与"改写上下文"两种请求模式。验收：给出命中率与重算成本的量化差异。

**P2-1｜1M 上下文阶段 6 压测（源自 §5 口径差异）**
落点 `vllm/…/transformers_utils/configs/step5.py`（`original_max_position_embeddings=1048576`）与工作区脚本 `run_step5_parallel.sh` / `test_step5_parallel.py`。128K→512K→1M 递增，重点看 CSA 槽位容量（`worker/step5_csa_slot_manager.py`）、sidecar 显存、prefix 重算。验收：每档显存/首 token 时延与失败模式，并注明官方 1M 输出 vs API 64k 输出口径。

**P2-2｜cann-ops 版本锁定（源自 U1 + #65）**
落点 `requirements*` 与镜像 Dockerfile。`IndexerRegionScore` / `GenericBlockSparseAttention` 接口版本 pin，并有"签名不符即 fail-fast"前置检查。

> 10-20 版提出的三条复核建议（H200 推理、Step-Code 长任务、语音 Agent）仍成立，但前两条在本项目的落点已经改写为 P1-1/P1-2 与 P0-1，第三条（语音）判定为不可用，见 §7.3。

---

## 11. 口径订正（相对 10-20 归档版）

1. **`patch/platform/patch_step5_vl_config.py` 不存在**：只出现在 `vllm-ascend/vllm-ascend-step4/vllm_ascend/patch/__init__.py:189` 的清单注释里；`step5_vl` 现由 vLLM 侧原生注册（见 §10 P0-2）。
2. **"两个仓库"改口径**：`vllm/vllm-ascend-step4`（water18）与 `vllm-ascend/vllm-ascend-step4`（main）同属 fork `earthmanylf/vllm-ascend-step4`，是同一仓库的两个分支检出。
3. **`water18` 不含 `e177c621`**：公开分支在 09-22～10-10 完成以 `cute_dsl → flashinfer` 为主的重组；§4 的行号只对 `upstream-step5-reference` 有效。
4. **提交清单时区混用**：`Step-Code-commits-20261005-20261020.tsv` 共 91 条（27 merge / 64 非 merge；10-20 正文写 63，实际 64），其中 **5 条为 `Z`（UTC）**、86 条为 `+08:00`；按字符串排序会把那 5 条排到最前（例如 `2026-10-09T21:18:37Z` 实为北京时间 10-10 05:18）。
5. **MoE 缺运行期风险**：10-20 归档只把 MoE 当结构描述，未覆盖 MC2 dispatch 超时；本版补入 §8.4 与 P1-1。
6. **`step5_prefill_mnd_slot_allocator` 定位精确化**：默认值 `default`（`vllm_ascend/ascend_config.py:253/389`），`runner` 模式为 opt-in（#44）。

---

## 12. 证据与复现

**上游（10-23 实测）**

```text
Step-Code 增量：
  GET https://api.github.com/repos/stepfun-ai/Step-Code/commits?since=2026-10-20T11:49:54Z
  → 16 条新增（13 非 merge），HEAD = 81e79bc4b6（10-23 14:28:19 +08）

step-5-preview-tracking：
  GET https://api.github.com/repos/stepfun-ai/vllm/commits?sha=step-5-preview-tracking
  → HEAD = e177c621f2（10-10 16:48:44），区间内未变化

Step-5-Preview 文档仓 / HF：
  HEAD = acf39aef5d（10-15）；HF lastModified = 2026-10-15T01:28:10Z

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

- Step-Code 提交清单：10-20 目录下 `../2026-10-20/Step-Code-commits-20261005-20261020.tsv`
- 关键提交 patch：10-20 目录下 `../2026-10-20/vllm-step5-e177c621-commit.json`
- 故障原文：同目录 `附件-MoeDistributeDispatchV3超时摘录-20261020.txt`（工作区原路径 `diagnostics/20261020_191341/error_context.txt`，未入库）
- 本项目结构基线：`Step5_模型结构分析报告.md`（**仅工作区，未入库**；§11 CUDA 现状 / §12 vllm-ascend 现状 / §12.4 三条算子路线 / §13 打通方案）
- 本文其余"本项目路径"均指工作区 `C:\Users\w00899129\step5` 下私有检出（`vllm/`、`vllm-ascend/`、`upstream-step5-reference/`），**不进本仓库**

---

## 13. 边界与未验证项

1. **未做真机复测**：所有"可用/部分可用"判断来自源码与接口契约，非昇腾/GPU 上的数值或性能验证。
2. **未验证 v2→V3 算子映射**：`token_dispatcher.py:232` 调 `_v2`，故障日志是 `MoeDistributeDispatchV3`；需在 177.2.25.1 上核对 torch_npu 版本。
3. **未做 `e177c621` 与 `water18` 的逐提交 diff**：只证明"两棵树不同 + 本地缺该对象"，重组提交内容未逐条比对。
4. **未审计上游合流**：Step5 能力只存在于 `step-5-preview-tracking`，未做"是否/何时进 vLLM 上游 main"的逐 commit 审计。
5. **cann-ops 接口未做版本矩阵**：`IndexerRegionScore` / `GenericBlockSparseAttention` 在 ≥9.2.0-beta.2 上的签名差异未逐一验证（#65 已 churn 一次）。
6. **Step-Code 10-05→10-20 的 PR 语义仅按提交标题与文件路径判读**，未逐个 review diff。
