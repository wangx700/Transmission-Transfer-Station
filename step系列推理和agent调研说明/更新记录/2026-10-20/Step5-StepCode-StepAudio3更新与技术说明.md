# Step 5 Preview、Step-Code 与 StepAudio 3 更新及技术说明

调研区间：2026-10-05 00:00 至 2026-10-20 本次读取时，北京时间。Step-Code 固定到 `be88ee77071188975475bcb5ef4e1edd0913d8ad`（10-20 19:49:54）；vLLM tracking 分支固定到 `e177c621f2b8ae1be06190bc34eef84646e54f11`（10-10 16:48:44）。本文为公开源码、提交记录与文档核查，未执行模型推理或性能复测。分支之后发生变化时，应重新核对。

## 1. 先看结论

- **推理侧本期最具体的代码增量是 H200/SM90 Sparse Attention 支持。**指定的 `step-5-preview-tracking` 分支在区间内查到 1 条提交：`e177c621`。MTP、视觉入口和稀疏注意力主体属于该分支现有能力，不能全部算成本期新增。
- **Step-Code 本期重点是上下文、后台子 Agent、编排容错和会话产品能力。**当前 main 可达历史中，区间内有 63 条非 merge 提交；这是 Git 提交数量，并非 63 项独立功能。主要变化集中在 10-08 至 10-20。
- **10-15 的 Step 5 官方仓、权重与技术说明更新，使此前“等待官方权重”的结论过时。**现在应引用 `stepfun-ai` 的官方资源，不能沿用本地旧版 `TypeSafeAI` 镜像 README 的许可、参数限制或部署结论。
- **StepAudio 3 与 Step-Code 尚不能画等号。**Step-Code provider 明确过滤语音类型；Realtime 论文中的 Voice Agent 是语音系统自身的异步工具链。论文 v3 在 10-12 修订，不代表这一天才发布模型，也不证明已接入 Step-Code。

## 2. 仓库与文档状态

表注：时间均为北京时间；“本期”指 2026-10-05 至本次读取的 2026-10-20。“未查到”仅指所核查公开仓库、分支和文档，不推断内部研发状态。仓库 push 时间不能代替特定分支的提交时间。

| 对象 | 本期核实结果 | 证据与含义 |
|---|---|---|
| Step-Code main | 持续更新，读取到 10-20 19:49 | [提交历史](https://github.com/stepfun-ai/Step-Code/commits/main/)；详见下一节 |
| vLLM step-5-preview-tracking | 10-10：H200 SM90 sparse attention，PR #30 | [固定提交](https://github.com/stepfun-ai/vllm/commit/e177c621f2b8ae1be06190bc34eef84646e54f11) |
| Step-5-Preview 官方仓 | 10-13 至 10-15 更新许可、vLLM/SGLang 部署与技术博客入口 | [仓库](https://github.com/stepfun-ai/Step-5-Preview)，以文档和资源入口为主；本次 checkout 未见 README 所述 inference/ 目录 |
| 官方 HF Step-5-Preview | FP8 官方模型可访问；API 返回 lastModified=10-15 09:28:10 | [模型元数据](https://huggingface.co/api/models/stepfun-ai/Step-5-Preview)、[权重入口](https://huggingface.co/stepfun-ai/Step-5-Preview) |
| Step-Cookbook | 10-09/10-10 英文 README、10-12 合并 | [仓库](https://github.com/stepfun-ai/Step-Cookbook)；不能把 9-29 的 API 教程当成本期新能力 |
| StepAudio 3 Realtime 文档 | 10-12 20:05:18 提交论文 v3 | [版本记录](https://arxiv.org/abs/2609.14005)；未在此进行 v2/v3 全文差分，因此不把全文技术点都称为 v3 新增 |

## 3. Step-Code 本期修改及逻辑

表注：日期以进入 main 的合并/提交时间为主；部分分支内作者时间早于合并日。“意义”为代码与文档所支持的工程解释，不表示已完成业务性能评测。

| 日期 | 主要变化 | 修改位置、逻辑及意义 |
|---|---|---|
| 10-08 | 远程 Step Plan 登录、then_run 动作衔接 | [PR #222](https://github.com/stepfun-ai/Step-Code/pull/222)、[PR #217](https://github.com/stepfun-ai/Step-Code/pull/217)；改进远程终端认证和动作执行衔接 |
| 10-09～10-10 | 后台子 Agent 结果、父会话唤醒和通知修复 | [#227](https://github.com/stepfun-ai/Step-Code/pull/227)、[#230](https://github.com/stepfun-ai/Step-Code/pull/230)、[#233](https://github.com/stepfun-ai/Step-Code/pull/233)；链式任务返回最后结果，后台结束唤醒父会话，并在最终通知携带完整结果 |
| 10-10～10-11 | /effort 统一；Full Access；CLM 成为默认压缩方式 | [#234](https://github.com/stepfun-ai/Step-Code/pull/234)、[#235](https://github.com/stepfun-ai/Step-Code/pull/235)、[#232](https://github.com/stepfun-ai/Step-Code/pull/232)；改变推理强度入口、权限层级与长会话工作上下文维护 |
| 10-12 | Ultracode 会话状态持久化，减少重复提示；Windows Git Bash 与 Skills 诊断修复 | [#245](https://github.com/stepfun-ai/Step-Code/pull/245)、[#241](https://github.com/stepfun-ai/Step-Code/pull/241)、[#243](https://github.com/stepfun-ai/Step-Code/pull/243)；避免会话恢复后编排模式丢失，减少重复上下文 |
| 10-14 | workflow 生命周期、并发、重试与恢复修复；then_run 走 run_command | [#249](https://github.com/stepfun-ai/Step-Code/pull/249)、[#248](https://github.com/stepfun-ai/Step-Code/pull/248)；取消时先排空子任务，共享并发准入，区分限流与额度耗尽，部分失败明确回传 |
| 10-15～10-16 | StepFun 使用 system role；usage/status；独立任务 review 与工具可用性 | [#252](https://github.com/stepfun-ai/Step-Code/pull/252)、[#251](https://github.com/stepfun-ai/Step-Code/pull/251)、[#242](https://github.com/stepfun-ai/Step-Code/pull/242)；修复协议兼容，补充消耗观测和任务检查 |
| 10-19 | 稳定 live-context notices；SQLite 会话搜索；账户 session 刷新 | [#267](https://github.com/stepfun-ai/Step-Code/pull/267)、[#266](https://github.com/stepfun-ai/Step-Code/pull/266)、[#261](https://github.com/stepfun-ai/Step-Code/pull/261)；保护追加式请求前缀稳定，支持会话检索与认证续期 |
| 10-20 | 飞书连接与镜像审批、最近模型、/btw、终端状态与 Windows 更新修复 | [#260](https://github.com/stepfun-ai/Step-Code/pull/260)、[#259](https://github.com/stepfun-ai/Step-Code/pull/259)、[#268](https://github.com/stepfun-ai/Step-Code/pull/268)、[#274](https://github.com/stepfun-ai/Step-Code/pull/274)、[#271](https://github.com/stepfun-ai/Step-Code/pull/271)；扩展同一会话的交互渠道并改善终端体验 |

### 3.1 Step 5 请求接口流程

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

代码证据（以下链接固定到本次读取版本）：

- [默认模型](https://github.com/stepfun-ai/Step-Code/blob/be88ee77071188975475bcb5ef4e1edd0913d8ad/packages/coding-agent/src/step/defaults.ts#L4)：`STEP_DEFAULT_MODEL = "step-5-preview"`。默认名称存在，不等于登录失败时仍有离线可用模型清单。
- [模型发现与过滤](https://github.com/stepfun-ai/Step-Code/blob/be88ee77071188975475bcb5ef4e1edd0913d8ad/packages/providers/src/step-provider/index.ts#L457)：`fetchStepModels`；[第 102 行](https://github.com/stepfun-ai/Step-Code/blob/be88ee77071188975475bcb5ef4e1edd0913d8ad/packages/providers/src/step-provider/index.ts#L102)仅保留大语言模型、路由模型。有类型元数据时过滤语音；无类型元数据的旧接口有兼容逻辑。
- [协议归一化](https://github.com/stepfun-ai/Step-Code/blob/be88ee77071188975475bcb5ef4e1edd0913d8ad/packages/providers/src/step-provider/index.ts#L270)：`normalizeStepModel` 固定当前 endpoint、OpenAI completions 方言与 `supportsDeveloperRole: false`，防止旧配置覆盖后错发 developer role。
- [子任务结果聚合](https://github.com/stepfun-ai/Step-Code/blob/be88ee77071188975475bcb5ef4e1edd0913d8ad/packages/coding-agent/src/features/subagent/execute.ts#L107)：链式结果突出最终输出，中间细节进入结构化结果。

### 3.2 CLM 为什么与推理成本有关

CLM 管理的是下一次请求的工作上下文视图，原始会话历史保留。触发维护后，模型选择替换旧正文，宿主验证来源与版本、保护消息和节省量，再接受新投影；拒绝、超时等情况下可回退原生摘要压缩。

默认维护最多两次请求、90 秒、8,192 输出 tokens；接受自动编辑需至少节省 1,024 tokens 且达到 5%。因此收益必须同时考虑维护调用开销、后续输入下降及前缀缓存受损，不能直接推断“默认 CLM 必然更便宜”。它也不把旧文本的 KV 无损搬到改写文本。

10-19 的 notices 修复让仅追加消息的请求维持旧提示的位置、内容和时间戳，有利于保持前缀稳定；真实缓存命中率仍须测量。[CLM 技术文档](https://github.com/stepfun-ai/Step-Code/blob/be88ee77071188975475bcb5ef4e1edd0913d8ad/packages/coding-agent/docs/context-management.md)

### 3.3 飞书集成边界

`/connect lark` 把消息和审批映射到当前终端会话；终端和飞书中第一个合法审批生效。它不启动第二套 Agent 循环，也不是后台常驻服务。不要把新增飞书连接解释成新增语音处理能力。[接口与生命周期说明](https://github.com/stepfun-ai/Step-Code/blob/be88ee77071188975475bcb5ef4e1edd0913d8ad/docs/lark-connection.md)

## 4. vLLM tracking：本期真正改了什么

核对分支：[step-5-preview-tracking](https://github.com/stepfun-ai/vllm/tree/step-5-preview-tracking)。区间内新增提交：[e177c621 / PR #30](https://github.com/stepfun-ai/vllm/commit/e177c621f2b8ae1be06190bc34eef84646e54f11)，时间 2026-10-10 16:48:44，北京时间。

### 4.1 H200/SM90 支持不是仅改设备白名单

1. **后端准入扩大。**[sparse_attention.py:933](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/vllm/models/step5/nvidia/sparse_attention.py#L933) 的 `supports_compute_capability` 从 SM103 扩大为 SM90、SM103；模型初始化的报错同步修改。
2. **按设备选择打分内核。**[flashinfer_sparse.py:46](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/vllm/models/step5/nvidia/flashinfer_sparse.py#L46)：SM90 使用新增 `paged_mqa_logits_sm90`，其他路径继续使用 FlashInfer 的 `fp8_paged_mqa_logits`。
3. **新增 Hopper 的摘要页打分。**[sparse_indexer_scores.py:274](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/vllm/models/step5/nvidia/ops/sparse_indexer_scores.py#L274)：输入 FP8 query、打包 summary KV、head weights、长度和页表；还原 FP8 keys 与 FP32 scales，计算加权分数。超出有效长度的区域置负无穷，避免被选中。
4. **top-k 的确定性与布局修复。**[sparse_indexer.py:557](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/vllm/models/step5/nvidia/ops/sparse_indexer.py#L557)：只有 DeepSelect 保证未排序选择的确定顺序；其他后端强制排序。DeepSelect 按 32 字节对齐，per_row 写连续行，避免错误复用对齐 stride。代码明确要求精确 FP32 排序，拒绝可能在临界分数选错的后端。
5. **增加硬件覆盖与参考一致性检查。**[稀疏 decode 测试](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/tests/kernels/core/test_step5_sparse_decode_attention.py)覆盖 top-k=2/7/512、打包 scales、query pairs 和 page boundary；SWA、KV layout 集成测试放宽到 SM90/SM103。这里只核查测试代码，没有在 H200 上运行。

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

主模型接口见 [model.py:2325](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/vllm/models/step5/nvidia/model.py#L2325)；attention 分流见 [sparse_attention.py:380](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/vllm/models/step5/nvidia/sparse_attention.py#L380)。均匀短 query 批次可进入 decode 路径，包含 MTP 验证的多 token 行，并支持 CUDA graph 捕获。MTP 模块见 [mtp.py:93](https://github.com/stepfun-ai/vllm/blob/e177c621f2b8ae1be06190bc34eef84646e54f11/vllm/models/step5/nvidia/mtp.py#L93)。

### 4.3 对部署的实际含义

- 本期增量让 Hopper/H200 可以使用该分支的原生稀疏路径；不能外推任意 CUDA GPU、昇腾或其他后端已支持。
- 当前实现包含 NVIDIA/CUDA/Triton/FlashInfer 依赖。向昇腾迁移需逐项替代摘要缓存、打分、top-k、稀疏 KV 访问和 SWA 等路径，不能仅修改模型 registry。
- tracking 分支存在代码，不证明同名功能已进入 vLLM 上游正式版本；本次未完成上游逐 commit 合入审计。
- 官方 README 提供专用 vLLM 镜像与 `step5` reasoning/tool parser、MTP 参数。部署必须确认镜像/源码支持这些接口；不能直接套用任意旧版 vLLM 命令。

## 5. Step 5 Preview：本期技术文档要读什么

10-15 的 [Inside Step 5 Preview](https://platform.stepfun.com/blog/inside-step-5-preview) 将架构、稳定性、数据与 Agentic RL 连成系统说明。重点是：92 层窄深设计；SWA 与 Sparse GQA 配合降低长上下文成本；训推数值对齐及异步轨迹重放降低 RL 偏差；MTP、FP8、EP 和 KV 管理联合提升 rollout 效率；压缩进入训练以支持超长任务。

这些是模型和训练基础设施的技术披露；不能把 rollout 系统收益直接当成 Step-Code 本地命令提速，也不能把训练中的 compaction 等同于 Step-Code CLM 的完全相同实现。

当前应特别保留文档口径差异：官方 GitHub README 写最大输出可至 1M，API 模型页仍列最大输出 64k。前者还注明输入输出共享总上下文预算。**线上集成按 endpoint 的实际限制，不能以开源部署说明覆盖托管 API 的限制。**[API 规格](https://platform.stepfun.ai/docs/en/guides/models/step-5-preview)、[部署说明](https://github.com/stepfun-ai/Step-5-Preview/blob/acf39ae/README.md)

## 6. StepAudio 3：对应技术与集成关系

### 6.1 本期确认的是论文修订，未确认官方三代推理仓新增

[Realtime 论文](https://arxiv.org/abs/2609.14005) 9-12 首发、9-19 v2、10-12 v3。现有官方组织仓还包括 Step-Audio、Step-Audio2、Step-Audio-R1、StepAudio-Skills 和旧实时 Demo；本次组织仓清单未发现独立 StepAudio 3 完整推理仓。旧仓名称、TTS-3B 的“3B”均不能作为三代已开源的证据。[官方仓库列表接口](https://api.github.com/orgs/stepfun-ai/repos?per_page=100)

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

技术重点是“持续听说 + 思考 + 行动”的并行协调，而非简单 ASR→文字 LLM→TTS 串接。论文还说明 Adaptive Thinking 和 MTP 用于控制推理响应成本；Voice Agent 在外部工具执行期间维持对话，并将结果带回后续交流。[Realtime v3 全文，第 2、6、7 节](https://arxiv.org/html/2609.14005v3)

**与 Step-Code 的关系：**当前 provider 的输入类型是 text/image，且按 model_type 过滤实时语音等条目。要做“语音驱动编程”，可设计 Realtime 前台与 Step-Code 执行端的桥接，处理任务 ID、异步状态、取消和权限；这是可选集成方案，不是本期仓库已实现的事实，也未查到 Realtime 论文宣称其后台固定使用 Step 5 Preview。

### 6.3 文档入口与阅读目的

表注：以下是截至 2026-10-20 可核查的技术与使用文档；除明确写明修订日期者外，不将文档全部内容认定为 10-05 后新增。API 示例与模型内部推理代码是不同层次的证据。

| 文档 | 主要阅读目的 |
|---|---|
| [StepAudio 3 Realtime v3](https://arxiv.org/html/2609.14005v3) | 全双工、边想边说、异步工具与 Voice Agent；本期有版本修订 |
| [StepAudio 3 Gen](https://arxiv.org/abs/2609.12945) | 通用音频生成，区别于实时会话执行链 |
| [StepAudio 3 Music](https://arxiv.org/abs/2609.16034) | 音乐生成专门技术，区别于 Coding Agent |
| [TTS Cookbook](https://platform.stepfun.com/cookbook/stepaudio-3-tts) | `/audio/speech` 合成调用、voice/speed/发音控制；标注 9-29 发布，属于本期前教程 |
| [Cookbook 总入口](https://platform.stepfun.com/cookbook) | 找到 Realtime 最小对话与 VAD 参数示例，以及 Step 5 工具调用教程 |

## 7. 建议优先复核的工程问题

1. **H200 推理：**固定 `e177c621` 与依赖，先跑新增打分/top-k 测试，再验证长 prefill、decode、MTP、工具调用。记录硬件、输入长度、并发、精度及 TP/EP，才能比较 TTFT、TPOT 和吞吐。
2. **Step-Code 长任务：**固定相同模型与 effort，对照 CLM 与原生压缩，统计任务成功率、维护调用、总输入输出和缓存命中；不能只比较单轮 tokens。
3. **语音 Agent：**先确认官方 API 的事件、打断与异步工具接口，再决定是否桥接 Step-Code；不要把两套独立的异步机制直接当成现成集成。

本地证据：`Step-Code/` 为完整调研 checkout，`vllm-step5/` 为指定分支的稀疏 checkout，`Step-5-Preview/` 为官方说明仓；`Step-Code-commits-20261005-20261020.tsv` 保存提交清单，`vllm-step5-e177c621-commit.json` 保存本期关键提交及 patch。未修改这些仓库的业务代码，未提交代码。


归档说明（2026-10-23）：本目录上传报告、[提交清单](Step-Code-commits-20261005-20261020.tsv)与[关键提交 patch 证据](vllm-step5-e177c621-commit.json)。上述源码 checkout 留在调研工作区，在线源码以正文固定版本链接为准。提交清单字段依次为 commit_sha（提交 SHA）、committed_at_with_timezone（带时区的提交时间）、subject（提交标题）；统计区间为北京时间 2026-10-05 至 10-20，包含 merge 提交，与正文的非 merge 数量口径不同。本次仅归档既有调研，未将技术状态刷新到 10-23。
