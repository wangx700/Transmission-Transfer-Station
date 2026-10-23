# Step 5 Preview、Step-Code 与 StepAudio 3 更新及技术说明

调研区间：2026-10-05 00:00 至 2026-10-20 本次读取时，北京时间。Step-Code 固定到 `be88ee77071188975475bcb5ef4e1edd0913d8ad`（10-20 19:49:54）；vLLM tracking 分支固定到 `e177c621f2b8ae1be06190bc34eef84646e54f11`（10-10 16:48:44）。本文为公开源码、提交记录与文档核查，未执行模型推理或性能复测。分支之后发生变化时，应重新核对。

> **本文件已增补到 2026-10-23**：上文 10-20 正文保留原结论不变，文末「增补（2026-10-23）」章节把区间推进到 10-23，给出上文各条内容的**可用性判定**与本项目路径映射，并订正 6 条旧口径；刷新后的固定版本见该章节开头。

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


归档说明（2026-10-23）：本目录上传报告、[提交清单](Step-Code-commits-20261005-20261020.tsv)与[关键提交 patch 证据](vllm-step5-e177c621-commit.json)。上述源码 checkout 留在调研工作区，在线源码以正文固定版本链接为准。提交清单字段依次为 commit_sha（提交 SHA）、committed_at_with_timezone（带时区的提交时间）、subject（提交标题）；统计区间为北京时间 2026-10-05 至 10-20，包含 merge 提交，与正文的非 merge 数量口径不同。**2026-10-23 增补**：已有刷新——文末增补章节把技术状态核到 10-23，逐条给出上文内容的可用性判定与本项目落点路径，并附路径映射表（`../2026-10-23/本项目路径映射-20261023.tsv`）与 10-20 MC2 dispatch 超时故障摘录。10-20 正文本身不做改写，以保证已发布结论可追溯。

---

## 增补（2026-10-23）：区间推进到 10-23，并给出本项目路径映射

> **增量窗口**：2026-10-20 19:49 → 2026-10-23 14:28（北京时间）；起点仍是 10-05，上文 10-20 正文结论保持原样不改，本节只负责增量复核与"上文哪些内容在本项目真的可用"。
> **刷新后的固定版本**：Step-Code `81e79bc4b6`（10-23 14:28:19 +08）；`stepfun-ai/vllm` 的 `step-5-preview-tracking` = `e177c621f2`（**未变**）；`stepfun-ai/Step-5-Preview` 文档仓 `acf39aef5d`（**未变**）；HF 权重 `lastModified` 仍为 2026-10-15T01:28:10Z。
> **本项目**：工作区 `C:\Users\w00899129\step5`。文中"本项目路径"均相对该工作区，除标注「未找到」外已在本地检出中逐条实测存在；这些私有检出**不随本仓库分发**。
> **方法边界**：公开 API/仓库/文档核查 + 本项目本地源码实测。**未**做真机推理、性能复测、昇腾算子数值验证；未修改任何业务代码。
> **编号说明**：本节内出现的「第 N 节」「§N」「增补 N」一律指**增补章节自身**的编号，与上文 10-20 正文的章节号无关（正文 §4 指 vLLM tracking 技术点，增补 §4 指本项目自身增量）。

### 增补 1. 先看结论

1. **上游在 10-20→10-23 的"推理侧"增量是 0。** `step-5-preview-tracking` 仍锁在 `e177c621`（10-10）；Step-5-Preview 文档仓停在 10-15；HF 官方权重 `lastModified` 仍是 2026-10-15T01:28:10Z。所以本版的重点不是"有没有新代码"，而是**上文 §4.1 那 5 条代码级技术点在本项目的哪条路径上真的可用**（第 3 节）。
2. **唯一有新提交的是 Step-Code（+16 条，13 条非 merge）。** 其中只有 2 条与本项目技术栈同向：长会话**前缀稳定**（`preserve task-state prompt cache prefixes`）与**压缩失败重试**；其余是终端 UI/工具面。对 Step5 推理性能与精度**无影响**（第 2.2 节）。
3. **本项目 10-05→10-23 的全部硬增量都在自有 fork 和一次故障里**：vllm-ascend 侧 `main@80d265f1`（10-14，含 #50 aclnn 替换、#65 接口适配）；vLLM 侧 `water18@5d7f9124`（09-20，**不含** `e177c621`）；10-20 的 MC2 `MoeDistributeDispatchV3` 超时（第 4 节）。**这条结论会改变上文 §4 的适用性判断，务必先读 4.3。**

---

### 增补 2. 上游状态复核（2026-10-23 实测）

#### 2.1 对象级复核

| 对象 | 10-20 正文结论 | 10-23 实测 | 增量 |
|---|---|---|---|
| Step-Code `main` | 固定 `be88ee77`（10-20 19:49） | `81e79bc4b6`（10-23 14:28:19 +08） | **+16 提交**（13 非 merge），全部落在 10-21～10-23 |
| `stepfun-ai/vllm` `step-5-preview-tracking` | `e177c621`（10-10 16:48:44） | `e177c621f2`（**未变**） | 0 |
| `stepfun-ai/Step-5-Preview`（文档/许可仓） | `acf39ae`（10-15） | `acf39aef5d`（**未变**） | 0 |
| HF `stepfun-ai/Step-5-Preview` | `lastModified` 10-15 09:28 | `2026-10-15T01:28:10Z`；downloads 2467；tags 含 `fp8`、`step5_vl` | 0 |
| StepAudio 3 独立推理仓 | 未发现 | 仍未发现；`stepfun-ai/vllm` 的 `feat/step-audio-support` 分支 HEAD 停在 **2026-03-10** | 0 |

> 口径提示：`stepfun-ai/vllm` 的 `main` HEAD 是 `b372ad3e90`（2026-05-28），比 `step-5-preview-tracking` 的 10-10 更旧——**不能用 main 的时间判断 Step5 分支的新旧**，必须按分支看。

#### 2.2 Step-Code 10-20→10-23 提交明细（16 条）

| 时间（+08） | SHA | 主题 | 与本项目相关性 |
|---|---|---|---|
| 10-21 10:57 | `1ccfa0819d` | `fix(cli)`: 命令预览按终端宽度自适应 | 无（UI） |
| 10-21 11:35 | `7ae808a7b9` | Merge #275 | — |
| 10-21 17:32 | `143a9d232a` | `fix(compaction)`: 未完整流重试 + 持久化安全失败诊断 | **方法同向**：长上下文压缩的失败面与重试（对应本项目 CLA‑/CLM 类上下文治理思路） |
| 10-21 17:35 | `118348c05c` | `feat(subagent)`: 模型自选前台/后台泳道（#282） | 无（Agent 编排） |
| 10-21 17:35 | `993e714243` | `fix(workflow)`: `phase()` 误用直接报错（#283） | 无（编排容错） |
| 10-21 17:40/17:41 | `9cc8064` / `8d98d4b` | Merge #284 / #279 | — |
| 10-21 17:42 | `84a16c90ad` | `fix(step)`: 搜索结果按结构解析而非重解析渲染文本（#278） | 无（TUI） |
| 10-21 17:42 | `dff413f436` | `fix(step)`: 更新失败信息包含安装器输出（#277） | 无 |
| 10-22 16:44 / 17:05 | `50aad0a962` / `3de15c1659` | `fix(coding-agent)`: **保留 task-state prompt cache 前缀**（#293） | **强相关**：追加式请求保持前缀稳定 = 本项目 prefix cache / DSA sidecar 命中率的前置条件 |
| 10-23 13:32 | `b0a129753d` | `feat(cli)`: `-C/--cwd` 显式工作目录（#289） | 无 |
| 10-23 13:33 | `29b02e1ac1` | `fix(cli)`: Enter 保存模型选择为默认（#285） | 无 |
| 10-23 13:42 | `a749d471de` | `fix`: plan mode 下工具块保持稳定（#296） | 无（也是"稳定前缀"的一类） |
| 10-23 14:23 | `a12325d322` | `fix`: 扩展命令轮次先结算再返回（#298） | 无 |
| 10-23 14:28 | `81e79bc4b6` | `feat`: 反馈报告可调查（#297） | 无 |

**判读**：`50aad0a962/3de15c1659` 与上文 §3.2 的 CLM 结论同向——**"上下文改写/压缩"与"前缀缓存命中"是互斥成本项**，Step-Code 这边两版都在做"只追加就保持旧提示位置/内容/时间戳"。本项目侧对应约束见 3.1 的 U6。

---

### 增补 3. 10-20 正文内容在本项目的可用性拆解

分类口径：**可用**＝本项目有对应实现位且可立即落地；**部分可用**＝语义可迁移、实现不可移植；**不可用**＝本项目无对应路径。

#### 3.1 可用（可直接落地）

**U4 — top‑k 确定性 / 平票规则（上游 `ops/sparse_indexer.py` 的 `select_regions`，stable top‑k）**

- 上游结论：只有 DeepSelect 保证未排序选择的确定性顺序，其他后端强制排序；要求精确 FP32 排序，拒绝可能在临界分数选错的 backend。
- 本项目对应位（**已实测**）：
  - CUDA 侧开关：`vllm/vllm-ascend-step4/vllm/envs.py` → `VLLM_STEP5_DSA_FORCE_STABLE_TOPK: bool = True`（**默认已是 True**，与上文"建议置 1"一致；注释写明"streaming selector 在 SM90 上不确定，稳定选择是正确性优先默认"）。
  - 该默认值的回归断言：`vllm/vllm-ascend-step4/tests/test_envs.py:66`（`test_step5_stable_topk_is_the_correctness_default`）。
  - NPU 侧确定性来源：`vllm-ascend/vllm-ascend-step4/vllm_ascend/models/step5/indexer_region_score.py`（aclnn `IndexerRegionScore`，`top_k` + `sort_mode=1` 升序、`mask_mode=3` 因果），以及 `generic_block_sparse_attention.py:58` 的 `_append_current_region_kernel`（升序追加、不挤掉选中项）。
- **可用动作**：把"平票/排序规则"写成**跨平台对齐用例**——同一份 scores 张量分别喂 aclnn 选择器与 CuTeDSL 选择器，断言 `ids` 序列逐元素一致（详见 5 节 P0‑1）。

**U6 — 长会话上下文治理（上文 §3.2 CLM 与 notices 修复）**

- 上文结论：CLM 改写工作上下文 → 后续输入下降，但前缀缓存受损；10-19 的 notices 修复让仅追加请求维持旧提示位置。**"默认 CLM 必然更便宜"不能推断**。
- 本项目对应位（已实测）：
  - prefix cache 单元：`vllm-ascend/vllm-ascend-step4/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:121`、`pool_worker.py:172`（读取 `vllm_config.cache_config.prefix_match_unit`）。
  - DSA 摘要 sidecar 与 KV 迁移的冲突（上文未提，本项目已显式约束）：summary 缓存随物理页复用，需 `prefix_match_unit` 为 8 的倍数、命中重算 1 block，且**与 KV‑connector/offload 不兼容（显式 raise）**；实现位于 `vllm_ascend/models/step5/summary_cache.py`、`vllm_ascend/models/step5/summary_kernels.py`。
- **可用动作**：把 Step-Code 的"只追加就保持前缀"当作**验收标准模板**——对本项目写一条"命中率可解释"的验证：记录 `prefix_match_unit`、命中 token 数、重算 block 数、是否触发 sidecar 重建；不接受只看单轮 token 数的结论。

**U5 — 测试覆盖思路（上游 e177c621 的测试矩阵）**

- 上游新增/放宽的用例在 `tests/kernels/core/test_step5_sparse_decode_attention.py`（top‑k=2/7/512、打包 scales、query pairs、page boundary）、`test_step5_swa_decode.py`、`tests/v1/attention/test_step5_kv_layout.py`（放宽到 SM90/SM103）。
  **注意**：本项目 `upstream-step5-reference/` 是**只检出 `vllm/models/step5/` 的 partial clone（`blob:none` + shallow）**，**没有 tests 目录**；用例需按上游 URL 阅读（见第 8 节）。
- 本项目对应位（已实测）：`vllm/vllm-ascend-step4/tests/kernels/core/test_step5_decode_batch_invariance.py`、`.../test_step5_optimus_qknorm_rope_cache.py`、`.../test_step5_prefill_paged_logits_validation.py`、`.../test_step5_prefill_union_stream_lifetime.py`、`tests/v1/attention/test_step5_{mtp_state_machine,sparse_summary_cache}.py`、`vllm-ascend/vllm-ascend-step4/tests/ut/models/test_step5_model.py`、`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_step5_decode_sparse_meta.py`。
- **可用动作**：把上游三张用例表（top‑k 边界、page boundary、打包 scales）**平移成 aclnn 侧参数化用例**——这三个维度正是本项目 `indexer_region_score` + GBSA 最容易出契约错的边界。

#### 3.2 部分可用（语义可迁移，实现不可移植）

**U3 — Hopper 摘要页打分核（上游 `nvidia/ops/sparse_indexer_scores.py`，`paged_mqa_logits_sm90`）**

- 上游语义：输入 FP8 query + 打包 summary KV + head weights + 长度 + 页表；还原 FP8 keys 与 FP32 scales 后算加权分数；**超出有效长度的区域置 −inf**，避免被选中。
- 判定：**实现层不可移植**（FlashInfer/FP8 打包 + warp 级打分）。但**语义在本项目已有等价物**：`indexer_region_score.py` 用 `actual_kv_lengths`(seq_len) 驱动因果上界（算子自推每行可见 region 上界 `floor((ctx+j)/8)`，"当前 region 永不占槽"），超界即不可见。
- **可用动作**：不要移植核，移植**测试构造**——用"有效长度边界 + 打包 scales"两类输入验证"超界必须是 −inf/不可见"。落点 `vllm_ascend/models/step5/indexer_region_score.py` 的 UT 与 `tests/ut/models/test_step5_model.py`。

**U2 — 按设备选择打分实现（上游 `nvidia/flashinfer_sparse.py:46-49`，SM90 走 `paged_mqa_logits_sm90`，其他走 FlashInfer）**

- 本项目 NPU 侧只有**单一路径**（aclnn `IndexerRegionScore`），没有设备二分。
- **可用动作**：把"打分实现可替换"留在适配层边界——`vllm_ascend/models/step5/dsa_attention.py`（`AscendStep5DSABackend` / `AscendStep5DSAImpl`）之上，不要在 `summary_cache.py` 内部写死调用点，便于后续换核/换算子版本（这也是 §3.2 U1 与 5 节 P2‑2 里 cann‑ops 版本风险的缓释手段）。

**U1 — 后端准入（上游 `nvidia/sparse_attention.py:933` `supports_compute_capability` 由 SM103 扩到 SM90/SM103；`model.py:994` 报错文案同步）**

- 本项目**没有 SM 版本门禁**（NPU 不适用）。等价的"能力门禁"实际落在两处：`vllm-ascend/.../vllm_ascend/models/step5/dsa_attention.py`（后端/实现类）与 `vllm_ascend/ascend_config.py`（`step5_prefill_mnd_slot_allocator` 等开关，`:253`/`:389`）。
- **可用动作**：照抄它的**写法**——"门禁失败要给出精确原因 + 具体可用集合"，补到 NPU 侧的 cann‑ops 版本检查（当前 `IndexerRegionScore`/GBSA 接口来自外部包 ≥9.2.0‑beta.2，且已经 churn 过一次，见 #65）。

#### 3.3 不可用（本项目无对应路径，或明确不成立）

| 10-20 正文内容 | 不可用原因 |
|---|---|
| H200/SM90 硬件能力本身 | 本项目目标硬件是 A2/910B（另有 950DT 验证）；SM90 结论无法外推 |
| FlashInfer / CUTLASS CuTeDSL 依赖列表 | NPU 侧对应实现是 aclnn 单算子 + Triton‑ascend，依赖树不同 |
| StepAudio 3 全套（Realtime v3 / Gen / Music / TTS） | 本项目无音频理解/生成路径，Step-Code provider 也过滤语音类型；**不构成集成方案** |
| Step-Code 的飞书连接、Step Plan 登录、TUI 面板、`/btw` | 属产品/交互层，与 Step5 推理无关 |
| "CLM 默认更便宜" | 上文已否证（维护调用开销 + 前缀缓存受损）；本版 Step-Code 新提交进一步说明"前缀稳定"才是收益前提 |
| 上文 §4.3 "改 registry 迁移不了" | **结论仍成立且本项目已逐项替代**（见 3.4）；但迁移正确性依赖 cann‑ops 版本锁定 |

#### 3.4 反向可用：上文 §4.3 的"迁移矩阵"，本项目已经做完

上文列出迁 NPU 必须逐项替代的 5 条：摘要缓存、打分、top‑k、稀疏 KV 访问、SWA。本项目对应实现（均已实测存在）：

| 上游环节（e177c621） | 本项目 NPU 实现位 |
|---|---|
| 摘要缓存更新 | `vllm_ascend/models/step5/summary_kernels.py`（Triton‑ascend 重实现）+ `summary_cache.py`（fp8 分页 sidecar） |
| 打分 + top‑k + 排序 | `vllm_ascend/models/step5/indexer_region_score.py`（aclnn `IndexerRegionScore`） |
| 稀疏 KV 访问（token‑wise flash attn） | `vllm_ascend/models/step5/generic_block_sparse_attention.py`（aclnn `GenericBlockSparseAttention`，`PA_BBND` 直通、`is_consistent_topk=False`） |
| SWA 层 | `AscendFlashAttentionBackend` → `flash_attn_npu_3.flash_attn_with_kvcache(window_size=(512,0))` |
| MoE 路由 / FP8 专家 | `device/device_op.py`（`npu_moe_gating_top_k`）、`quantization/methods/w8a8/fp8_block.py`、`ops/grouped_matmul_v5.py`、`ops/fused_moe/moe_mlp.py` |

→ **用法**：这份映射可以直接作为"对标上游 tracking 分支的移植台账"，每季度核对一次 E‑W 漂移即可。

---

### 增补 4. 本项目自身增量（10-05→10-23，代码级）

#### 4.1 两个 checkout 同源（重要，上文表述需修正）

| 本地路径 | 分支 / HEAD | 日期 | 内容 |
|---|---|---|---|
| `vllm/vllm-ascend-step4` | `water18` / `5d7f9124db` | 2026-09-20 | **vLLM 侧**：`vllm/models/step5/**`（CuTeDSL SM90 核）、`model_executor/models/step5_vl.py`、`transformers_utils/configs/step5.py` |
| `vllm-ascend/vllm-ascend-step4` | `main` / `80d265f13e` | 2026-10-14 | **vllm-ascend 插件侧**：`vllm_ascend/models/step5/**`、`worker/step5_csa_slot_manager.py` 等 |
| `_ml` | `main` / `775f7d1` | 2026-10-15 | 上游 `vllm-project/vllm-ascend` 干净镜像（无 Step5 适配） |
| `upstream-step5-reference` | `e177c62` | 2026-10-10 | 上游 `stepfun-ai/vllm` 的 **partial clone**，只检出 `vllm/models/step5/` |

实测：`vllm/vllm-ascend-step4` 与 `vllm-ascend/vllm-ascend-step4` 的 `origin` **是同一个私有 fork** `github.com/earthmanylf/vllm-ascend-step4`，只是检出分支不同。上文把它们描述成两个独立仓库，建议按上表改口径（避免"两个仓库领先上游多少提交"的重复计数）。

#### 4.2 vllm-ascend 侧 10-05→10-14 的 Step5 提交（本项目真正的本期增量）

`main` 最新 6 条（`git log`，本机实测）：

| 日期 | SHA | 主题 | 落点 |
|---|---|---|---|
| 10-14 | `79c6aca27d` | `fix(step5)`: 适配 `indexer_score_topk` 的 `(score, ids)` 返回（#65） | `vllm_ascend/models/step5/indexer_region_score.py` |
| 10-14 | `80d265f13e` | `[BugFix] Docker fix`（#66） | 镜像/依赖 |
| 10-13 | `22ead721f4` | `[Feat] Support Step5 ViT`（#54） | `vllm_ascend/models/step5/vl.py` |
| 10-12 | `1cb6581387` | `feat(model)`: fused_indexer_qkz_norm_rope 接受 split index_q/k/z（#55） | `vllm_ascend/models/step5/fused_qknorm_rope.py` |
| 10-12 | `eadc44d092` | `[Perf][Model]` 用 aclnn 自定义算子替换 DSA indexer 选择与稀疏注意力（#50） | `indexer_region_score.py` + `generic_block_sparse_attention.py` |
| 10-10 | `cf7a81f93c` | `feat(step5)`: runner 管理的 prefill CSA slot 改为 opt‑in（#44） | `vllm_ascend/ascend_config.py`、`worker/model_runner_v1.py` |

> 结论：10-05→10-23 本项目在 NPU 侧的主要工程动作是 **#50（替换为 aclnn 算子）→ #55（接口拆分）→ #65（接口 churn 修复）** 这一条链，与上游 10-20 版报告"不能只改 registry"的判断完全吻合。本地分析报告 §12.5 的风险 1（cann‑ops 硬依赖）已被 #65 验证过一次。

#### 4.3 vLLM 侧 `water18` 与公开分支已经分叉（**推翻上文 §4 的部分适用性**）

实测事实：

1. `water18` HEAD = `5d7f9124db`（2026-09-20），该 SHA **出现在公开 `step-5-preview-tracking` 的历史里**（同一提交 `chore(step5): clean kernel license metadata`）。
2. 本地 `vllm/vllm-ascend-step4` 的对象库中**没有** `e177c621f2b8ae1be06190bc34eef84646e54f11`（`git cat-file` 失败）。
3. 公开分支在 `5d7f9124` 之后经历了 09-22～10-10 的一串重组（`mrv2 and code reorg (#14)`、`coder improve (#15)`、`FP8 and BF16 megamoe (#17)`、`Step5 attention optimization (#18)`、`MTP (#19)`、`done (#22)`、`#23`、`#30`）；到 `e177c621` 时，`vllm/models/step5/nvidia/` 下**已无 `ops/cute_dsl/`**，改为 `flashinfer_sparse.py` + `ops/{sparse_indexer_sparse_indexer_scores,swa_decode,summary_update,sparse_prefill_pages}.py` 这一套（用 GitHub contents API 按 `ref=e177c621` 逐目录实测）。
4. 反观 `water18` 的树里是 `nvidia/ops/cute_dsl/**`（`topk_selector_sm90*`、`decode_sm90_gqa`、`csa_compact_*` 等）。

**含义（必须写进结论）**：

- 上文 §4 描述的"SM90 打分核选择 / FlashInfer 回退 / `supports_compute_capability`"这套代码，**在本项目的 CUDA 检出（water18）里根本不存在**；它在 `upstream-step5-reference/`（= `e177c621` 同版本）里可以逐行对上（`sparse_attention.py:933`、`flashinfer_sparse.py:46-49`、`ops/sparse_indexer_scores.py:274`、`ops/sparse_indexer.py` 的 `select_regions`）。
- 因此：**做 NPU 对标时，代码基线要选 `upstream-step5-reference`（公开分支）而不是 `water18`**；但 `water18` 才是与本项目 vLLM v0.27 基座配套的运行时。两者**不能混用行号**。
- 本地 `water18` 在本线落后公开分支 **26 个提交**（09-21～10-10，`compare 5d7f9124db...e177c621` 侧证实这些 Step5 提交都在区间内；同一区间另有一次 `Sync with upstream (#10)` 合并带入的上游 vLLM 主线提交，`compare` 的 `total_commits` 为 2220，**不能当作 Step5 提交数引用**）。是否需要回补，取决于上文 §4 的这些优化对 A2 侧仍有参考价值的部分（目前看主要是"稳定 top‑k"与"摘要打分语义"，其余是 NVIDIA 专属）。

#### 4.4 10-20 的 MC2 超时（本项目唯一的一次线上故障证据）

证据文件：`../2026-10-23/附件-MoeDistributeDispatchV3超时摘录-20261020.txt`（工作区原路径 `diagnostics/20261020_191341/error_context.txt`）。设备 177.2.25.1，容器 `codex-step5-225-yyy-cann92`，时间 2026‑10‑20 19:13:41。

- 明确故障：`KERNEL_AIVEC` 任务超时 → 故障 kernel `MoeDistributeDispatchV3_e92e3b86..._1024_mix_aiv`（device 4，stream 61）；随后 `Stream Synchronize failed, retCode=0x30 [vector core timeout]`。
- 已排除：HCCL 的 `FindTaskInfo not found` 是"该 task 不是 HCCL task"的后续查询失败；`UDMA ungrant segment failed, ret=-22` 晚于超时约 6.5 秒，属清理噪声。**不能作为根因。**
- 本项目对应调用点（已实测）：`vllm-ascend/vllm-ascend-step4/vllm_ascend/ops/fused_moe/token_dispatcher.py:232` → `torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)`；MC2 参数装配在同文件 `get_dispatch_mc2_kwargs`，通信方法选择见 `vllm_ascend/ops/fused_moe/moe_comm_method.py`。
- **待核项（不要下结论）**：日志里 kernel 名是 `MoeDistributeDispatchV3`，而本项目代码调的是 `_v2` 包装。二者可能是 torch_npu/CANN 内部的版本映射（v2 接口下发 V3 kernel），也可能是容器内 torch_npu 版本与本项目 requirements 不一致。**必须在 177.2.25.1 上核对 `torch_npu.__version__` 与 `npu_moe_distribute_dispatch_v2` 的算子映射**（见 5 节 P1‑1）。
- 与上文的关系：这是上文完全没有覆盖的风险源——上文把 Step5 的 MoE 只当作结构（`352` 专家、FP8/BF16 混布），没有覆盖 **MC2 dispatch 的运行期超时**。本版据此在 5 节新增一条看护项。

---

### 增补 5. 可执行清单（按优先级，含路径与验收）

**P0‑1｜跨平台 top‑k / ids 对齐用例（源自 U4、U5）**
- 落点：`vllm-ascend/vllm-ascend-step4/tests/ut/models/test_step5_model.py`（新增参数化 case）、必要时 `vllm/vllm-ascend-step4/tests/v1/attention/`。
- 做法：固定一组构造 scores（含**平票**、`top_k=2/7/512`、page boundary、超有效长度），分别喂 aclnn `IndexerRegionScore` 与 CuTeDSL 选择器。
- 验收：`ids` 序列逐元素一致；超界列必须不可见（−inf 语义）；`-1` 尾填充位置一致。

**P0‑2｜订正文档与代码不一致（源自 4.1、第 7 节）**
- 落点：`vllm-ascend/vllm-ascend-step4/vllm_ascend/patch/__init__.py:189`（`7b. patch_step5_vl_config.py` 段落）。
- 事实：该 patch 文件在本地检出中**不存在**；`step5_vl` 已由 vLLM 侧原生注册（`vllm/transformers_utils/config.py:126`、`configs/step5.py:214`、`model_executor/models/registry.py:595`）。
- 验收：注释清单与 `patch/platform/*` 实际文件集合一一对应；CI 加一条"清单↔文件存在性"断言（防再次漂移）。

**P1‑1｜MoE dispatch 超时看护（源自 4.4）**
- 落点：`vllm_ascend/ops/fused_moe/token_dispatcher.py:232`、`vllm_ascend/ops/fused_moe/moe_comm_method.py`。
- 动作：① 先核对 torch_npu 版本与 v2→V3 的算子映射；② 在 Step5 大 EP（352 专家 + MC2）压测里加"dispatch 超时"判据与 plog 采集脚本；③ 把 `MoeDistributeDispatchV3` 写进风险清单。
- 验收：能给出"该 kernel 超时的触发条件（负载/量化/`x_active_mask`/`global_bs` 组合）"或明确排除版本映射嫌疑。

**P1‑2｜前缀缓存与"上下文治理"可解释性验证（源自 U6 + Step-Code #293）**
- 落点：`vllm-ascend/.../kv_pool/ascend_store/pool_scheduler.py:121`、`vllm_ascend/models/step5/summary_cache.py`。
- 动作：固定模型/effort，记录 `prefix_match_unit`、命中 token、重算 block、sidecar 是否重建；对照"只追加"与"改写上下文"两种请求模式。
- 验收：给出命中率与重算成本的量化差异，而不是单轮 tokens 对比。

**P2‑1｜1M 上下文阶段 6 压测（源自上文 §15 + 官方口径差异）**
- 落点：`vllm/vllm-ascend-step4/vllm/transformers_utils/configs/step5.py`（`original_max_position_embeddings=1048576`）与工作区脚本 `run_step5_parallel.sh` / `test_step5_parallel.py`。
- 动作：128K→512K→1M 递增，重点看 CSA 槽位容量（`worker/step5_csa_slot_manager.py`）、sidecar 显存、prefix 重算。
- 验收：给出每档的显存/首 token 时延与失败模式；口径按 endpoint 实际限制注明（官方 README 1M 输出 vs API 页 64k 输出）。

**P2‑2｜cann‑ops 版本锁定（源自 3.2 U1 + #65）**
- 落点：依赖声明与 CI（`vllm-ascend/vllm-ascend-step4/requirements*`、镜像 Dockerfile）。
- 验收：`IndexerRegionScore` / `GenericBlockSparseAttention` 的接口版本被 pin，并有"算子签名不符即 fail‑fast"的前置检查。

---

### 增补 6. 路径映射总表（上游 → 本项目）

机器可读版本见 `../2026-10-23/本项目路径映射-20261023.tsv`。

| # | 上游对象 | 上游定位（`e177c621` 实测） | 本项目路径 | 判定 |
|---|---|---|---|---|
| 1 | 后端准入 | `vllm/models/step5/nvidia/sparse_attention.py:933`、`model.py:994` | `upstream-step5-reference/vllm/models/step5/nvidia/{sparse_attention.py,model.py}`；NPU 等价位 `vllm-ascend/.../models/step5/dsa_attention.py`、`vllm_ascend/ascend_config.py` | 部分可用 |
| 2 | 打分实现分派 | `nvidia/flashinfer_sparse.py:46-49` | NPU 单路径：`vllm_ascend/models/step5/indexer_region_score.py`；CUDA 私有实现：`vllm/vllm-ascend-step4/vllm/models/step5/nvidia/ops/cute_dsl/sparse_gqa/indexer_ops/**` | 部分可用 |
| 3 | Hopper 摘要打分核 | `nvidia/ops/sparse_indexer_scores.py:274`（`paged_mqa_logits_sm90`） | 无实现位；语义等价位 `indexer_region_score.py`（`actual_kv_lengths` 因果上界） | 部分可用（仅语义/测试） |
| 4 | top‑k 确定性 | `nvidia/ops/sparse_indexer.py`（`select_regions`，`sort=True` + stable） | `vllm/vllm-ascend-step4/vllm/envs.py`（`VLLM_STEP5_DSA_FORCE_STABLE_TOPK = True`）、`tests/test_envs.py:66`；NPU：`indexer_region_score.py`（`sort_mode=1`）、`generic_block_sparse_attention.py:58` | **可用** |
| 5 | 测试矩阵 | `tests/kernels/core/test_step5_sparse_decode_attention.py`、`test_step5_swa_decode.py`、`tests/v1/attention/test_step5_kv_layout.py`（本地上游检出**无 tests**，需按 URL 读） | `vllm/vllm-ascend-step4/tests/kernels/core/test_step5_*.py`、`tests/v1/attention/test_step5_*.py`、`vllm-ascend/.../tests/ut/models/test_step5_model.py`、`tests/e2e/nightly/.../test_step5_decode_sparse_meta.py` | **可用** |
| 6 | 摘要缓存（CSA） | `nvidia/sparse_summary_cache.py`、`nvidia/ops/summary_update.py`（公开分支）；`nvidia/ops/cute_dsl/sparse_gqa/indexer_ops/csa_compact_update_sm90_gqa.py`（water18） | `vllm_ascend/models/step5/summary_kernels.py`、`summary_cache.py`、`worker/step5_csa_slot_manager.py` | 部分可用 |
| 7 | 稀疏精确注意力 | `nvidia/ops/sparse_decode_attention.py`、`sparse_prefill.py`、`sparse_prefill_pages.py` | `vllm_ascend/models/step5/generic_block_sparse_attention.py` | 部分可用 |
| 8 | SWA 层 | `nvidia/ops/swa_decode.py` | `vllm_ascend` 的 `AscendFlashAttentionBackend`（FA3 `window_size=(512,0)`） | 部分可用 |
| 9 | MTP | `nvidia/mtp.py`；`vllm/models/step5/sparse_attention_mtp.py`、`mtp_state_machine.py`（water18） | `vllm_ascend/models/step5/mtp.py`、`vllm/vllm-ascend-step4/vllm/models/step5/{mtp,sparse_attention_mtp,mtp_state_machine}.py` | 部分可用 |
| 10 | MoE（含 megamoe） | `nvidia/mega_moe.py`、`nvidia/model.py:726` | `vllm_ascend/ops/fused_moe/{token_dispatcher,moe_comm_method,moe_mlp,routed_experts}.py`、`ops/grouped_matmul_v5.py`、`device/device_op.py` | 部分可用 |
| 11 | 模型入口 / 配置 | `nvidia/model.py`、`vllm/model_executor/models/step5_vl.py`、`transformers_utils/configs/step5.py` | `vllm/vllm-ascend-step4/vllm/{models/step5/model.py,model_executor/models/step5_vl.py,transformers_utils/configs/step5.py}`；NPU 覆写 `vllm_ascend/models/step5/model.py`、`vl.py` | **可用** |
| 12 | Step-Code 上下文治理 | Step-Code `be88ee77` 的 `packages/coding-agent/docs/context-management.md`；10-22 `#293` | 本项目无对应代码（Agent 侧不在本仓）；方法论映射到 `summary_cache.py` + `pool_scheduler.py:121` | 仅方法论 |
| 13 | StepAudio 3 | arxiv 2609.14005 v3 等 | 无 | 不可用 |

---

### 增补 7. 旧口径订正（相对 2026-10-20 版）

1. **`patch/platform/patch_step5_vl_config.py` 不存在。** 该条目只出现在 `vllm-ascend/vllm-ascend-step4/vllm_ascend/patch/__init__.py:189` 的清单注释里；`step5_vl` 现由 vLLM 侧原生注册（`vllm/transformers_utils/config.py:126`、`transformers_utils/configs/step5.py:214`、`model_executor/models/registry.py:595`）。→ 本地分析报告 §12.2 的"配套"清单需删掉此项或标注"已合并/已失效"。
2. **"两个仓库"应改口径。** `vllm/vllm-ascend-step4`（water18）与 `vllm-ascend/vllm-ascend-step4`（main）同属 fork `earthmanylf/vllm-ascend-step4`，是同一仓库的两个分支检出。
3. **`water18` 不含 `e177c621`。** 本地对象库无该提交；公开分支在 09-22～10-10 完成了一次以 `cute_dsl → flashinfer` 为主的重组。→ 上文 §4 的行号只对 `upstream-step5-reference` 有效；`water18` 不能按同一行号索骥。
4. **提交清单时区混用。** `2026-10-20/Step-Code-commits-20261005-20261020.tsv` 共 91 条（27 merge / 64 非 merge，上文正文写 63），其中 **5 条为 `Z`（UTC）**、86 条为 `+08:00`；按字符串排序会把那 5 条排到最前（例如 `2026-10-09T21:18:37Z` 实为北京时间 10-10 05:18）。→ 引用清单时必须先统一时区。
5. **上文对 MoE 只做了结构描述，缺运行期风险。** 本项目 10-20 已出现 `MoeDistributeDispatchV3` 超时（4.4），应进风险清单。
6. **上文"step5_prefill_mnd_slot_allocator 属配套"的定位需精确化**：默认值是 `default`（`vllm_ascend/ascend_config.py:253/389`），`runner` 模式为 opt‑in（#44），未开启时 prefill CSA 槽位由默认分配器管理。

---

### 增补 8. 证据与复现

**上游（10-23 实测，均在北京时间）**

```text
Step-Code 增量：
  GET https://api.github.com/repos/stepfun-ai/Step-Code/commits?since=2026-10-20T11:49:54Z
  → 16 条新增（13 非 merge），HEAD = 81e79bc4b6（10-23 14:28:19 +08）

step-5-preview-tracking：
  GET https://api.github.com/repos/stepfun-ai/vllm/commits?sha=step-5-preview-tracking
  → HEAD = e177c621f2（10-10 16:48:44 +08），与 10-20 版一致

Step-5-Preview 文档仓 / HF：
  HEAD = acf39aef5d（10-15）；HF lastModified = 2026-10-15T01:28:10Z

e177c621 目录结构（确认无 cute_dsl）：
  GET https://api.github.com/repos/stepfun-ai/vllm/contents/vllm/models/step5/nvidia/ops?ref=e177c621f2b8ae1be06190bc34eef84646e54f11
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

- 10-20 基线：即本文前半部分（10-20 正文）；证据文件为同目录 `Step-Code-commits-20261005-20261020.tsv`、`vllm-step5-e177c621-commit.json`
- 故障原文：`../2026-10-23/附件-MoeDistributeDispatchV3超时摘录-20261020.txt`；工作区原始路径为 `diagnostics/20261020_191341/error_context.txt`（未入库）
- 本项目结构基线：`Step5_模型结构分析报告.md`（**仅工作区，未入库**；§11 本项目 CUDA 现状 / §12 vllm-ascend 现状 / §12.4 三条算子路线 / §13 打通方案）
- 本文其余"本项目路径"均指工作区 `C:\Users\w00899129\step5` 下的私有检出（`vllm/`、`vllm-ascend/`、`upstream-step5-reference/`），**不进本仓库**；引用时以路径本身为准，不随仓库分发。

---

### 增补 9. 边界与未验证项

1. **未做真机复测**：所有"可用/部分可用"判断来自源码与接口契约，非昇腾/GPU 上的数值或性能验证。
2. **未验证 v2→V3 算子映射**：`token_dispatcher.py:232` 调 `_v2`，故障日志是 `MoeDistributeDispatchV3`；需在 177.2.25.1 上核对 torch_npu 版本（4.4 P1‑1）。
3. **未做 e177c621 与 water18 的逐提交 diff**：本版只证明了"两棵树不同 + 本地缺该对象"，重组提交的具体内容未逐条比对。
4. **未审计上游合流**：Step5 能力只存在于 `step-5-preview-tracking`，未做"是否/何时进 vLLM 上游 main"的逐 commit 审计（上文同样保留此边界）。
5. **cann‑ops 接口未做版本矩阵**：`IndexerRegionScore` / `GenericBlockSparseAttention` 在 ≥9.2.0‑beta.2 上的具体签名差异未逐一验证（#65 已 churn 一次）。
