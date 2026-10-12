# Agent特性详解、承载仓库与竞争力评估

资料时点：2026-10-12（北京时间）。聚焦编码、研究、GUI与实时语音Agent。每项特性给出承载、接口链、收益、边界和参考；未新增业务实测。已有公开技术不一律称为最新原创；本次介绍的是当前仓库公开机制，不推断首次发布时间。PaCoRe的模型推理时计算扩展见06A，本文仅说明它与Agent编排的关系。

## 1. 特性、说明与承载

表注：承载列独立列出实际模型/仓库/宿主；“代码”与“论文机制”分别说明公开范围。P0表示建议优先读实现并验证，P1表示按场景接入，不代表成熟度排名或已支持NPU。

| 特性 | 特性说明 | 承载仓库/组件 | 竞争力与投入判断 | 代码/文档参考 |
|---|---|---|---|---|
| CLM工作视图 | 选择性编辑旧正文，保留原始会话 | Step-Code上下文管理；来源pi-clm | P1：长日志有价值，净成本与语义遗漏需消融 | [context-management](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/docs/context-management.md) |
| 原生压缩完整性 | 摘要与保留消息构成有效续跑上下文 | Step-Code agent-core compaction | P0：可靠性基础，非独有性能优势 | [compaction.ts](https://github.com/stepfun-ai/Step-Code/blob/main/packages/agent-core/src/harness/compaction/compaction.ts) |
| 独立子会话 | 隔离子任务上下文，向主会话回传结果 | Step-Code subagent | P0：适合可分解任务，重复工作与协调有成本 | [执行入口](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/src/features/subagent/execute.ts) |
| Goal持久化与续跑 | 保存目标、预算和停止/恢复状态 | Step-Code Goal宿主 | P0：长期任务运行价值高，仍需验收证据 | [goal-lifecycle](https://github.com/stepfun-ai/Step-Code/blob/main/docs/goal-lifecycle.md) |
| Ultracode/workflow | 用户启用有界编排，区分工作流与Goal | Step-Code workflow扩展 | P1：明确预算和生命周期；不是自动提升推理能力 | [orchestration-lifecycle](https://github.com/stepfun-ai/Step-Code/blob/main/docs/orchestration-lifecycle.md) |
| Skills/Plugin/MCP | 方法说明、功能包与工具连接形成扩展体系 | Step-Code；gelab-zero MCP；StepAudio-Skills | P0：复用与集成价值；权限/取消仍属宿主 | [skills.ts](https://github.com/stepfun-ai/Step-Code/blob/main/packages/agent-core/src/harness/skills.ts) |
| 研究ReAct闭环 | 搜索、浏览、计划与报告交替推进 | StepDeepResearch模型及cortex工具框架 | P1：研究链公开有参考价值，来源正确仍需校验 | [官方架构](https://github.com/stepfun-ai/StepDeepResearch) |
| GUI观测—动作闭环 | 截图理解、执行动作、再次确认状态 | gelab-zero模型/设备客户端/MCP服务 | P1：真实设备执行价值，任务成功率另验 | [gui_agent_loop](https://github.com/stepfun-ai/gelab-zero/blob/main/copilot_agent_client/mcp_agent_loop.py) |
| Think-While-Speaking | 私有推理与短段表达并发 | Audio3论文系统；Console只承载客户端交互 | P1：实时体验潜力，完整公开链未确认 | [论文§6.3](https://arxiv.org/html/2609.14005#S6.SS3) |
| 全双工与异步工具 | 处理听说交叠、打断和工具结果回流 | Audio3系统；Realtime Console/Python Demo与Skills提供接口参考 | P1：语音Agent关键能力，须单独验证旧结果一致性 | [Realtime论文](https://arxiv.org/html/2609.14005)、[Skills](https://github.com/stepfun-ai/StepAudio-Skills) |

## 2. Step-Code使用规模：已有关注，实际用户数未知

本次读取仓库主页为823 Star、62 Fork、3 Watch，页面显示49个Open Issue和2个Open PR；公开仓库有持续迭代和反馈迹象。此前组织列表快照为818 Star、51 Issue，页面统计存在时点差异，因此这里使用本轮主页快照，不将短期差值解释为增长率。[Step-Code主页](https://github.com/stepfun-ai/Step-Code)。

表注：以下为2026-10-12（北京时间）公开页面快照。Star/Fork为累计仓库指标，Issue/PR为页面当时Open数量；不是当天新增，更不是安装数、日活或付费用户。

| 指标 | 本轮快照 | 可支持的判断 | 不能推导 |
|---|---:|---|---|
| Star | 823 | 有一定公开关注度 | 823个实际用户 |
| Fork | 62 | 有人复制或探索代码 | 62个生产部署 |
| Watch | 3 | 少量用户订阅仓库通知 | 只有3人使用 |
| Open Issue | 49 | 存在公开反馈与待处理事项 | 故障率、总反馈人数或满意度 |
| Open PR | 2 | 当前有待处理改动 | 总贡献者数量或生态规模 |

**评价：适合作为有一定关注度的工程研究对象，现有数据不足以断言用户很多或已广泛生产采用。** 下载量、DAU/MAU、安装转化、留存和成功任务数本次未获得可靠公开数据。与大型编码Agent的用户规模比较需同口径数据，不能直接拿Star作为市场份额。近期技术评价应基于实现、任务表现与成本，而不是关注度。

## 3. Step-Code：上下文与长任务运行机制

### 3.1 CLM详细说明
**说明。** 此处CLM指Context Language Models思路在Step-Code中的应用。保留原始会话，同时让模型选择性缩短供后续决策使用的工作视图；它与全历史摘要压缩是不同路径，不是给模型增加永久记忆。

**接口流程：** `检查上下文压力 → 构建可编辑旧正文索引 → 模型返回replacements[{id,text}] → 宿主校验来源/版本/保护内容/节省量 → 接受新工作视图 → 后续模型调用`。不合法编辑被拒绝；符合回退条件时使用原生压缩。新输入、取消或上下文变化还需处理维护中止。

**竞争力评价：长日志任务具有工程价值，单项领先性待验证。** 编译/调优日志可缩短为失败原因、修复位置、当前结果与完整日志入口，使后续多轮更聚焦。代价是维护调用、可能的语义遗漏及前缀缓存复用下降。来源文档明确借鉴pi-clm，不能视为Step独有创新；结构校验也不等于语义无损。

**参考：** [Step-Code上下文管理及校验契约](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/docs/context-management.md)；[pi-clm固定版本](https://github.com/lolipopshock/pi-clm/tree/b84a9d7cbb625cd39539db3bcef72ea9cc89aa89)；[原生压缩实现（用于区分）](https://github.com/stepfun-ai/Step-Code/blob/main/packages/agent-core/src/harness/compaction/compaction.ts)。

**验证重点：** 固定模型/任务/预算，对照原生摘要与CLM；统计成功率、关键约束遗漏、恢复表现和包含维护调用及缓存变化的总成本。

#### 具体场景与成本逻辑

示意：自动调优跑20轮，每轮产生长编译日志。原始历史保留日志以便审计，工作视图将旧日志缩短为“失败算子、报错位置、修复尝试、最新结论、完整日志路径”。当前仍需原文的片段可保留或重新读取；历史保存并不表示模型每轮读取全部历史，也不保证宿主自动实现任意检索。

若一次编辑减少D个后续输入token，预计再调用R轮，潜在输入节省约R·D；但还需减去维护调用、重新读取及前缀缓存损失。不同缓存单价会改变净费用。这个公式仅是粗估，旧上下文可能再次增长，维护也可能反复发生，实际必须累计所有请求用量。

宿主校验能拒绝未知ID、过期版本、保护块编辑或非法结构；无法自动判断“把精度要求0.1%写成1%”这类语义错误。因此必须把约束、已验证配置、失败原因和日志入口保留为关键状态，并做约束遗漏的任务级评测。CLM不是与KV压缩同一技术：文本改变后模型需处理新输入，不能把旧文本KV无条件移植。

### 3.2 原生压缩完整性

**承载：** Step-Code agent-core压缩和会话重建。**流程：** `shouldCompact → prepareCompaction → 生成摘要 → 写CompactionEntry → 用摘要及保留消息重建 → 下一轮`。这条路径重组历史；CLM选择性编辑工作视图，两者应分别消融。

**竞争力：** 提高长任务持续执行可靠性，但属于运行时基本能力。摘要需要保留目标、约束、正在验证的配置、已完成工作和未解决问题；若应生成的摘要为空或与源上下文不匹配，不应覆盖上一有效状态。并发工具结果、新用户输入和取消会改变压缩时机，必须遵守实际会话契约。

**参考：** [compaction.ts](https://github.com/stepfun-ai/Step-Code/blob/main/packages/agent-core/src/harness/compaction/compaction.ts)、[压缩完整性说明](https://github.com/stepfun-ai/Step-Code/blob/main/docs/compaction-integrity.md)。应分别测压缩失败恢复、关键约束遗漏和续跑成功率，不能仅测输入token变少。

### 3.3 独立子会话与结果交付

**承载：** Step-Code subagent执行模块。**流程：** `主任务划分 → 创建独立上下文 → 子任务执行/工具反馈 → 保存完整详情 → 返回状态/结论/证据 → 主任务合并与验收`。[executeSubagent](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/src/features/subagent/execute.ts)。

例如主任务是优化推理服务，可将日志分析、模型配置核对和性能数据分析交给不同子任务；主会话收关键结论与文件定位，而不是三份完整日志。独立上下文减少主会话污染，但子任务可能重复读取和生成，总token不一定下降；同一关键路径的任务也不能随意并行。

**竞争力：** 模块边界明确且子任务相对独立时有价值。共享目录修改冲突、同NPU资源争用、依赖未满足、主会话误解结果会抵消收益。工程建议采用状态、结论、证据、修改范围、下一步组成结果契约；这是一项迁移建议，不宣称当前源码统一强制此格式。验收看总耗时、总费用、冲突与合并返工。

### 3.4 Goal：持久目标、预算、取消与续跑

**承载：** Step-Code Goal与会话宿主。**流程：** `设置目标/预算 → 执行当前轮 → 保存状态和用量 → 允许继续则续跑 → 完成/受阻/预算/取消 → 保存停止状态`。[生命周期](https://github.com/stepfun-ai/Step-Code/blob/main/docs/goal-lifecycle.md)。

Goal提供生命周期，不会自动获得更强模型能力。持久化失败不能声称清理或状态切换成功；取消应进入运行中的session abort路径。预算改变后如何恢复须依宿主状态机。Goal没有独立完成evaluator，模型声明complete不等于产物通过验收。

**竞争力：** 对编译、调优和跨多轮任务有明确工程价值。目标需要可检查，例如“在精度门槛下得到更低P99且输出对照结果”；只写“优化到最好”无法可靠判定。应在宿主增加测试/产物证据门禁，记录设备、参数、日志与结果，这部分是建议改造。验证暂停恢复、进程中断、预算耗尽与错误完成声明。

### 3.5 Ultracode、workflow与cron：近期公开编排机制

**承载：** Step-Code workflow扩展及会话journal。当前文档描述`/ultracode on/off/status`、单轮提示启用、工作流预算与`/ultraloop`别名。workflow是有界编排运行，Goal是持久目标，cron是日历触发，三者不是同一个状态。[orchestration-lifecycle](https://github.com/stepfun-ai/Step-Code/blob/main/docs/orchestration-lifecycle.md)。

**流程：** `用户启用本轮/会话模式 → 宿主记录启用范围 → 工作流调度子任务 → 预算与状态回流 → 完成或停止`。能力发现不等于授权；文档说明模式关闭是引导与日志契约，不是独立硬权限拦截。因此不能称为具备安全沙箱或权限隔离，工具权限与模型thinking设置仍单独管理。

**竞争力：** 明确任务范围、预算和恢复语义有价值；编排名称不证明任务速度或质量领先。cron触发也不保证任务幂等，重跑会修改外部状态时需要租约、去重与验收。建议测同任务单Agent/子会话/workflow三种模式，固定预算并记录协调开销。

### 3.6 Skills、Plugin与MCP

**承载：** Step-Code Skill加载和插件/MCP集成；gelab-zero设备MCP；StepAudio-Skills音频接口封装。**流程：** `发现功能/读取方法说明 → 选择工具及参数 → 宿主执行 → 返回结构化结果/错误 → 模型继续决策`。[skills.ts](https://github.com/stepfun-ai/Step-Code/blob/main/packages/agent-core/src/harness/skills.ts)、[Audio-Skills](https://github.com/stepfun-ai/StepAudio-Skills)。

Skill提供方法，Plugin组合能力，MCP连接工具；它们本身不保证并发、取消、恢复与权限隔离。对NPU调优可先封装“环境核对—编译—运行—指标提取—验收”方法，再用工具控制具体执行。连接SSH只是工具能访问NPU设备，不代表模型部署在NPU。

**竞争力：** 降低重复集成工作和操作遗漏，属于生态互操作能力，不是Step独有路线。集成后应检查工具结果schema、超时、错误分类与资源释放；发现插件不代表调用链在目标环境已经可用。

## 4. StepDeepResearch：证据驱动的研究闭环

**特性与承载：** 单Agent ReAct由研究模型与工具框架共同承担；`batch_web_surfer`执行搜索/浏览，file管理产物，todo管理任务，shell提供命令执行。[官方架构与工具说明](https://github.com/stepfun-ai/StepDeepResearch)。

**流程：** `用户问题 → 明确子问题 → 搜索/浏览 → 原文证据回流 → 评估缺口与矛盾 → 更新todo → 再搜索或写报告 → 引用与内容检查`。这是对公开ReAct路线的逻辑解释，不声称框架强制每步采用同一模板。模型负责何时检索和如何推理，工具负责实际获取内容。

例：调研某模型attention，应将官方配置、论文机制和第三方推断分别保存；工具返回摘要不能取代原文。搜索命中多篇重复新闻并不等于独立证据。`BaseContext.add/get_all`是消息管理抽象，不能据此断定有跨任务长期记忆。[接口代码](https://github.com/stepfun-ai/StepDeepResearch/blob/main/cortex/context/base_context.py)。

**竞争力：** 可读的完整研究执行链具有复用价值，特别适合长文、多个来源和持续反馈；单Agent路线也较容易观察故障。报告流畅不证明事实正确。验收应测引用支持率、原始来源比例、矛盾发现、遗漏、更新时间口径及任务总成本，不将仓库示例报告当作独立benchmark。

## 5. gelab-zero：GUI观测、动作与真实设备反馈

**特性与承载：** 模型提供截图理解与动作决策；设备客户端执行；MCP提供多设备访问与任务接口。[模型与基础设施](https://github.com/stepfun-ai/gelab-zero)、[gui_agent_loop](https://github.com/stepfun-ai/gelab-zero/blob/main/copilot_agent_client/mcp_agent_loop.py)、[MCP服务入口](https://github.com/stepfun-ai/gelab-zero/blob/main/mcp_server/detailed_gelab_mcp_server.py)。

**流程：** `任务/设备状态 → 截图 → 模型输出动作 → 解析/映射 → ADB执行 → 等待界面变化 → 新截图 → 检查进度 → 继续/完成/失败`。点击坐标正确与任务完成是两层指标；动作执行成功只说明命令被接受，界面可能未响应或处于错误页面。

例：查找一项设置，Agent点击后应观察是否进入目标页面；若弹窗遮挡，需根据新截图重规划，而非继续沿旧坐标点按。多设备执行必须绑定设备标识，防止将A设备动作发到B设备。任务重试、租约和停止后的资源释放属于目标宿主需要检查或补充的契约，不由MCP名称保证。

**竞争力：** 模型和设备工程一起公开，降低GUI原型落地门槛；不依赖每个App专用API。但界面变化、加载延迟、分辨率、权限和动作副作用会影响可靠性。应测端到端成功率、误操作、恢复率、步数/费用及多设备并发，而不是只测截图grounding。

## 6. Think-While-Speaking及实时语音Agent
**说明。** 规范名称为Think-While-Speaking（TWS）。Audio3 Realtime以同一音频模型的两个并发调用承担Formulation Brain与Articulation Brain：前者推进私有推理，后者依据当前可用推理和已说内容生成短响应段，并按播放进度释放。默认Speak-First无需等待初始推理前缀；Think-First先等待短前缀。

**接口流程：** `用户音频/历史 → 推理分支持续思考 ∥ 表达分支分段生成 → 播放进度调度 → 新推理供下一段使用 → 完成后补充/纠正`。Adaptive Thinking决定是否进入显式推理，MTP加速私有推理，两者与TWS分别归因。它也不同于全双工听说和异步工具执行。

**竞争力评价：实时语音体验有条件竞争力，正确性与最终完成时间仍需验证。** 能缩短开始表达的等待，但先说出的内容可能需要纠正；两个并发调用亦有额外资源成本。Audio3整系统分数不能证明TWS单项领先，MTP表6的加速也不能算作TWS收益。播放停止不等于工具任务取消，需求变化须丢弃或重新验证旧结果。

**参考：** [Audio3 Realtime §6.3](https://arxiv.org/html/2609.14005#S6.SS3)；[双过程来源Mind-Paced Speaking](https://arxiv.org/abs/2510.09592)；[Realtime Console](https://github.com/stepfun-ai/Step-Realtime-Console)仅作采集、WebSocket和播放接口参考，不证明Audio3完整TWS已开源或已在NPU运行。

**验证重点：** 同质量条件下首音时间、首个有效回答时间、最终正确结论时间、提前错误承诺/纠正率、打断恢复和总费用；填充话术提前不能替代有效回答提前。

### 6.1 并发不等于两个调用互不依赖

表达分支每次利用当时已有推理和已说内容；思考分支继续推进，新的结论只能影响未来片段，已播放内容无法撤回。播放进度决定何时补下一段，避免一次生成过多基于不完整证据的内容。可以先表达已确认条件，复杂结论等待更多推理；但填充话术提早出现不等于首个有效答案提前。

例：用户问比较两种部署的成本，表达分支先明确硬件和时延口径，推理分支计算差异，后续片段给结论。若最初给出未经算完的确定排名，后续可能需要纠正。技术价值在于协调等待与表达，不保证“边说边想”总比先想完更正确。

### 6.2 Adaptive Thinking与MTP分别承载什么

Adaptive Thinking负责是否投入显式推理；MTP加速私有推理流；TWS负责推理与表达调度。Audio3论文中Adaptive Thinking存在推理频率下降但部分任务分数下降的情况，说明少思考不是无代价优化。宽松接受用于私有推理时仍可能改变最终决策，应以最终回答质量验收；不能把更高候选接受率作为正确性保证。[论文§6.3](https://arxiv.org/html/2609.14005#S6.SS3)。

### 6.3 全双工、打断和异步工具

全双工允许系统说话期间继续接收用户；TWS允许完整推理结束前开始表达；异步工具允许外部操作与会话推进重叠。这三条链可以组合，但需要各自的状态：播放片段、当前用户需求、工具任务及结果版本。

建议接口链为`用户新输入 → 判断是附和或需求变更 → 停止/调整当前播放 → 标记需求revision → 取消可取消旧任务 → 新结果按revision检验 → 回流当前会话`。这是建议的宿主一致性契约，不声称所有公开客户端已实现。停止扬声器播放不会自动取消网络搜索、远程命令或外部写入；旧结果到达时需判断仍然适用。

**承载边界：** Audio3论文描述完整系统；Realtime Console/Python Demo提供采集、事件与播放客户端参考，StepAudio-Skills封装音频API。不能把客户端连接成功当作完整TWS或Audio3 NPU链跑通。参考：[Console](https://github.com/stepfun-ai/Step-Realtime-Console)、[Python Demo](https://github.com/stepfun-ai/Step-Realtime-Python-Demo)、[Skills](https://github.com/stepfun-ai/StepAudio-Skills)。

## 7. 竞争力如何比较与投入顺序

表注：以下是建议验收口径，非仓库已实现功能列表。时间均指用户提交任务至相应事件的墙钟耗时；费用包括模型调用与工具/设备消耗，不仅主Agent输入token。

| 特性组 | 主要指标 | 需控制的条件 | 典型误判 |
|---|---|---|---|
| CLM/压缩 | 任务成功率、约束遗漏、总费用、恢复成功 | 相同模型、任务、预算、缓存计费 | 输入变短就认为总成本下降 |
| 子会话/workflow/Goal | 总完成时间、返工、错误完成、资源泄漏 | 相同任务依赖、总预算、工具权限 | 并行数量多就认为更快 |
| 研究ReAct | 引用支持、来源质量、覆盖、矛盾处理 | 相同问题、资料时点、搜索可用性 | 报告长且流畅就认为正确 |
| GUI闭环 | 端到端成功、误操作、恢复、总步数 | 相同设备/App版本、初始状态 | 点击成功就认为任务完成 |
| TWS/全双工/工具 | 首音、首个有效回答、最终正确结论、纠正率 | 相同问题、回答质量、网络/音频条件 | 首音快就认为正确结论更快 |

当前建议先研究Step-Code的结果交付、Goal恢复、压缩完整性与工具契约，再把CLM作为单项可关闭策略验证。需要研究任务时参考StepDeepResearch；有真实设备需求时接gelab-zero；有明确语音交互需求时再接TWS与全双工。对昇腾自动调优，优先建立设备/端口/进程租约、实验记录和验收门禁，这些是迁移建议，不将其归为Step-Code已具备的全部能力。

## 8. Step-Code近期做了什么：按提交历史核验

核验依据为公开commit及合入记录，重点变化是CLM默认化、后台子Agent结果交付与Ultracode状态管理。多数属于已有机制的完善，不能全部称为新技术。当前[CHANGELOG](https://github.com/stepfun-ai/Step-Code/blob/main/CHANGELOG.md)将部分调整列在Unreleased下；主分支合入不等于安装包已发布，使用前需核对版本。

表注：近期范围为2026-09-24至2026-10-12，北京时间。日期对应相关提交或合入；跨日UTC记录换算到北京时间。它不是技术首次提出时间，也不是统一发布日。“承载组件”独立说明实现层次；参考固定到提交。

| 技术/特性 | 提交或合入时间 | 近期变化 | 承载组件 | 技术价值与参考 |
|---|---|---|---|---|
| 模型管理的CLM默认化 | 10月11日合入 | 默认采用工作视图编辑；保留原始历史、校验编辑并保留原生压缩回退 | 上下文维护、会话持久化 | 长日志任务值得优先消融；属于默认策略调整，不是首次发明CLM。[合入](https://github.com/stepfun-ai/Step-Code/commit/bd42141a2077bba8db7cef942929b1fe758a5ce4)、[功能提交](https://github.com/stepfun-ai/Step-Code/commit/baa1ef8e72786aea04ab0d4bc39699f3538ecb26) |
| 后台子Agent事件驱动交付 | 10月9—10日 | 完成/失败/需要输入事件唤醒主Agent；进度通知避免积压及反复触发模型调用 | subagent后台lane、事件通知、主会话 | 修复子任务完成但主任务仍等待的问题。[唤醒](https://github.com/stepfun-ai/Step-Code/commit/01af3937c56edfe9225cf137d11862f9c585c9fd)、[积压修复](https://github.com/stepfun-ai/Step-Code/commit/9fbc37f57decb2e342e6d0e984dbfa6a07d2dae9) |
| 子任务结果与状态契约修复 | 10月9—10日 | 链式任务返回最终结果或停止步骤错误，增加queued/skipped；后台结果上限2000→50000字符 | 子任务执行链、结果记录、最终通知 | 防止只收到第一步或截断报告；不是无限完整结果保证。[链式结果](https://github.com/stepfun-ai/Step-Code/commit/49e42afd8343728d53f1169d2eac8973a806d6c7)、[结果上限](https://github.com/stepfun-ai/Step-Code/commit/bd87c0c214909173f5e96d255ee947ca89514fe9) |
| Ultracode会话状态持久化 | 10月12日 | on/off写入当前会话journal，重开恢复；新会话默认关闭 | workflow扩展、会话journal、状态栏 | 明确编排恢复范围，不修改全局或项目设置。[提交](https://github.com/stepfun-ai/Step-Code/commit/579772281f26ab37d839c3b76c1ac12395620225) |
| Ultracode上下文提示去重 | 10月12日 | 不再每轮重复教程；状态变化时提示，上下文压缩/替换后协调必要状态 | workflow提示、provider请求上下文 | 减少提示冗余，保持模式一致；暂无独立性能数字。[提交](https://github.com/stepfun-ai/Step-Code/commit/b84d7e82f1e850cc729c17dc20cf1bd39569f523) |
| 文件修改后验证then_run | 10月1日 | edit_file/write_file增加后续验证执行能力 | 文件编辑工具 | 衔接修改与检查；不代表验证充分或修改自动回滚。[提交](https://github.com/stepfun-ai/Step-Code/commit/a5c8839a) |
| 压缩溢出恢复与结果保留 | 9月26—28日 | 修复上下文保护、摘要溢出重试和最终工具结果边界 | compaction、工具结果处理 | 长任务可靠性基础；需验关键状态未丢失。[溢出重试](https://github.com/stepfun-ai/Step-Code/commit/13a190ad)、[上下文保护](https://github.com/stepfun-ai/Step-Code/commit/4686fefa) |
| 插件/MCP发现与环境修复 | 9月28—30日 | 完善工具目录刷新、插件Skills/命令加载、MCP发现和stdio环境约束 | 插件加载、MCP目录、子进程 | 提高扩展可用性，属于集成完善。[插件发现](https://github.com/stepfun-ai/Step-Code/commit/bda152e0)、[环境约束](https://github.com/stepfun-ai/Step-Code/commit/466e4377) |

### 8.1 CLM近期变化：成为默认维护策略

接口链是`上下文压力检查 → 索引可编辑旧正文 → 模型提出替换 → 宿主校验 → 保存工作视图 → 后续调用使用新视图`。近期改动还处理显式关闭设置、关闭自动压缩、结束回答后不执行无用维护，以及失败/溢出时原生压缩恢复。完整机制见3.1节。

竞争力应比较包含维护调用及缓存影响的净成本，并确保约束和证据没有被误删；“默认启用”不构成收益证明。对自动调优，重点保留精度门槛、已验证最优配置、未解决失败和日志索引。

### 8.2 子Agent近期变化：可靠唤醒与结果交付

接口链是`主任务分派 → 后台子任务执行 → 普通进度随上下文回流 → 完成/失败/需要输入事件唤醒 → 主Agent读取结果继续`。通知若进入队列后逐条消化，旧进度可能挡住完成事件；每条进度都触发模型也会造成重复调用。近期修复将普通进度与需要续跑的终态事件区别处理。

另一类修复处理结果内容：成功链返回最后步骤产物，失败链返回停止原因及最后完成结果，未启动步骤标为queued或skipped。结果上限提升降低截断风险，但超过上限的内容仍须外部产物/日志入口。对多NPU调优，借鉴价值是状态和证据交付可靠，不是只增加Agent数量。

### 8.3 Ultracode近期变化：编排模式的生命周期

接口链是`用户启用 → 写会话状态 → 当前任务使用workflow → 上下文变化时协调模式提示 → 重开会话恢复`。近期持久化与提示去重针对宿主状态，不提升模型参数能力，也不构成硬权限隔离；工具权限与模式开关分别管理。

Goal、Skills、MCP及子会话保留为核心机制，但本批提交只能证明近期调整，不能证明它们全部首次开发于该时间段。Think-While-Speaking属于Audio3，PaCoRe属于独立推理框架，不列作Step-Code近期新增技术。
