# Step 3.5 / 3.7：推理与 Agent 技术竞争力报告

**资料时点：2026-10-12，北京时间。**基于本次系列调研已读取的官方文档、代码、论文及标注来源的报道整理；部分动态页面于本轮复核。未在 NPU 上加载模型、运行 benchmark 或复现训练。

证据分为【官方配置/代码/论文】【官方测试】【独立评测】【非官方报道/博客】【工程解释/建议】。网络证据链接附在相关段落；文件名仅作交付索引。没有公开代码的机制不补造接口或参数，相关项目也不自动等于模型内部生产实现。

## 1. 主要优势与值得借鉴的点

- **可核查的高效模型结构：**大总容量、较小激活路径，混合局部/全局注意力与 MTP；推理适配可直接检查官方配置和算子语义。
- **SWA + 输出 Gate：**窗口减少直接读取范围，门控调节各 head 的信息注入，Full 层保留全局交互路径。门控不自动跳过 attention 计算。
- **稀疏 MoE 与负载约束：**288 routed experts、top-8、1 shared expert；3.5 论文区分全局专家均衡与 micro-batch EP rank 均衡。
- **后训练与 rollout 一致性：**3.5 MIS-PO、截断感知价值估计及路由稳定性分析，为大规模异步采样提供具体技术抓手。
- **3.7 的视觉与可部署实现：**在语言主干上增加视觉编码器；官方提供多个精度版本和 MTP 服务实验，但收益必须按硬件和负载说明。

来源：[3.5 论文](https://arxiv.org/html/2602.10604v2)、[3.5 官方仓库](https://github.com/stepfun-ai/Step-3.5-Flash)、[3.7 官方仓库](https://github.com/stepfun-ai/Step-3.7-Flash)、[3.7 配置](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/config.json)。

## 2. 结构概览与部署价值

3.5 的语言主干为196B总参数/11B激活、45层，3.7 增加约1.8B视觉编码器，标称约198B。主干配置为33 SWA+12 Full、窗口512、前3层dense+后42层MoE、288 routed experts选8并配1个shared expert，以及3层MTP。完整参数口径、门控、路由代码及视觉流程见 `03-模型结构与技术特性详解.md`。

其竞争力是结构与实现可审计，适合建立本地推理、量化与 rollout 基线；这些公开机制并非全部独有，实际收益仍取决于后端、负载和任务。Step 5 的模型规模、Sparse GQA 以及长程RL披露应独立评价，不将旧代结构当作新代证明。[3.5论文](https://arxiv.org/html/2602.10604v2)、[3.7配置](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/config.json)

## 3. 公开能力结果：分别说明，不拼成统一排名

表注：以下为 3.5 技术报告/官方仓库及 3.7 官方仓库公开结果，2026-10-12 查阅；未经本次独立复现。数字为百分比，不同模型的基准、工具、版本和设置不同，不能横向直接排名；官方宣传的榜首/次席不作为当前全球名次。3.7 工具增强评测也不代表纯模型输入输出能力。

| 模型 / 基准 | 公开分数 | 可关注方向 | 统计或应用边界 |
|---|---:|---|---|
| 3.5 SWE-bench Verified | 74.4 | 软件工程问题定位与修补 | 官方任务/harness 条件，不是任何仓库成功率 |
| 3.5 Terminal-Bench 2.0 | 51.0 | 终端工具执行 | 不与 3.7 的 2.1 或 Step 5 的 v4 直接比较 |
| 3.5 BrowseComp + Context Manage | 69.0 | 搜索与上下文管理 | 含上下文管理组件，不归因于模型单项变化 |
| 3.5 τ²-Bench | 88.2 | 多轮工具与业务任务 | 指定环境评分，非生产 SLA |
| 3.7 ClawEval-1.1 | 67.1 | 多轮工作流完整性 | 官方公布任务集结果，当前排名需另查 |
| 3.7 SimpleVQA（Search） | 79.2 | 视觉问题与检索验证 | 带搜索工具 |
| 3.7 V*（Python） | 95.3 | 视觉定位与工具推理 | 带 Python 工具 |
| 3.7 SWE-Bench PRO | 56.3 | 工程任务 | 不与 SWE-bench Verified 74.4 直接比较 |

来源：[3.5 结果与设置](https://github.com/stepfun-ai/Step-3.5-Flash)、[3.7 结果与设置](https://github.com/stepfun-ai/Step-3.7-Flash)。不同部署平台的速度、价格和限额会变动；本报告不把宣传峰值当作固定端点能力。

## 4. 后训练 RL 与推理设施

### 4.1 可借鉴的训练机制

Step 3.5 的关键问题定义是：训练策略与 rollout 策略可能因权重过时、数值精度、内核或路由而产生概率偏差；长链会放大偏差。**MIS-PO 使用 token 与轨迹层面的概率比构造过滤 mask**，控制失配样本进入更新。工程上应同时看原始吞吐、过滤保留率和学习效果，否则生成更快可能只带来更多无效样本。[论文 §5.2](https://arxiv.org/html/2602.10604v2)

其他值得保留的机制是：区分上下文截断与真实失败，用 value estimate 处理截断的后续价值；可验证任务使用程序 checker，非可验证任务使用校准过的奖励模型/rubric；监控 MoE 路由信心和训练/推理一致性。带 value 的处理不能直接等同于无 critic 算法。上述是 3.5 证据，未确认 Step 5 沿用。

### 4.2 Rollout 与训练的接口流程

**异步 RL 总览：任务流、权重流与四个控制点。** 实线表示主流程，虚线表示跨侧通知或条件影响；生成侧与训练侧可以并行，不要求每批生成结束后才启动训练。

`数据源 → 积压准入 → 记录调度版本 → 等待推理版本/并发名额 → rollout → 待训练队列 → 凑批/训练 → 权重同步 → 后续采样`。

调用链：`start → _control_worker / _generation_worker → 完成回调 → get_train_samples → _wait_for_next_weight / sync_weight`，训练和生成可并行。具体同步与取样顺序以源码为准；它是 StepTronOSS 的可核查流控，不等于 Step 5 生产 RL 内部架构。[流控代码](https://github.com/stepfun-ai/SteptronOss/blob/dev/steptronoss/core/generators/flow_controller.py)

表注：以下单位来自代码语义；版本差按训练版本计，不是秒。prompt/生成任务的数量不等于 tokens 数，不能将配置直接当作 GPU 容量。

| 控制点 | 判断或配置 | 等待发生在哪里 | 目的 |
|---|---|---|---|
| 积压背压 | `len(pre_train) + len(running_genables) <= max_untrained_prompts`，同时检查 `pre_gen` 是否为空 | 任务数据源暂停接单 | 控制尚未训练的任务积压 |
| 生成并发 | `max_concurrent_genables` | 生成控制器的请求执行入口 | 控制同时执行的生成任务 |
| 训练批次 | `len(pre_train) >= prompt_per_iter` | 训练侧等待结果凑批 | 决定每次交接的 prompt 数 |
| 版本流控 | 在途任务最大 `next_version - scheduled_version <= max_staleness` | 训练侧推进版本之前等待旧任务 | 限制长任务落后训练策略的版本距离 |

两个容易混淆的边界：积压检查使用 `<=`，是提交前准入阈值，不是任何时刻严格不超过配置值的硬容量；`scheduled_version` 是任务调度标签，生成侧只等待 `infer_weight_version >= scheduled_version`，不能因此认定每个采样 token 都来自该标签所指的同一权重。版本流控也不替代训练/rollout 的 logprob 对齐与概率失配处理。[相关条件与回调实现](https://github.com/stepfun-ai/SteptronOss/blob/dev/steptronoss/core/generators/flow_controller.py)

StepTronOSS 提供可检查的实现入口：`VLLMDeployConfig` 生成服务配置，controller 启动并注册 endpoint，client 生成/重载权重；`SimpleFlowController` 与 `FullyAsyncFlowController` 管理版本、在途请求与队列。`max_staleness` 按策略版本步数计，`max_untrained_prompts` 控制背压，`max_concurrent_genables` 控制并发。热加载接口存在，不等于已证明零停顿或任意 KV 可复用。[部署配置](https://github.com/stepfun-ai/SteptronOss/blob/dev/steptronoss/exp/inference.py)、[流控](https://github.com/stepfun-ai/SteptronOss/blob/dev/steptronoss/core/generators/flow_controller.py)、[RL 配置](https://github.com/stepfun-ai/SteptronOss/blob/dev/steptronoss/exp/rl.py)、[vLLM client](https://github.com/stepfun-ai/SteptronOss/blob/dev/steptronoss/generation/vllm/vllm_client.py)

### 4.3 StepMesh 与其他设施的职责

StepMesh 实现 Attention–FFN（A/F）分离通信：**attention worker PushPull/ZBatchPushPull → FFN server GetBatch → 计算 → Respond → worker Wait**。它关注低延迟通信和 RDMA；P/D 分离划分 prefill/decode，训练/rollout 分离划分工作负载，三者不可混称。[StepMesh 接口](https://github.com/stepfun-ai/StepMesh)

StepTronOSS 便于学习训练配置与版本流控，verl/slime 可作为通用 RL 系统参考，vLLM/SGLang 负责模型服务与调度。未发现足够同条件证据证明 StepTronOSS/StepMesh 性能胜过这些系统。选择应基于真实瓶颈和支持的算子，而不是将所有框架叠加。[verl](https://github.com/verl-project/verl)、[slime](https://github.com/THUDM/slime)、[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)

## 5. Agent、Skills 与 Plugin：接口与业务借鉴

Model 决定下一步和工具参数；harness 管理执行循环、上下文、预算与恢复；plugin/MCP 接入工具；skill 提供领域指令、脚本和资源；bench 定义任务环境与验收。模型支持 tool calling，并不意味着它自带企业系统访问能力。

```mermaid
flowchart LR
  U[用户目标与验收条件] --> D[发现 skill：名称与描述]
  D --> S[按需读取指令、脚本、模板]
  S --> M[模型判断下一步]
  M --> P[Plugin 或 MCP 工具]
  P --> O[执行结果与环境状态]
  O --> C[测试、公式、引用、最终状态检查]
  C -->|未通过且预算允许| M
  C -->|通过| F[交付产物]
  C --> L[失败分类、重放与训练样本]
```

图注：建议的业务流程。Step-Code 已实现技能/插件及执行框架接口；各领域 checker 需按业务补齐，不由 loader 自动保证。

**代码证据：**Step-Code `loadSkillsFromDir` 扫描技能目录，`loadSkillFromFile` 解析 frontmatter、校验名称/描述，`formatSkillsForPrompt` 生成可发现的技能信息。插件侧 `readStepPluginManifest` / `parseStepPluginManifest` 读取并解析 `step.plugin.json`，代码也包含兼容 manifest 路径；再由框架装载对应能力。调用链可追溯，不等于任意插件全部兼容。[skills.ts](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/src/core/skills.ts)、[plugins.ts](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/src/step/plugins.ts)、[goal 生命周期](https://github.com/stepfun-ai/Step-Code/blob/main/docs/goal-lifecycle.md)


### 5.1 哪些项目值得单独看

表注：本表为相关项目职责和工程借鉴，不是 3.5/3.7 训练配方证明，也不是对全部框架的性能排名；建议依任务和后端支持进行验证。

| 项目 | 技术入口 | 借鉴点 | 边界 |
|---|---|---|---|
| StepTronOSS | 生成控制器、权重同步、版本/积压流控 | 异步 rollout 与样本生命周期 | 不等于已证明其接口就是两个模型的完整生产 RL 栈 |
| StepMesh | Attention/FFN worker-server 与 RDMA | 分开资源池与专家计算通信 | A/F 分离不同于 prefill/decode 分离 |
| Step-Code | Skills loader、Plugin manifest、子会话、Goal | 工具工作流、执行与验收、上下文管理 | harness 组件不是模型架构，后来的 Step 5 框架分数不移植到 3.5/3.7 |
| StepDeepResearch | 搜索、计算、证据组织与研究验收 | 专业研究工作流 | 不认定为 3.5/3.7 金融能力的完整训练来源 |
| PaCoRe | 多轮并行结论提取与综合 | 高价值可验证推理 | 独立方法；不是每个模型请求默认采用，也不是低成本机制 |

来源：[StepTronOSS](https://github.com/stepfun-ai/SteptronOss)、[StepMesh](https://github.com/stepfun-ai/StepMesh)、[Step-Code](https://github.com/stepfun-ai/Step-Code)、[StepDeepResearch](https://github.com/stepfun-ai/StepDeepResearch)、[PaCoRe](https://github.com/stepfun-ai/PaCoRe)。

## 6. Step 3.5 / 3.7：已有支持，但按设备、版本与特性分层说明

### 6.1 硬证据与适用范围

**PR #10556 已于 2026-07-13 合入**，标题为支持 Step 3.5/3.7 Flash on Ascend 950，正文同时记录 A3 与 950 的验证，并包含融合 Q/K norm + RoPE、FlashComm1 等工作。3.7 日志展示 API 200 和 MTP 接受统计；这些是具体功能证据，不能从短日志推导稳定吞吐 SLA。[PR 内容与合入状态](https://github.com/vllm-project/vllm-ascend/pull/10556)

**v0.26.0rc1 发布说明明确列出 Step 3.5/3.7 的 Ascend 950 支持。**同一 release 也明确它是 model-restricted 版本，列出的充分验证集合为 Kimi K3、GLM-5.2、DeepSeek V4 两个版本，其他模型可用性不保证。因此同时承认“有适配”与“未进入该 release 的全部充分验证集合”，避免只摘支持一句话。[release 原文](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.26.0rc1)

**3.7 MRV2 任务 #14205 当前仍 open。**任务说明 MRV1 路径可用，并要求迁移 proposer、attention metadata 和采样。未关闭任务不等于没有任何 MRV2 实验代码，但不足以声称完成精度/性能验收。近期版本将部分 context-parallel 能力迁到 MRV2，因此 Step 的 MRV1/MRV2 与长上下文功能组合必须单列，不能分别看到“模型支持”和“PCP 支持”就认为交集可用。[MRV2 任务](https://github.com/vllm-project/vllm-ascend/issues/14205)

表注：2026-10-12 公开证据快照；“另验”不表示功能不存在。A3/950 为合入 PR 的设备范围；910B 为社区分支观察，证据等级不同。本次未真机复现。

| 场景 | Step 3.5 | Step 3.7 | 证据与限制 |
|---|---|---|---|
| A3/950 文本推理 | 有适配与验证记录 | 有适配与功能记录 | #10556，不推广为任意 tag/参数组合 |
| MTP-3 | 有 Ascend proposer 与测试入口 | 共用相关 MTP 路径及测试模型列表 | 测试代码存在不代表最新 CI 已执行通过 |
| 图片/视觉 | 3.5 为文本模型 | 上游有 vision encoder/projector 与语言接入 | 3.7 图片输入需独立验证，不用纯文本 API 200 代替 |
| 256K/并发/缓存复用 | 另验实际 cache 和部署容量 | 另验并包含图像 token | 开源上下文长度不是给定设备上的容量证明 |
| MRV2 + MTP/CP | 不给统一可用结论 | 未取得完整完成验收证据 | #14205；Runner/metadata/采样须一致 |
| 910B | 社区有已验证分支报道 | 未由本次材料证明同样完整支持 | [上海适配团队分支说明](https://github.com/Ascend-ShangHai-LLM)，不是所有 910B 安装的保证 |

### 6.2 具体接入代码流程

```text
HF architecture
  → vLLM ModelRegistry
  → Step3p5ForCausalLM / Step3p7ForConditionalGeneration
  → 3.7 视觉编码与 projector（视觉请求）+ text_config 对应语言模型
  → Ascend 模型 patch / DeviceOperator
  → QKV + Q/K norm/RoPE → Attention → head gate → o_proj
  → FusedMoEFactory：路由/bias/缩放/激活/EP
  → 主模型 logits 与采样
  → 若启用 MTP：AscendStep3p5MTPProposer → 验证/接受与 cache 更新
```

图注：根据固定源码整理的接口链；不是每次请求都走视觉或 MTP。3.7 使用 `config.text_config` 加载语言模型，不应随意扁平化嵌套配置。源码入口：[3.7 模型](https://github.com/vllm-project/vllm/blob/03091e28b8ba956634c153beab0e52decafdef31/vllm/model_executor/models/step3p7.py)、[Ascend Step patch](https://github.com/vllm-project/vllm-ascend/blob/c076a131097fdfac9db4b57c8cf55bfd45ea0ca5/vllm_ascend/patch/worker/patch_step3p5.py)、[MTP proposer](https://github.com/vllm-project/vllm-ascend/blob/c076a131097fdfac9db4b57c8cf55bfd45ea0ca5/vllm_ascend/spec_decode/step3p5.py)。

代码中的 Gate 仍是在 attention 输出后乘 `sigmoid(g_proj(hidden))`，再做输出投影；NPU 融合优化不应改变这一语义。MoE 配置还涉及 `router_bias`、`moe_router_scaling_factor`、`swiglustep` 等，不能把它当成任意普通 softmax MoE 自动替换。[对应 patch](https://github.com/vllm-project/vllm-ascend/blob/c076a131097fdfac9db4b57c8cf55bfd45ea0ca5/vllm_ascend/patch/worker/patch_step3p5.py)

**测试证据要按实际内容读。**当前四卡 MTP 测试的 `MODELS` 同时列 3.5/3.7；函数名虽带 `tp8`，实际配置 `tensor_parallel_size=4`、`max_model_len=8192`、MTP=3。主要断言候选位置接受率，不是完整验证“所有生成 token 与 HF/无 MTP 完全一致”，也不是 256K 或图片测试。[e2e 测试](https://github.com/vllm-project/vllm-ascend/blob/c076a131097fdfac9db4b57c8cf55bfd45ea0ca5/tests/e2e/pull_request/four_card/spec_decode/test_mtp_step3p5.py)

另有 source regression 对 metadata、query 起点、draft window、reject token 参数的 AST 检查；它可保护接口契约，但并非真机算子精度或性能测试。[接口回归测试](https://github.com/vllm-project/vllm-ascend/blob/c076a131097fdfac9db4b57c8cf55bfd45ea0ca5/tests/ut/spec_decode/test_step3p5_source_regression.py)

## 7. 面向昇腾的优先级

优先选择已有 A3/950 证据的3.5/3.7版本组合，固定 BF16 文本基线，再逐项验证工具、多模态、MTP、长上下文与并发；MRV2/PCP和具体量化格式单独判断。NVIDIA MTP吞吐与NVFP4格式不外推到NPU。依赖和容量估计见 `05-昇腾适配状态依赖与验证边界.md`。

Agent侧优先借鉴可复现验收、上下文与任务生命周期；Step-Code框架优势及其到MindSpeed-Agent的落点在独立专题中介绍，不把Step5框架测试分数归给3.5/3.7。PaCoRe作为独立推理方法保留在结构专题附录与图集，不宣称它是任何一代模型的默认推理路径。
