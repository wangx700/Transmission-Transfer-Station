# 推理与Agent七项特性说明及竞争力评估

详细内容已按领域拆开：[推理特性详解（含Step5 Attention）](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step系列推理和agent调研说明/核心技术报告/06A-推理特性详解与竞争力评估.md)；[Agent特性详解、承载仓库与竞争力](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step系列推理和agent调研说明/核心技术报告/07-Agent特性详解、承载仓库与竞争力评估.md)。本页保留七项特性简要对照；Agent详版另含Step-Code使用规模及近期公开编排机制。

资料时点：2026-10-12（北京时间）。仅关注推理与Agent；依据公开论文、官方文档及代码入口，未新增性能实测。竞争力结论是针对所列场景的工程判断，不是全球排名，也不将整套模型或系统的成绩归因于某一个特性。源码main链接可能变化，实际复现需固定版本。

## 1. 竞争力总览

表注：“较强”表示有明确收益机制并有相关实验支持；“条件性”表示收益依赖负载、硬件或预算；“待验证”表示已有实现或机制，但缺少充分单项对照。独特性单独判断：通用技术有竞争力，不等于Step独有。表内评价对应后文来源及边界。

| 特性 | 主要解决的问题 | 是否有竞争力 | 相对独特性与证据边界 |
|---|---|---|---|
| EP均衡 | MoE最忙rank拖慢同步执行 | 条件性：负载偏斜时有价值 | rank级约束有针对性；不是Step独有，训练约束不能替代线上EPLB |
| Gated SWA | 窗口注意力省计算但可能引入无用信息 | 较强：成本与质量折中 | 门控并非独有；混合布局、head数与MTP兼容的组合设计值得关注 |
| AFD | Attention与FFN资源需求、有效batch不匹配 | 条件性：高并发服务更有潜力 | 模型与系统协同有价值；跨池往返可能损害低并发延迟 |
| PaCoRe | 单条推理轨迹或单次上下文限制解题能力 | 较强：高预算复杂推理；成本优势须另验 | 多轮协调与压缩有辨识度；需要将质量收益与大量额外生成成本一起比较 |
| MTP-3 | 自回归生成串行轮次多 | 较强：接受率高、验证高效时 | MTP/投机解码是通用路线；Step的轻量预测模块及注意力协同值得复用 |
| CLM | 长任务旧日志反复进入模型上下文 | 有工程竞争力，领先性待验证 | Step-Code借鉴pi-clm；工作视图、保护与回退机制比名称更重要 |
| Think-While-Speaking | 必须等完整推理才开始说话 | 条件性：实时语音响应体验有价值 | 双过程与播放调度有针对性；单项质量/延迟领先证据仍不足 |

## 2. EP均衡：降低专家并行拖尾

**说明。** EP是Expert Parallelism，专家分布在不同rank上。全局专家平均接收token相近，不代表每个micro-batch里各rank的工作量相近。Step 3.5在全局loss-free均衡之外加入EP-group级训练损失，使router关注rank利用率。这里讨论训练塑造的路由倾向；服务期EPLB通过专家副本/映射调整应对实时热点，是另一层机制。

**接口流程：** `hidden states → router分数/Top-k → 按专家所在rank分发token → 各rank计算 → combine返回`；训练时从路由概率及分配统计计算group级均衡损失，反馈到router。简化关键路径为`T_MoE ≈ T_dispatch + max_r(T_compute,r) + T_combine`，最忙rank决定计算等待。

**竞争力评价：有条件的系统价值，独立领先证据不足。** 在偏斜明显的MoE服务中值得关注；热点不明显时收益有限。过强均衡可能限制专业化，训练时group与实际部署映射也可能不同。不能由训练损失推出线上吞吐固定提升；预训练约束的使用阶段亦须和mid-training/SFT区分。

**参考：** [Step 3.5 §2.2 EP均衡公式及§4训练设置](https://arxiv.org/html/2602.10604v2#S2.SS2)；[vLLM专家并行/EPLB指南](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)；[Ascend EPLB指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/expert_parallelism_load_balancer.html)。公开模型router可解释推理路由，不能作为训练损失完整实现证据。

**验证重点：** 每rank token/专家计算负载、最忙rank与平均rank之比、dispatch/combine耗时、P99每token延迟和同条件吞吐；EPLB另计专家复制与迁移成本。

## 3. Gated SWA：控制局部注意力的信息注入

**说明。** SWA只访问滑动窗口，降低历史访问成本；head-wise gate根据当前输入缩放各head输出。窗口没有有用证据时，softmax仍分配权重，gate可抑制该head的信息注入。Step 3.5结合窗口512、3:1的SWA/Full混合布局及增加SWA query heads；门控不是扩大窗口或稀疏位置选择器。

**接口流程：** `输入x → Q/K/V → 窗口内因果attention得到y → g=sigmoid(W_gate·x) → 按head执行g×y → 输出投影/残差`。Full层补充远程连接；SWA保留attention语义，可用KV masking支持投机候选并行验证。实际KV容量收益仍取决于缓存后端是否回收窗口外状态。

**竞争力评价：较强的成本—质量设计，有局部消融支撑。** 论文在100B-A10B、相同S3F1布局的预训练对照中，head-wise gate相对固定sink token使平均分由62.46到64.43，增加1.97分。这支持该条件下的门控质量收益；混合布局的计算节省不能全部归因于gate，也不证明胜过所有注意力架构。

**参考：** [Step 3.5 §2.3表2及附录A.1/A.2](https://arxiv.org/html/2602.10604v2#S2.SS3)；[公开模型实现](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/modeling_step3p7.py)。阅读代码应区分attention输出门控、FFN门控和MoE router，三者职责不同。

**验证重点：** 同质量要求下的长上下文任务表现、Prefill时间、Decode延迟和实际KV占用；消融时分别控制窗口、head数、门控和MTP。

**Gate论文证据补充。** Gated Attention Table 1在15B总参数、约2.54B激活的MoE训练400B tokens后，无Gate与head-wise输出Gate对照的平均PPL为6.026→5.792，MMLU为58.79→60.05；增加约160万参数。该实验为GQA，不能作为Step普通SWA对比Gated SWA的实测结论。其价值是输入相关的输出抑制与非线性；已有Ascend Step patch可复用，专项融合优化应以profiling为依据。详见[推理特性详解中的Gated SWA章节](06A-推理特性详解与竞争力评估.md#3-gated-swa控制局部注意力的信息注入)及[论文Table 1](https://papers.neurips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf)。

## 4. AFD：Attention与FFN分池执行

**说明。** AFD是Attention-FFN Disaggregation。Attention池持有请求/KV，FFN池执行dense FFN或MoE；跨多个Attention实例汇聚兼容的FFN工作，可提高专家有效batch，并分别配置两侧设备与并行拓扑。它与Prefill/Decode分离、DBO计算通信重叠解决的问题不同，可以按后端条件组合。

**接口流程：** `Attention计算 → PushPull发送激活/元数据 → FFN GetBatch → 应用执行FFN → Respond → Attention Wait后进入后续层`。传输主要是中间激活，不是每层搬迁整份KV。同一微批次的层间依赖不消失；重叠来自其他独立微批次。StepMesh负责交接通信，不负责执行模型FFN。

**竞争力评价：高并发和资源失配场景有潜力，部署复杂度较高。** Step-3系统论文报告Hopper、4K上下文、FP8、无MTP、50ms TPOT约束下4039 tok/s/GPU，对照DeepSeek-V3为2324；这是不同模型及系统的整体对照，不能当作纯AFD收益。低并发、单请求延迟优先时，额外往返和聚批等待可能得不偿失。

**参考：** [Step-3 §3与§7](https://arxiv.org/html/2507.19427#S7)；[StepMesh接口](https://github.com/stepfun-ai/StepMesh)及[运行示例](https://github.com/stepfun-ai/StepMesh/blob/main/tests/fserver/test_fserver.py)；[AFD Attention运行时设计](https://github.com/vllm-project/afd-plugin/blob/bf5bd280b7481358fc0df4a8ec5897b12be73802/docs/design/module/attention_runtime.md)。StepMesh的xPU占位不证明完整NPU后端已实现。

**验证重点：** 相同总设备数、精度、上下文与延迟约束下的吞吐/成本；记录A/F资源比例、专家batch、队列等待、通信及KV容量。昇腾优先核对已有afd-plugin路径，再判断Step模型接入缺口。

## 5. PaCoRe：并行探索与多轮推理综合

**说明。** PaCoRe是Parallel Coordinated Reasoning，通过多条探索轨迹和综合模型跨轮协调，扩展推理时计算；压缩消息让后续轮次在有限上下文内利用前轮成果。价值来自探索和综合能力，不只是多次采样投票。

**接口流程：** `问题/前轮消息 → 并行调用K条轨迹 → 压缩/综合 → 综合消息反馈下一轮 → 最终答案`。代码`_run_parallel_calls()`创建异步任务，`_get_semaphore()`限制并发，`process_single()`组织处理。框架能接任意兼容模型，不保证模型具有专门训练的综合能力。

**竞争力评价：高预算难题具有竞争力，低成本优势不能直接成立。** 官方报告PaCoRe-8B在HMMT 2025达到94.5%，其有效推理时计算约为两百万token。它证明特定预算下可提高质量，不是与单轨迹模型等预算的全面胜出。适合数学、代码等可验证任务；外部工具有副作用时还需去重、取消和结果协调，不能直接套用纯文本并行流程。

**参考：** [官方说明与报告入口](https://github.com/stepfun-ai/PaCoRe)；[base_exp.py](https://github.com/stepfun-ai/PaCoRe/blob/main/pacore/batch_inference/base_exp.py)。

**验证重点：** 固定任务和总预算，对照单轨迹、best-of-N/投票与多轮综合；统计成功率、总token、完成时间、失败轨迹及综合丢失的信息。

## 6. MTP-3：以三个预测模块辅助投机解码

**说明。** MTP是Multi-Token Prediction；“-3”指三个附加预测模块，不是固定3倍加速。Step 3.5的轻量模块采用SWA+Dense FFN，训练主要先优化第一个，后期复制并联合训练其余模块。主LM预测下一token，MTP模块提供更远位置候选；候选仍须由目标模型验证。

**接口流程：** `主模型状态 → MTP提出候选 → 主模型批量验证 → 按验证规则接受连续候选/处理拒绝 → 提交有效KV → 下一轮`。平均接受长度、候选成本与验证成本共同决定收益；严格分布保持还要求正确的接受/校正规则，不能只因使用MTP就声称无损。

**竞争力评价：较强的解码优化，但收益依赖后端与负载。** 长生成、带宽受限Decode且接受率高时更有利。Audio3论文表6在自身计时条件下报告MTP3严格验证1.76倍、Medusa式接受2.05倍；后者允许生成分布变化，且不是Step 3.5或昇腾速度。更多预测头也未必更快。模型结构支持与后端完整MTP-3支持应分别核实。

**参考：** [Step 3.5 MTP架构](https://arxiv.org/html/2602.10604v2#S2.SS2)；[Ascend proposer固定源码](https://github.com/vllm-project/vllm-ascend/blob/c076a131097fdfac9db4b57c8cf55bfd45ea0ca5/vllm_ascend/spec_decode/step3p5.py)；[Audio3 §6.3.2表6](https://arxiv.org/html/2609.14005#S6.SS3.SSS2)。

**验证重点：** 平均接受候选数、拒绝率、每有效输出token的时间、吞吐与质量；区分严格验证/宽松接受、并发、采样参数、生成长度和KV提交成本。

## 7. CLM：模型维护工作上下文

**说明。** 此处CLM指Context Language Models思路在Step-Code中的应用。保留原始会话，同时让模型选择性缩短供后续决策使用的工作视图；它与全历史摘要压缩是不同路径，不是给模型增加永久记忆。

**接口流程：** `检查上下文压力 → 构建可编辑旧正文索引 → 模型返回replacements[{id,text}] → 宿主校验来源/版本/保护内容/节省量 → 接受新工作视图 → 后续模型调用`。不合法编辑被拒绝；符合回退条件时使用原生压缩。新输入、取消或上下文变化还需处理维护中止。

**竞争力评价：长日志任务具有工程价值，单项领先性待验证。** 编译/调优日志可缩短为失败原因、修复位置、当前结果与完整日志入口，使后续多轮更聚焦。代价是维护调用、可能的语义遗漏及前缀缓存复用下降。来源文档明确借鉴pi-clm，不能视为Step独有创新；结构校验也不等于语义无损。

**参考：** [Step-Code上下文管理及校验契约](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/docs/context-management.md)；[pi-clm固定版本](https://github.com/lolipopshock/pi-clm/tree/b84a9d7cbb625cd39539db3bcef72ea9cc89aa89)；[原生压缩实现（用于区分）](https://github.com/stepfun-ai/Step-Code/blob/main/packages/agent-core/src/harness/compaction/compaction.ts)。

**验证重点：** 固定模型/任务/预算，对照原生摘要与CLM；统计成功率、关键约束遗漏、恢复表现和包含维护调用及缓存变化的总成本。

## 8. Think-While-Speaking：推理与表达并发

**说明。** 规范名称为Think-While-Speaking（TWS）。Audio3 Realtime以同一音频模型的两个并发调用承担Formulation Brain与Articulation Brain：前者推进私有推理，后者依据当前可用推理和已说内容生成短响应段，并按播放进度释放。默认Speak-First无需等待初始推理前缀；Think-First先等待短前缀。

**接口流程：** `用户音频/历史 → 推理分支持续思考 ∥ 表达分支分段生成 → 播放进度调度 → 新推理供下一段使用 → 完成后补充/纠正`。Adaptive Thinking决定是否进入显式推理，MTP加速私有推理，两者与TWS分别归因。它也不同于全双工听说和异步工具执行。

**竞争力评价：实时语音体验有条件竞争力，正确性与最终完成时间仍需验证。** 能缩短开始表达的等待，但先说出的内容可能需要纠正；两个并发调用亦有额外资源成本。Audio3整系统分数不能证明TWS单项领先，MTP表6的加速也不能算作TWS收益。播放停止不等于工具任务取消，需求变化须丢弃或重新验证旧结果。

**参考：** [Audio3 Realtime §6.3](https://arxiv.org/html/2609.14005#S6.SS3)；[双过程来源Mind-Paced Speaking](https://arxiv.org/abs/2510.09592)；[Realtime Console](https://github.com/stepfun-ai/Step-Realtime-Console)仅作采集、WebSocket和播放接口参考，不证明Audio3完整TWS已开源或已在NPU运行。

**验证重点：** 同质量条件下首音时间、首个有效回答时间、最终正确结论时间、提前错误承诺/纠正率、打断恢复和总费用；填充话术提前不能替代有效回答提前。

## 9. 对当前工作的投入判断

**推理侧优先测MTP净收益与EP拖尾，再按长上下文负载分析Gated SWA；AFD在确认KV容量、专家batch或A/F资源失配后专项验证。Agent侧先将CLM做成可消融的上下文策略，再用PaCoRe评估复杂可验证任务；TWS在明确实时语音业务后投入。** 这些特性改善的是不同指标，应分别消融，再评估组合；论文中的GPU、模型、任务与接受规则不能直接转换成昇腾收益。
