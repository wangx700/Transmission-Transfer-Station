# Step 系列图片与 PPT 配套说明

**整理日期：2026-10-12，北京时间。**本文件是本次报告配图的统一讲解和使用说明。配套目录保存此前图/PPT的原样副本，未将旧图自动改写成Step5架构。正式陈述以本次各专题正文的证据范围为准；历史图中的动态速度、排名和状态不作为新的测量结果。

文件名用于交付定位，不作为技术证据链接；下文参考全部为网络地址。`附件清单.json`列出实际文件、大小及SHA-256，方便核对。历史目录中的旧报告是备查材料，当前阅读入口为本包根目录的00～06文档。

## 1. Step 5 Preview：一页讲三个重点

表注：以下均在`配套图与PPT/Step5/`；PPT和PNG为既有单页重点版，交互HTML为历史片段源，不保证脱离原渲染宿主即可独立运行。当前文本口径以本次报告为准。

| 文件 | 对应文档 | 用途与讲法 |
|---|---|---|
| Step5Preview-稀疏注意力长程任务与StepCode-单页重点版.pptx | 01、03、04 | 可编辑单页，按模型规模→稀疏注意力→长程反馈→Step-Code讲解 |
| Step5Preview-单页重点版.png | 同上 | 适合直接粘贴；无法像PPT一样修改图元 |
| Sparse块合并成本-历史交互片段.html | 03原理说明 | 解释候选重叠和局部成本；作为交互源保留，不作为Step5已公开kernel证明 |

推荐口播：Step5扩大总容量，以MoE控制FFN激活计算，并通过报道中的Sparse GQA控制历史访问；长程任务依赖工具反馈和上下文更新；Step-Code提供可持续执行的宿主机制。600B/27B与API输入输出为官方规格；92层、块合并与RL细节标为媒体技术转述。[官方发布](https://www.stepfun.com/step-5-preview)、[技术转述](https://news.qq.com/rain/a/20260920A06EUP00)

**图片中的边界必须同时讲清：**

- H1/H3等是教学历史块，可代表一组token的K/V，不是已核实的Step5块大小、索引粒度或物理cache page。
- 相邻query来自prefill时多个输入位置；共同K/V可由融合调度在片上复用。K/V投影本来就不为每个query重算，节省的是后续重复读取机会。
- 普通单token decode每请求每轮只有一个新query；多query复用不能直接套用。推测验证和chunked prefill块内可讨论类似条件。
- 索引结果去重不自动减少已经发生的Indexer计算。局部1/8来自报道，具体评分共享或合并实现未知。
- 旧图的合并箭头是概念示意，不表示必定“Top-k后做并集”。不得把允许集合扩大而不保留query mask后仍称数学等价。
- AA分数是指数；历史PNG中的速度为当时API观测。本轮AA输出速度为89.2 tokens/s、指数44；无需据此修改所有历史图。[AA模型页](https://artificialanalysis.ai/models/step-5)

## 2. Step 3.5 / 3.7：详细机制图与总览

表注：主PPT位于`配套图与PPT/Step3.5-3.7完整历史图集/`，逐页PNG位于其`图集/`。以下顺序对应六页全量图解及单页总览；示意中的token、专家缩小数量不代替真实配置。PaCoRe和Agent不属于Transformer主干层。

| 资料 | 主要内容 | 推荐讲解与限定 |
|---|---|---|
| Step3.5-3.7-全量图解-原版保留.pptx | 六页机制详解 | 配合03文档展开，适合技术分享 |
| Step3.5-3.7-单页总览-原版保留.pptx | 一页架构和机制概览 | 用于开场；MTP简化箭头不表示普通主干串行层 |
| 图集/01-模型结构.png | 45层语言主干、混合attention、MoE | 3.7视觉分支单独进入语言特征；旧代参数不得填到Step5 |
| 图集/02-SWA与GatedSWA.png | 窗口和输出gate | SWA限制看哪里；gate调节每head输出注入多少；不会自动跳过attention |
| 图集/03-稀疏MoE.png | routed/shared专家与合并 | 真配置288选8、另1shared；router bias影响选择，权重来自原概率 |
| 图集/04-EP负载均衡.png | rank热点与拖尾 | 训练均衡、服务EPLB和请求调度是不同层次；均衡需权衡通信开销 |
| 图集/05-PaCoRe.png | 并行轨迹、结论提取、多轮综合 | 独立推理方法，以更多总计算提高难题质量，非Step默认路径 |
| 图集/06-Agent组件.png | Skill/Plugin/工具/执行与验收 | 框架能力不能直接归因于3.5/3.7模型结构 |
| 图集/07-单页总览.png | 单页图的图片形式 | 适合粘贴；细节以机制图和报告说明为准 |

**结构图推荐顺序：**先讲“attention访问历史、FFN处理当前位置表示”，再讲混合SWA/Full、输出gate、MoE路由与shared expert，最后讲MTP。EP是专家的跨设备执行安排，不是另一种注意力。[3.5论文](https://arxiv.org/html/2602.10604v2)、[3.7配置与实现](https://huggingface.co/stepfun-ai/Step-3.7-Flash/tree/main)

**EP图推荐解释：**`router → dispatch → 专家计算 → combine`，最慢rank影响完成时间。3.5训练中的全局bias与EP-group约束提供均衡倾向；服务EPLB通过等价专家的物理映射/副本缓解热点，但不应改变逻辑专家语义，也不是已证明Step5线上采用的实现。[vLLM EP说明](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)

**PaCoRe图推荐解释：**多条探索轨迹提取最终结论，送下一轮综合并收敛；论文的RLVR-8B对照不等于3.5/3.7自身收益。75.4→94.5与48K→1796K应成对展示，不能只放分数而隐藏计算预算。[PaCoRe论文](https://arxiv.org/html/2601.05593v1)

## 3. 其他原图与流程源文件如何使用

`机制原图/`保留SWA矩阵、Gated SWA、稀疏MoE与PaCoRe图片；`交互图/`保留HTML源片段和既有预览；`历史版本/`保留此前全部PPT/PNG。优先使用上表主PPT与逐页图，历史版本用于回看，不混用不同版本口径。

表注：下表为`Step3.5-3.7完整历史图集/流程源文件/`的源文件序列；每项同时有MMD与SVG。编号对应此前旧代技术报告的Mermaid顺序，不代表模型执行层号。

| 文件前缀 | 流程内容 | 适合配合的说明 |
|---|---|---|
| 01-flow | 语言主干与3.7视觉入口 | 03模型结构 |
| 02-flow | SWA与输出gate | 03注意力机制 |
| 03-flow | MoE路由、shared expert与EP | 03路由语义与EP |
| 04-flow | 异步rollout、版本与背压 | 02后训练设施；是StepTronOSS控制器示意 |
| 05-flow | Skill发现、工具执行与验收 | 04 Agent分层；业务checker需另行实现 |
| 06-flow | PaCoRe并行探索与综合 | 03附录B；独立方法背景 |

SVG适合放大，MMD便于改图；Markdown中的Mermaid是可编辑文本源。交互图的演示参数用于观察机制，不视为生产部署参数。公开接口证据分别见[StepTronOSS流控](https://github.com/stepfun-ai/SteptronOss/blob/dev/steptronoss/core/generators/flow_controller.py)、[Step-Code技能加载](https://github.com/stepfun-ai/Step-Code/blob/main/packages/coding-agent/src/core/skills.ts)。

## 4. 昇腾图：部署判定与Agent改进落点

表注：以下位于`配套图与PPT/昇腾流程图/`，每项均提供MMD/SVG/PNG。内容为已核验状态和工程建议，未真机测试；需区分实线现状与虚线建议。

| 文件前缀 | 内容 | 讲解要点 |
|---|---|---|
| 01-ascend-flow | 模型与后端适配判定 | 权重/config→registry/算子→环境配套→功能/精度/性能；API可用不等于本地NPU可用 |
| 02-ascend-flow | MindSpeed-Agent优化角色与建议能力 | 实线为已有优化流程，虚线为证据卡、资源lease、上下文/预算建议 |

Step5目前未确认公开主线开箱支持；3.5/3.7已有A3/950合入记录，但Runner、视觉、MTP和量化逐项验证。[支持PR](https://github.com/vllm-project/vllm-ascend/pull/10556)、[MRV2任务](https://github.com/vllm-project/vllm-ascend/issues/14205)

MindSpeed-Agent图只对应inference-model-optimize状态机，不强制另一套prompt-first migration插件使用相同流程；CLM、取消与设备租约需要宿主执行支持。[官方优化流程](https://gitcode.com/Ascend/MindSpeed-Agent/blob/f56bb671309ee1bd9f9fdf6c6c7b17b488f4a448/plugins/inference-model-optimize/workflows/optimize-pipeline.yaml)

## 5. 正式汇报的取图建议

面向Step5：用Step5单页讲能力和三条技术主线，配昇腾判定图说明本地部署边界；Step-Code机制从04专题单独讲解。面向本地推理适配：用3.5/3.7模型结构、MoE/EP和MTP说明建立实际基线，再讨论Step5需补的算子。需要深入时再加SWA/Gate、异步RL与PaCoRe。

图中短句应保留“官方/报道/工程示意”来源级别和最必要的范围。所有倍数、分数必须随同条件说明；无法在一页容纳的参数和服务端点明细放报告附录，不用缩小字号把完整正文塞进PPT。
