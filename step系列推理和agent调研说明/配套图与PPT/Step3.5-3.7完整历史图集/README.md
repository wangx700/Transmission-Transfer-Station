# Step 3.5 / 3.7 独立报告与全量图集

整理日期：2026-10-12，北京时间。保留此前内容，另存独立报告和图集；原 Step 5 报告、图示及旧版原稿均未覆盖或删除。

## 推荐阅读顺序

- `Step3.5与3.7推理与Agent技术报告.md`：3.5/3.7 独立技术报告，含结构、优势、代码流程、EP、后训练和服务端点附录。
- `Step3.5-3.7-全量图解-原版保留.pptx`：六页完整 PPT，原样复制，适合逐项讲解。
- `Step3.5-3.7-单页总览-原版保留.pptx`：原单页总览，适合汇总展示。

## 图集：原版高清逐页图

1. `图集/01-模型结构.png`：参数、语言主干、视觉接入、Attention/MoE 与 MTP。
2. `图集/02-SWA与GatedSWA.png`：Full/SWA 可见区域与逐 head 输出门控。
3. `图集/03-稀疏MoE.png`：逻辑专家选择、共享专家与加权输出。
4. `图集/04-EP负载均衡.png`：训练均衡、服务端重排/副本概念和 rank 拖尾。
5. `图集/05-PaCoRe.png`：多轨迹探索、结论提取与综合；属于独立方法。
6. `图集/06-Agent组件.png`：Skills、Plugin/MCP、子会话、Goal、交付组件。
7. `图集/07-单页总览.png`：此前汇总图，原样保留。

`机制原图/` 另含 SWA 矩阵、Gate 局部图、完整 SWA/Gate 示意、MoE 示意与 PaCoRe 示意，适合裁取引用。

## 流程图：可编辑源文件与 SVG/PNG

每个编号提供 `.mmd`、`.svg`、`.png` 三种形式：

1. `流程源文件/01-flow.*`：模型内部计算，包含 3.7 视觉接入。
2. `流程源文件/02-flow.*`：SWA/Gate 的输出侧门控流程。
3. `流程源文件/03-flow.*`：MoE 的概率、bias、top-k、dispatch/combine。
4. `流程源文件/04-flow.*`：StepTronOSS 异步 RL 的版本、并发、积压与权重同步。
5. `流程源文件/05-flow.*`：Skills、工具执行与业务验收。
6. `流程源文件/06-flow.*`：PaCoRe 的推理和训练流程。

## 交互图与历史版本

`交互图/` 保留原片段及可在浏览器打开的 `*-交互预览.html`：SWA/Gate、MoE 路由、PaCoRe。预览为离线演示，使用本地模拟数据，不调用模型。原始交互参数均为缩小教学示例。

`历史版本/` 保留之前各版 PPT、PNG 和旧版完整机制背景报告，可能含当时 Step 5 的范围说明。其旧版 Step 5 备注不能替代现在的 Step 5 重点报告。

## 证据与范围

- 模型结构对应 3.5/3.7；MTP 为生成加速分支，图中简化箭头不表示它是普通主干层。
- PaCoRe 指标来自独立论文对照，不是 3.5/3.7 的同条件成绩。
- EP 示例和 flowcontroller 图不证明其为模型的完整生产内部实现。
- Step-Code 是相关 harness，后来的 Step 5 框架测试分数不移植到 3.5/3.7。
- 文档表注包含日期、单位、模型/硬件/负载范围；所有技术证据均使用网络链接。

原始参考：[3.5 技术报告](https://arxiv.org/html/2602.10604v2)、[3.7 官方仓库](https://github.com/stepfun-ai/Step-3.7-Flash)、[3.7 模型实现](https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/modeling_step3p7.py)、[PaCoRe](https://github.com/stepfun-ai/PaCoRe)、[StepTronOSS](https://github.com/stepfun-ai/SteptronOss)、[Step-Code](https://github.com/stepfun-ai/Step-Code)。
