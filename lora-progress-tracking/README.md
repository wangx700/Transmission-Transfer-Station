# 非量化 LoRA 进度跟踪（SGLang / vLLM / vLLM-Ascend）

跟踪三个上游仓库非量化 LoRA 推理支持的阶段变化、合入 PR 与新增跟踪项。每期说明回答三个问题：间隔期间哪些功能发生变化、对应 PR 修改了什么、现在应新增跟踪哪些 PR。

## 本期（2026-10-16 → 2026-10-23）要点

- **SGLang 执行栈路线更替**：21 部分堆叠 LoRA runtime stack 以单 PR [sglang#45892](https://github.com/sgl-project/sglang/pull/45892) 于 10-23 合入 main（plan-driven dense/MoE 引擎、CuTeDSL SM90/SM100 BF16/FP8 与 Marlin NVFP4 provider、dense MLA correction、ascend_backend 更新）；原堆叠系列 #42875/#42882/#42884 已 Closed、#42284 仍 Draft，跟踪入口切换到 #45892 及后续 follow-up。
- **vLLM**：Step5（文本+VL）LoRA 合入（#62498）；BERT 扩展到 token classification（#61391）与 cross-encoder（#61279）；生命周期与调度集中修复（sleep 恢复 #62358、失败激活可重试 #61618、config 拒绝 #61884、max_loras #60306、AsyncLLM 重载 #41482、多模态默认 LoRA #62906、文本 LoRA×多模态 wrapper #62224）；上期跟踪 #61284（fully-sharded LoRA 禁用 shared-expert aux stream）合入。
- **vLLM-Ascend**：本周零 LoRA 合入、零新 LoRA PR；replay/DSA 底座扩展（#18936、#19321、#19190）不改变 LoRA 组合限制——replay 遇 LoRA 关闭、PCP LoRA NotImplementedError、dynamic EPLB/FusedMC2 断言在 10-23 main 原样保留。
- **新跟踪重点**：SGLang #45231（DSV4/V4.1 attention + shared expert，与 vLLM #57245 平行）、vLLM #62049（lm_head prompt logprobs）、#63077（idle DP + MoE slot）、#62034（attention_k_eq_v packed V slice）、#62019（router weight on input）、SGLang #44704（GDN adapters）。

## 文件

| 文件 | 说明 |
|---|---|
| [LoRA阶段进展与PR跟踪_2026-10-16至2026-10-23.md](./LoRA阶段进展与PR跟踪_2026-10-16至2026-10-23.md) | 本期阶段进展与 PR 跟踪说明（主文档） |
| [SGLang_vLLM_Ascend_LoRA支持与工作报告.md](./SGLang_vLLM_Ascend_LoRA支持与工作报告.md) | 整体支持与工作报告（10-16 基线，主文档引用） |
| [LoRA阶段进展与PR跟踪_2026-10-10至2026-10-16.md](./LoRA阶段进展与PR跟踪_2026-10-10至2026-10-16.md) | 上一期阶段进展（10-10 → 10-16） |
| [evidence/merged-prs-2026-10-16_to_10-23.json](./evidence/merged-prs-2026-10-16_to_10-23.json) | 本期合入提交与上期跟踪 PR 状态的结构化证据 |

## 方法与边界

比较起点为上期报告固定 HEAD（vLLM `dd9e07a25`、SGLang `4e85e8ed`、vLLM-Ascend `c5766cd5`），终点为 2026-10-23 12:03 fetch 的 origin/main（vLLM `19c0e170d2`、SGLang `f4a2df3dac`、vLLM-Ascend `db8836752`）。合入 PR 用 git 提交扫描（关键词 lora/adapter/punica/peft；Ascend 另按 `vllm_ascend/lora` 与 `vllm_ascend/models/deepseek_v41` 路径），PR 状态用 GitHub 页面核验（OPEN/MERGED/CLOSED/DRAFT）。未运行模型测试；已合入其他分支不算 main 已支持；CI-only 提交不计为功能；diffusion LoRA 不纳入主体结论。
