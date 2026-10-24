# 模型推理与Agent技术资料

## Step 5、Step-Code 与 StepAudio 3 阶段更新

- [2026-10-05 至 10-23 阶段更新（统一版，四块式）](step系列推理和agent调研说明/更新记录/2026-10-23/Step5-StepCode-StepAudio3更新与技术说明-2026-10-23.md)：**当前阅读入口**。按四块组织，前两块以 **RL 场景推理侧** 与 **Agent** 两条主线展开：**一 社区更新点**（1.1 状态复核；1.2 围绕 Step 5 给出 `step-5-preview-tracking` 的推理引擎改动——先列 Step5 结构要点，再给 5 条改动并标注各自落在 rollout 哪一环；1.3 Step-Code 全部 54 个合入 PR 单表（时间 / PR / 修改内容 / 昇腾适配 / 说明），10-05→10-23 不分段；1.4 非主线口径）；生态线（batch 一致性 / DSA 索引回放 / LoRA）不在本文记录，见各跟踪目录；**二 昇腾需要关注的点**（2.1 五条判断：SM90 不可外推、行号口径、cann-ops churn、sidecar×KV 互斥、MC2 超时；2.2 可用/部分可用/不可用；2.3 迁移矩阵；2.4 P0–P2 待办；2.5 自有 fork 增量；2.6 RL 场景专属关注点：确定性、logprob 一致性、batch 不变性、replay 闭环、权重更新）；**三 当前任务项进展**、**四 人力投入矩阵与日常活动**：本期暂空，待补。附 [路径映射表](step系列推理和agent调研说明/更新记录/2026-10-23/本项目路径映射-20261023.tsv)、10-20 MC2 dispatch 超时故障摘录与 6 条口径订正。
- [2026-10-05 至 10-20 更新与技术说明](step系列推理和agent调研说明/更新记录/2026-10-20/Step5-StepCode-StepAudio3更新与技术说明.md)：当日归档（10-23 上传），内容已并入上方统一版，保留原发布结论以备追溯；同目录含[提交清单](step系列推理和agent调研说明/更新记录/2026-10-20/Step-Code-commits-20261005-20261020.tsv)与[关键提交 patch 证据](step系列推理和agent调研说明/更新记录/2026-10-20/vllm-step5-e177c621-commit.json)。


## RL Replay 与训练推理一致性

- [六项特性汇总与阅读入口](rl-replay-consistency/README.md)：Sampling Mask、Index、R3、Hidden States、Logprob、Batch 一致性的当前状态、接口与数据格式。
- [各类 Replay 可视化流程图](replay-flowcharts/README.md)：总览 + M/I/R/H/L/T/B/S 八张流程图（PNG/SVG/Mermaid 源），含组合约束与一致性诊断闭环。
- [特性流程与代码证据](rl-replay-consistency/特性流程说明.md)、[上一周合入 PR 与流程映射](rl-replay-consistency/上一周PR与流程映射.md)：状态核对于2026-10-23；上一周明确指北京时间2026-10-12至10-18。
- [vLLM / vllm-ascend Batch 一致性周报](batch-consistency-weekly/README.md)：2026-10-16 → 10-23 合入 PR 说明、当前 batch 一致性状态与新增跟踪项；含整体报告（刷新至 2026-10-23 HEAD）与 [PR diff 证据](batch-consistency-weekly/evidence/README.md)。

## DSA Index Replay 上游跟踪

- [当前整体支持与适配报告](dsa-index-replay/DSA_index_replay_status.md)：vLLM、vLLM-Ascend、SGLang 索引采集返回能力、限制与代码证据，状态截至 2026-10-23。
- [10-16 至 10-23 阶段变化说明](dsa-index-replay/DSA_index_replay_changes_2026-10-16_to_2026-10-23.md)：vLLM 新增 #62915 Profile-based indexer replay Draft 等本期合入 PR 与新增跟踪项；[核查证据](dsa-index-replay/evidence/README.md) 附 PR 状态与合入 SHA 清单。

**优先阅读：[技术精简版](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/%E6%8A%80%E6%9C%AF%E7%B2%BE%E7%AE%80%E7%89%88.md)。** 集中保留结构、机制、接口与代码；详细专题按需展开。


面向模型架构、推理系统、Agent执行与昇腾适配的技术调研。资料时点及证据等级见各文档；模型与 Agent 调研部分未进行 NPU 真机复现，Sparse HCCL 材料的真机验证范围见独立目录。

## 核心报告

- [核心目录与本次刷新说明](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/README.md)

- [00-核心总览](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/00-%E6%A0%B8%E5%BF%83%E6%80%BB%E8%A7%88.md)
- [01-推理技术核心分析](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/01-%E6%8E%A8%E7%90%86%E6%8A%80%E6%9C%AF%E6%A0%B8%E5%BF%83%E5%88%86%E6%9E%90.md)
- [02-Agent技术核心分析](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/02-Agent%E6%8A%80%E6%9C%AF%E6%A0%B8%E5%BF%83%E5%88%86%E6%9E%90.md)
- [03-昇腾适配与验证规范](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/03-%E6%98%87%E8%85%BE%E9%80%82%E9%85%8D%E4%B8%8E%E9%AA%8C%E8%AF%81%E8%A7%84%E8%8C%83.md)
- [04-重点对象代码参考与工程投入](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/04-%E9%87%8D%E7%82%B9%E5%AF%B9%E8%B1%A1%E4%BB%A3%E7%A0%81%E5%8F%82%E8%80%83%E4%B8%8E%E5%B7%A5%E7%A8%8B%E6%8A%95%E5%85%A5.md)

## 详细资料与图集

- [完整目录](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/README.md)

核心报告作为当前阅读入口；详细报告保留参数、原始案例、接口与证据索引，历史图/PPT用于备查。

## Sparse HCCL 权重更新

- [PR #18915 当前版本时序图与接口说明（2026-10-24）](vllm-ascend-pr18915-sparse-hccl/04-current-sequence-20261024.md)：固定 `f6ad8bd5`，含独立 Mermaid 源文件及当前支持边界；旧版材料另行保留。
- [VIME #492 优化候选与 #480 已验证流程](vime-sparse-hccl-492-480/README.md)：当前断点、优化时序、代码补丁、96项CPU回归、真实单NPU loader检查，以及188容器与历史性能口径。

- [PR #18915 材料目录](vllm-ascend-pr18915-sparse-hccl/README.md)：整体 UML 时序图、核心代码与接口流程、80 个 pytest 用例说明，以及真实 HCCL 验证记录。

## Qwen3-30B-A3B 稀疏权重同步

- [Megatron → HF → 推理侧权重分片与 slot 六元组](qwen3-30b-megatron-hf-sparse-sync/README.md)：TP2/PP2/EP2 下的 QKV、MoE name/shape、训练与推理并行差异、NaN probe 操作示例、可运行教学脚本及代码证据。

## MoE 专家计算说明

- [gate/up/down 计算与专家权重命名](MoE专家计算说明/README.md)：Qwen3-30B-A3B 的 SwiGLU、fc1/fc2融合、ETP2计算分解、router与专家gate的区别，以及weight6的本地编号含义。
