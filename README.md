# 模型推理与Agent技术资料

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

- [PR #18915 材料目录](vllm-ascend-pr18915-sparse-hccl/README.md)：整体 UML 时序图、核心代码与接口流程、80 个 pytest 用例说明，以及真实 HCCL 验证记录。

## Qwen3-30B-A3B 稀疏权重同步

- [Megatron → HF → 推理侧权重分片与 slot 六元组](qwen3-30b-megatron-hf-sparse-sync/README.md)：TP2/PP2/EP2 下的 QKV、MoE name/shape、训练与推理并行差异、NaN probe 操作示例、可运行教学脚本及代码证据。
