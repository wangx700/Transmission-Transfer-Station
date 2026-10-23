# 核查证据与复查方法

核对日期：2026-10-23（北京时间）。主线 SHA：`432b1003f8ea7c871750f27bc82815a7ad0252d5`；当前结论固定到该源码，不用未合入分支证明主线能力。

- `merged-prs.json`：18项相关 PR 的标题、URL、`merged_at`（UTC）、北京时间、合入 SHA、流程编号及合入文件列表。仅提取公开技术元数据。
- `pending-prs.json`：5项未合入工作的公开状态摘录。`merged_at=null` 不等于已完成。
- 来源为 `https://api.github.com/repos/vllm-project/vllm/pulls/{number}`，文件列表通过对应合入 SHA 的 `git show --name-only` 核查；正文对关键改动同时阅读 diff。

上一周 UTC 判据：`merged_at >= 2026-10-11T16:00:00Z && merged_at < 2026-10-18T16:00:00Z`，转为北京时间即10-12至10-18完整自然周。检索范围为本次六项特性及明确相邻的采样正确性/版本归属/文档/CI，不是全仓所有合入。

源码复查示例：

```bash
git show a8be1500bad93d3c459e435bf71194aa8062e49d -- vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
git show 03e8300d2cac77b24f7ab6013a803369ee64c19e -- vllm/model_executor/determinism/batch_invariant.py
git show 9e2b579aabe9f289791867de539f999c188b63cd -- vllm/model_executor/models/granitemoe.py
```

这些是复查命令，不是模型测试。未执行 GPU/NPU 数值、组合或性能验证。详细数据格式附录保留10-21固定源码及示例，主报告状态已在10-23重核；两种日期不能混用。
