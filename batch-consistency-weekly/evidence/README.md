# 核查方法与证据

## DeepInfer V4.1 固定版本源码补充

2026-10-23（北京时间）整理了[专题报告](../deepinfer_v41_batch_invariance.md)。分析 DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b` 及锁定子模块：DeepGEMM `1fc9c66975ee322d628b3288534ca0e9ce12a64d`、FlashMLA `2e5429fc5653bab6e081f09477126f731882a6a9`、DeepEP `93eb6eb238127e96c6d7a4a625a6dad158348509`、DeepGEMM-Ascend `3a35326271b069b8b5bf1a754435acd6cc91575b`、DeepEP-Ascend `3b25377d04b24fc6154698ded78a2bcb2c59afff`。专题中提供固定 SHA 的源码链接，便于复核。

核查路径：Worker 开关 → Decoder/Attention/Indexer/MoE 调用链 → 子模块确定性分支 → 测试断言。明确区分 HC prenorm 已有适配、dispatch 排序可选能力、combine 参数冲突和其余待测项。模型测试的 golden token 用于固定历史，验收仍要比较真实 logprob；算子测试代码存在不表示本次运行通过。未做硬件实验，不将条件冲突外推为全部 NPU 算子失败，也不把发布实现计作本周新增改动。

核对时点：2026-10-23 12:00 前后（北京时间）。

## 方法

1. `git fetch origin main` 快进 vllm / vllm-ascend / sglang / verl 四仓库，基线为上一期报告保存的 HEAD（vLLM `dd9e07a25f`、vllm-ascend `c5766cd5e`、SGLang `4e85e8ed37`、verl `5b550b0c`）。
2. 在窗口提交范围内以 batch invariant / determinism / GDN / DS4 / replay / indexer / router / inductor 等关键词交叉检索提交主题，再对候选 PR 用 GitHub API / PR 页面核对状态（Open / Closed / Merged）。
3. 合入状态以本地提交树为准：提交主题中的 PR 号 + `git show` 检查文件与关键 diff。GitHub API 匿名额度（60 次/小时）中途耗尽，未合入 PR 的内容改以 `https://github.com/<repo>/pull/<n>.diff` 核对。
4. vllm-ascend 主线 BI 面零改动的证据：对算子注册 `vllm_ascend/batch_invariant.py`、采样侧 `vllm_ascend/worker/v2/sample/`、文档 `docs/source/user_guide/feature_guide/batch_invariance.md`、测试 `tests/e2e/pull_request/four_card/rlhf/consistency/` 在窗口范围取 diff，结果均为空。
5. SGLang 窗口内"关闭"的 PR（#44906、#45579）用本地提交树确认实际已合入（search API 的 state=closed 不能区分 merged / closed-unmerged）；#45896 确认关闭未合入。

## 本目录 diff 快照（新增跟踪 PR 的当前改动）

| 文件 | PR | 状态 |
| --- | --- | --- |
| vllm-project_vllm_pull_62192.diff | [vLLM #62192](https://github.com/vllm-project/vllm/pull/62192) | Open，快照 2026-10-23 |
| vllm-project_vllm_pull_62193.diff | [vLLM #62193](https://github.com/vllm-project/vllm/pull/62193) | Open，快照 2026-10-23 |
| vllm-project_vllm-ascend_pull_19442.diff | [Ascend #19442](https://github.com/vllm-project/vllm-ascend/pull/19442) | Open，快照 2026-10-23 |
| vllm-project_vllm-ascend_pull_19237.diff | [Ascend #19237](https://github.com/vllm-project/vllm-ascend/pull/19237) | Open，快照 2026-10-23 |
| sgl-project_sglang_pull_45063.diff | [SGLang #45063](https://github.com/sgl-project/sglang/pull/45063) | Open，快照 2026-10-23 |
| sgl-project_sglang_pull_44923.diff | [SGLang #44923](https://github.com/sgl-project/sglang/pull/44923) | Open，快照 2026-10-23 |
| sgl-project_sglang_pull_44861.diff | [SGLang #44861](https://github.com/sgl-project/sglang/pull/44861) | Open，快照 2026-10-23 |
| sgl-project_sglang_pull_45599.diff | [SGLang #45599](https://github.com/sgl-project/sglang/pull/45599) | Open，快照 2026-10-23 |
| verl-project_verl_pull_8025.diff | [verl #8025](https://github.com/verl-project/verl/pull/8025) | 已合入（2026-10-22），diff 为合入前版本 |

vLLM #62017 的 diff 约 293KB（含大量 Mamba2/SSD 测试），未随目录保存；其改动文件清单见阶段变化文档第 5 节描述，原文件在 GitHub PR 页面可获取。

## 边界

- diff 为快照，PR 后续演化以远端为准。
- 未运行 GPU/NPU 数值测试；不把 PR 作者报告的通过率作为验收结论。
- 上一期（10-16 快照）的 GitHub 状态原始摘录保存在工作区 `.pr-analysis/consistency_refresh_20261016.json`，未随本目录提交。
