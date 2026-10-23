# 核查证据与复查方法

核对日期：2026-10-23（北京时间）。本次固定到官方 main：vLLM `19c0e170d2`、vLLM-Ascend `db8836752`、SGLang `f4a2df3dac`，均为 10-23 提交；结论只用该源码与公开 PR 状态，不用未合入分支证明主线能力。

- `index-replay-prs.json`：本期跟踪的未合入 PR/RFC（含 state 与 draft 标识）与窗口内已合入 PR（含合入 SHA 与提交日期）。
- 未合入项的 state 核查方式：GitHub PR/Issue 页面 `state` 字段与 Draft 标识（10-23 页面核查）；`merged_at=null` 不等于已完成。
- 已合入项来自本地 `git log <上期基线>..origin/main`，vLLM/Ascend/SGLang 上期基线分别为 `dd9e07a25f`、`c5766cd5e`、`4e85e8ed37`；候选经 indexer/DSA/top-k 相关主题与路径过滤筛选，非全量穷举。

复查命令示例：

```bash
git -C <vllm> log --format='%h|%ad|%s' --date=short dd9e07a25f..origin/main -- vllm/v1/attention vllm/model_executor/layers
git -C <vllm-ascend> show 597465395 --stat          # 19220 DSpark bidirectional SparseFlashMla
git -C <sglang> show origin/main:python/sglang/srt/state_capturer/indexer_topk.py | sed -n '30,35p'
```

窗口口径：本目录按"上期基线之后到 10-23 HEAD"的阶段间隔判定（vLLM 窗口 454 个提交、Ascend 151 个、SGLang 562 个）；[rl-replay-consistency](../rl-replay-consistency/evidence/README.md) 按 `merged_at` 自然周口径，两者不混用。日期取主线合入提交日期，与上期报告口径一致。未执行 GPU/NPU 数值、组合或性能验证。
