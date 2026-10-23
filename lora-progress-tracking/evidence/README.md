# evidence 说明

- `merged-prs-2026-10-16_to_10-23.json`：本期（2026-10-16 → 2026-10-23）合入提交与上期跟踪 PR 状态的结构化记录。
- 生成方式：本地 `git fetch` 三仓库后，对 origin/main 在上期基线 HEAD（vLLM `dd9e07a25`、SGLang `4e85e8ed`、vLLM-Ascend `c5766cd5`）与 2026-10-23 12:03 fetch 终点（vLLM `19c0e170d2`、SGLang `f4a2df3dac`、vLLM-Ascend `db8836752`）之间做提交扫描；关键词 lora/adapter/punica/peft，Ascend 另按 `vllm_ascend/lora`、`vllm_ascend/models/deepseek_v41` 路径过滤；PR 状态以 GitHub 页面（OPEN/MERGED/CLOSED/DRAFT）核验，PR 描述取自 squash commit 与页面 og:description。
- 字段：`merged_vllm` / `merged_sglang` / `merged_vllm_ascend` 为合入提交（pr、subject、commit、UTC/北京时间）；`tracked_status` 为上期跟踪 PR 的当前状态与备注。
- 合入统计：vLLM 17 条（含 3 条 CI）、SGLang 11 条（含 CI 与 diffusion/量化边界项）、vLLM-Ascend 5 条（均为触及 LoRA 相关路径的底座提交，非 LoRA 支持）。
- 状态核验时间：2026-10-23 12:10（北京时间）。未运行模型测试。
