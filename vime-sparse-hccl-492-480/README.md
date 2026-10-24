# VIME #492 优化候选与 #480 已验证流程

整理日期：2026-10-24（北京时间）。

**结论：优化候选在接口与本地回归层面可行，并通过单 NPU loader 检查；尚未完成候选版的多卡 HCCL/PP 端到端及 30B 性能复测。** 当前材料发布到本仓库；没有推送修改到 VIME #492 分支。

## 版本与阅读入口

- [#492 原始流程与断点](01-pr492-current.md)：固定到 VIME PR #492 `b7a462f5d7351627109ace757e88a9b01fe854a7`。
- [优化候选时序图、修改位置与可行性](02-optimized-sequence.md)：恢复 #480 的协议实现，增加公共 loader 的 int64 边界比较。
- [#480 已验证时序与性能口径](03-pr480-validated.md)：固定到 VIME PR #480 `1e46fc61f71f6452994dc2dc299650bca2031019`。
- [188 容器与验证记录](04-environment-validation.md)：区分历史验证容器和本次实际可用容器。
- [验证产物](validation/)：补丁应用／源码一致性检查、96 项 CPU 回归摘要、真实单 NPU loader 结果和原始日志。
- [可审阅代码补丁](patches/)：相对 #492 的增量及完整 dependency patch。
- [源码摘录](core-source/)：dense seed、PP、批量校验、MoE 直写、公共 loader 与回归用例。

## 修改边界

实际修改 VIME 工程的 `docker/npu_patch/vllm-ascend.patch`、`docker/npu_patch/vllm.patch` 和 `docker/npu_patch/README.md`。训练 exporter、diff、gather、VIME updater、full NCCL 路径没有改动。

Ascend 依赖段恢复 #480 的 dense bootstrap、PP 完整接收后筛选、发送端批量数值校验、默认关闭的 MoE direct，以及对应 UT/E2E/example。公共 loader 和测试恢复相配套的 #480 段，并对边界比较显式转换 int64。引擎少量 import 排序调整不改变行为。

候选不是上游 160 行 wrapper 的逐字复制；它保留 VIME 所需协议扩展。当前没有依据宣称候选比 #480 更快。

## 安装与复核

`patches/pr492-optimized.patch` 应用在 #492 的上述固定提交；其中会修改两个 dependency patch 和安装说明。`patches/vllm.patch`、`patches/vllm-ascend.patch` 则分别应用到原始固定依赖源码，二者不要叠加应用。

```bash
# 在 VIME #492 b7a462f5 的干净源码中：
git apply --check /path/to/pr492-optimized.patch
git apply /path/to/pr492-optimized.patch

# 在两份未打补丁的固定依赖源码中：
git -C /path/to/vllm apply --check /path/to/vllm.patch
git -C /path/to/vllm apply /path/to/vllm.patch
git -C /path/to/vllm-ascend apply --check /path/to/vllm-ascend.patch
git -C /path/to/vllm-ascend apply /path/to/vllm-ascend.patch
```

固定依赖：vLLM `e6bfe03ad73a3330cb427885aa90d97a12e1c704`；vLLM-Ascend `fd815467c221ee600137f6bdd53fe354d5e7c999`。测试与安装使用现有 native/CANN 库；本次没有重新编译镜像。

运行设置继续为 `--update-weight-mode delta --update-weight-transport sparse_hccl --update-weight-delta-verify-every 0`。MoE 直写需在 rollout worker 设置 `VIME_SPARSE_MOE_DIRECT=1`，默认关闭。

源码快照保留原有版权；Apache 2.0 许可证见 [LICENSE](core-source/LICENSE)。
