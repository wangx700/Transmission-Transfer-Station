# Sparse HCCL 发送校验优化

2026-10-21；状态：代码、单测、真实 30B 固定 payload 对照和五项完整同步对照全部完成。

## 改动边界

仅修改 VIME 的 Ascend patch：`sparse_weight_patch.py` 内增加私有批量校验函数；`SparseHCCLTrainerWeightTransferEngine._prepare_patches` 在公共准备流程后调用它。删除 Ascend 的逐 patch 完整数值校验 override，让公共结构检查继续由 vLLM 实现。完整单 patch 校验入口仍保留，并复用批量实现。

`SparseWeightPatch`、init/update info、`send_weights`、`send_weight_chunk`、broadcast 次数与 wire 内容不变。vLLM patch 不因本次发送校验优化而改变；训练侧 exporter、diff、gather 不变；不增加 packed 或跳过校验开关。接收端公共校验不变。

流程：公共 send → 公共 lifecycle/结构准备 → Ascend 补充元数据检查 → 每个 patch 在原设备计算越界、重复索引和 NaN 标志 → 按设备汇总读取 → payload 规范化 → 接收 RPC / HCCL broadcast → 等待应用完成。

代码证据：[对应候选源码](../core-source/)、[对应候选源码](../core-source/)、[对应候选源码](../core-source/)、[对应候选源码](../core-source/)。VIME 实际提交的是 `docker/npu_patch/vllm-ascend.patch`，上面为其对应源码树；代码已同步 PR #480 的 1e46fc61。

正常路径每个实际设备、每个发送 chunk 只读取一个汇总标志；发现错误时重新检查并定位具体参数。重复索引按 patch 检查，跨参数相同位置合法；各参数使用自己的 full_shape。不得缓存 tensor 校验结果，原地修改后的 indices/values 每次重新检查。数值检查仍包含排序和扫描，因此不把减少同步次数等同于同比例降低耗时。

## 性能原因及验证口径

旧路径每个非空 patch 的索引和 NaN 检查分别 `.item()`，18,867 个 patch 约 37,734 次主机读取。此前独立同步诊断的 sender_validate 约 119 秒。批量路径减少读取次数，未去掉检查，未改变 MoE 映射或转置。

固定 payload 实验：同一脚本中 legacy / batch / batch / legacy 顺序，两者均开启 MoE 直写，分别预热；计时含发送校验、通信与接收应用完成，不含训练导出/diff/gather、哈希审计和推理。诊断 profiling 与吞吐样本分开。四个 rank 全参数哈希以及恢复到正常模型后的 token 必须一致。

完整同步实验：复用同一 Qwen3-30B-A3B BF16 ND、TP2/PP2/EP2/ETP1、4 训练卡与 4 推理卡配置，分别运行 batch + direct off/on，与已完成的 legacy + direct off/on 对照。首次 seed 单列；每种稳态三轮。full 对照采用同配置独立运行。独立完整训练运行存在权重变化比例及设备抖动，严格归因以固定 payload 对照为主。

## 已完成检查

- Ascend sparse engine / MoE direct：83 项通过，包含正常路径一次读取、跨参数共享位置、late invalid 拒绝 RPC/广播、原地修改重新校验和空批次不读取。
- 公共 loader / profiler 隔离测试：8 项通过。直接运行 vLLM 全局 conftest 的尝试在 allocator teardown 出现 8 个环境错误，未作为通过结果；隔离运行用于核实公共逻辑。
- Ruff 检查通过；固定依赖版本的两个 patch 均可应用，应用源码与候选树一致。

## 已完成硬件对照

| 校验路径 | 两个正式样本（秒） | 均值（秒） |
| --- | --- | ---: |
| 旧版逐 patch 校验 | 132.94499 / 135.70205 | 134.32352 |
| 新版按设备批量校验 | 34.66633 / 33.55274 | 34.10954 |

表注：同一进程、相同 18,867 个 patch、59,633,124 个位置和 341.22347 MiB；两者均开启直写，先分别预热，再 legacy/batch/batch/legacy 顺序；每种 n=2。单位为发送校验、广播和应用完成的墙钟秒数，不含 diff/gather、finish、哈希或推理。均值降低 74.61%；不等于完整同步降幅。

| 独立同步诊断阶段 | 旧校验（秒） | 批量校验（秒） |
| --- | ---: | ---: |
| sender_validate | 116.671 | 18.323 |
| sender_broadcast | 5.272 | 3.663 |
| sender_wait_apply | 11.109 | 10.849 |
| receiver_apply，四 rank 最大值 | 10.893 | 10.622 |

表注：独立开启同步 profiling 的诊断，不混入上表吞吐样本。发送和接收存在等待重叠，不能相加。广播差异属于样本波动，wire 未改变。发送校验降低 84.29%；接收应用基本不变，支持收益来自发送校验的判断。排序、扫描及 Python/算子提交仍存在，未声称消除全部校验成本。

ABBA 四次参数哈希一致；legacy 和 batch 两条路径均精确恢复原始正常模型四 rank 的完整参数哈希；token 均为 `[2796,315,279,17097]`、finish reason `length`。回放脚本退出 0；已审计两条 sender 与八条 receiver 诊断记录。

固定回放原始证据：`current/validation-fixed-replay/comparison.json`、`current/validation-fixed-replay.log`、`current/validation-fixed-replay/server.log`。审计摘要：`replay-audited-summary.json`。完整训练同步结果见下表。

## 完整同步最终对照

| 模式 | seed（秒） | 三轮稳态（秒） | 稳态均值（秒） | seed 加三轮累计（秒） |
| --- | ---: | --- | ---: | ---: |
| Full HCCL | 98.7 | 59.7 / 62.3 / 57.2 | 59.73 | 277.9 |
| 旧校验，关闭直写 | 133.7 | 102.3 / 110.8 / 115.2 | 109.43 | 462.0 |
| 旧校验，开启直写 | 132.9 | 71.0 / 95.1 / 95.7 | 87.27 | 394.7 |
| 批量校验，关闭直写 | 134.5 | 77.0 / 77.6 / 81.6 | 78.73 | 370.7 |
| 批量校验，开启直写 | 122.4 | 49.7 / 51.3 / 52.6 | 51.20 | 276.0 |

表注：Qwen3-30B-A3B BF16/ND、TP2/PP2/EP2/ETP1、NPU 0–7，四训练卡与四推理卡；单位为完整 update_weights 的 elapsed seconds，包含 export/diff、gather、传输、apply 和 lifecycle，排除启动、训练和生成。seed 不进稳态均值；累计列只求四次权重同步耗时之和，不是整个训练任务耗时。各模式为独立运行，payload 略有差异，严格因果对比见固定 payload ABBA。所有五项 Ray 作业 SUCCEEDED、客户端退出 0、四次计时齐全、profiling 关闭。batch/direct-OFF seed 含一次非阻塞栈观测，不用于严格 seed 比较；稳态均未采样。

观测均值：仅批量校验、关闭直写，相比旧校验降低 28.05%；保持直写开启、切换到批量校验，降低 41.33%；批量校验基础上再开启直写，降低 34.97%。当前组合的稳态均值相比 full 降低 14.29%。这些是独立训练运行的观测差异，不是同 payload 严格因果估计。

首次 seed 加三轮累计：full 为 277.9 秒，批量校验加直写为 276.0 秒。seed 仍走 dense bootstrap，优化没有消除其额外成本，因此持续更新的均值收益不能直接等同于短任务总收益。

作业及 payload 详情见相邻 `../complete-sync-audited.json`；每项原始 driver 日志、job metadata、源码指纹和退出码在 `current/` 中。旧校验/full 控制日志位于相邻 `../moe-direct-1021/current/`。两个远程根目录分别是 `/home/w00899129/pr480-moe188-1021` 和 `/home/w00899129/pr480-batch188-1021`。
