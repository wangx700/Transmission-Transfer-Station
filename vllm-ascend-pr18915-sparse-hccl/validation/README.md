# 验证说明

验证日期：2026-10-21。Ascend 提交 ee1f30111；vLLM 源码 c4df37d。结果来自隔离远端环境，未以 CPU mock 代替硬件 E2E。

- [UT 原始日志](unit.log)：74 passed，15 warnings，0.33 秒。
- [完整 E2E 日志](e2e.log)：6 passed，14 warnings，414.31 秒，零跳过。4 个硬件场景加 2 个 CPU 示例布局用例。
- [真实示例结果](example.log)和 [trainer 日志](example/trainer.log)：8192 sparse values，四个 worker 参数哈希与 dense 参考一致，恢复推理 token 一致。
- [移除 int64 转换的预期失败](without-int64.log)：1 failed、48 deselected；这是独立副本中的反向回归，不计入通过数。
- [汇总 JSON](validation.json)：退出码、四个 E2E 场景及真实示例的完整参数哈希、源码摘要和验证范围。
- [复现命令](commands.sh)：依赖保留的隔离源码、native/CANN 扩展和独立 NPU；请按自己的部署环境修改目录及设备号。

硬件测试使用小型 BF16/ND Qwen3 MoE checkpoint，并显式启用 EP。首次启动继承容器 0.31.0 release 标识，与固定源码的 profile_run 接口不匹配；成功重跑设置 VLLM_VERSION=0.30.0 选择此前同基线的兼容路径。该环境纠正不改 PR 运行代码；这些结果不能被解释为其他上游版本或生产模型性能的验证。

公开验证评论：https://github.com/vllm-project/vllm-ascend/pull/18915#issuecomment-6314522982
