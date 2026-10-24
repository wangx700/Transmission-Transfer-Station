# 188环境与验证边界

核查日期：2026-10-24，北京时间。未把内网连接口令放入本仓库。

## 当前容器与历史容器

本次通过SSH读取188的Docker列表，确认`cjy-vllm-acend-1019`运行中；进入容器后确认以下三个历史目录存在：

- `/home/w00899129/pr480-batch188-1021`
- `/home/w00899129/pr480-moe188-1021`
- `/home/w00899129/pr480-opt188-1019`

```bash
ssh root@192.168.13.188
docker exec -it cjy-vllm-acend-1019 bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/w00899129/vllm-code/vllm_vime/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
```

历史脚本与cleanup记录显示，#480后期验证使用的是`cjy-pr480-opt188-r2`，更早脚本使用`cjy-pr480-opt-1019`。后期验证结束后关闭了专用容器；当前进入保存历史目录的运行容器，不意味着历史测量容器名称改变。

本次`.190` SSH连接超时；188上的`cjy-vime-0623`不存在。随后按历史信息核实了上述当前可用容器。没有停止、重启或清理其他任务。

## 本次已经完成

| 验证层级 | 结果 | 明确范围 |
| --- | --- | --- |
| 原#492接口核查 | 找到dense方法缺失与PP拒绝 | 静态代码＋执行PP guard |
| 优化候选补丁 | 两份固定依赖均可直接应用 | vLLM 10个文件，Ascend 18个文件；生成源码逐文件一致、Python语法通过 |
| 本地CPU回归 | 96 passed | NPU/HCCL相关引导和通信模拟；真实CPU tensor逻辑，不代表多卡通信通过 |
| Ruff | 通过 | 公共loader/test及Ascend engine/validator/direct，使用各依赖工程配置 |
| 单NPU loader | PASS | 真实NPU tensor；BF16 dense→sparse目标值覆盖、有效/无效边界检查 |
| 候选多卡HCCL/PP | 未执行 | 不能替代为历史#480结果 |
| 候选30B性能 | 未执行 | 没有新延迟数字或加速承诺 |

表注：这是本次2026-10-24优化候选的验证状态。96为本地测试总数，包含复用回归及新增5项边界测试；测试会话耗时不是权重同步延迟。

真实NPU检查在当前188容器内，使用`ASCEND_RT_VISIBLE_DEVICES=14`，独立目录`/home/w00899129/pr492-optimized-loader-20261024`，没有替换已安装依赖或历史验证源码。PyTorch显示`2.10.0+cpu`，配合`torch_npu 2.10.0.post4`执行真实NPU tensor；版本后缀不表示本次只用了CPU。

BF16全量`[1,2,3,4]`后，用indices`[0,3]`、values`[0,7]`做sparse覆盖，实际结果`[0,2,3,7]`。大shape仅执行真实校验，由测试回调消费patch，避免分配数十GiB staging；不代表完成巨型权重实际加载。

### int32边界差异

在本地PyTorch 2.14.1 CPU，int32 tensor与`2**31`／`2**31+1`直接比较可把合法索引判为越界。本次188 NPU版本未复现旧表达式的错误，三个边界shape均通过。因此显式int64比较是消除后端差异的防御性修正，不声称已在该NPU栈复现旧版故障。

候选对`2**31-1`、`2**31`、`2**31+1`三个shape接受合法int32索引；对负数和等于numel的索引拒绝。wire仍为int32，不扩大其可表示坐标范围。

原始结果：[npu-loader-result.json](validation/npu-loader-result.json)、[npu-loader.log](validation/npu-loader.log)。历史日志及性能使用不同版本和测试范围，见[历史流程说明](03-pr480-validated.md)。

## 后续多卡验收

在不影响其他作业的可用设备上，为候选单独准备完整固定源码与native库映射，运行依赖patch中的`test_sparse_hccl_checkpoint.py`，至少覆盖`restore-normal`与`tp2-pp2-ep2`、direct OFF/ON。断言每个rank完整参数hash和确定性token匹配dense reference。然后以相同配置重跑30B full/sparse，seed单列，性能对照关闭profiling；严格归因继续使用同payload replay。

这些是未完成验收项，不是本次成功记录。
