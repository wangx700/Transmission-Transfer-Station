# 每个测试用例的说明

PR [#18915](https://github.com/vllm-project/vllm-ascend/pull/18915)，Ascend 提交 `ee1f30111881795103af01a0fe19caf41d20a5ac`；上游 vLLM 固定 `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`。本材料按 2026-10-21 验证版本整理，不代表自动更新后的其他上游版本。

## 范围、结果与读法

本清单以远端 `pytest --collect-only -q` 的 80 个真实 nodeid 为准：49 个 sparse engine UT、25 个 routed experts UT、6 个 pytest E2E。74 个 UT 合计通过，6 个 E2E 合计通过且零跳过；没有为每个单项记录耗时。另列真实示例和反向回归两个手工验证项，不把它们混入 pytest 通过数。

| 文件/验证项 | 数量 | 结果与范围 |
| --- | --- | --- |
| sparse engine UT | 49 | CPU tensor + mock 通信，含真实上游 loader 边界校验 |
| routed experts UT | 25 | 布局/存储/视图与已有专家导出回归；其中本 PR 新增 4 个 view 参数化项，另 21 个为已有回归 |
| checkpoint pytest 文件 | 6 | 4 个真实 HCCL 场景 + 2 个 CPU 示例布局项 |
| 真实示例 CLI | 独立一次 | TP2/PP2/EP2，trainer/worker 独占设备，8192 values |
| 移除 int64 的反向回归 | 独立一次 | 预期失败，用于证明新增 UT 能捕获修复回退 |

表注：数量是参数化展开后的执行项；74=49+25，6=4+2。真实示例与反向回归不是额外 pytest 通过项；74 passed 用时 0.33 秒，6 passed 用时 414.31 秒，均为测试进程日志耗时，不含环境准备。所有结果对应 2026-10-21 的固定基线验证。

下文“结果”引用整批执行结果；输入直接来自源码参数表达式。CPU/mock 用例不能据此宣称验证了真实 NPU 通信、异步竞争或生产性能。

## TC-001 · 上游公共接口复用

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_sparse_public_types_and_lifecycle_are_reused`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：比较 patch 类型身份、init/update info 继承，以及 trainer 的构造和公开 send 方法身份。
- 断言：SparseWeightPatch 为同一类型；相关继承成立；__init__/send_weights/send_weight_chunk 未复制实现。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:33](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L33)。

## TC-002 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs0]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'name': ''}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-003 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs1]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'name': 1}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-004 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs2]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'full_shape': None}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-005 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs3]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'full_shape': (-1,)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-006 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs4]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'full_shape': (1.5,)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-007 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs5]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'full_shape': (True, 16)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-008 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs6]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'indices': torch.tensor([0], dtype=torch.int64)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-009 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs7]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'indices': torch.tensor([-1, 15], dtype=torch.int32)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-010 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs8]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'indices': torch.tensor([0, 16], dtype=torch.int32)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-011 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs9]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'indices': torch.tensor([0, 0], dtype=torch.int32)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-012 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs10]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'indices': torch.tensor([[0, 15]], dtype=torch.int32)}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-013 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs11]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'values': torch.tensor([float('nan'), 3.0])}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-014 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs12]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'values': torch.tensor([2, 3])}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-015 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs13]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'values': torch.tensor([2.0])}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-016 · 非法 patch 在通信前被拒绝

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_bad_patch_rejected_before_start[kwargs14]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'values': torch.tensor([[2.0, 3.0]])}`
- 步骤：以参数覆盖合法基准 patch；先直接校验，再经 send_weights 调用。
- 断言：两条入口均抛 ValueError；不调用 start_weight_update，不进入 broadcast。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:74](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L74)。

## TC-017 · 合法 patch 与零元素 payload 校验

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_valid_and_empty_patch`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：校验普通 patch，再校验 indices/values 同时为空的 patch。
- 断言：两者均合法；本用例不证明零 nnz patch 会被过滤。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:85](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L85)。

## TC-018 · trainer 大形状算术边界

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_large_checkpoint_bounds_do_not_wrap_int32`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：用 2**31 元素形状和两个合法 int32 索引调用 validate_sparse_patch，再检查超过 int64 形状。
- 断言：合法边界通过；2**63 形状抛 ValueError。这里只覆盖 trainer，不覆盖接收端。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:90](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L90)。

## TC-019 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs0]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'dtype_names': []}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-020 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs1]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'shapes': []}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-021 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs2]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'num_updates_list': []}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-022 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs3]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'num_updates_list': [-1]}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-023 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs4]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'num_updates_list': [1.5]}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-024 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs5]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'dtype_names': ['int32']}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-025 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs6]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'dtype_names': ['invalid']}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-026 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs7]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'shapes': [[-1]]}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-027 · worker update metadata 防御

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_metadata[kwargs8]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'names': ['']}`
- 步骤：覆盖名称、dtype、shape、num_updates 等 metadata 后构造 UpdateInfo。
- 断言：非法 metadata 在构造阶段抛 ValueError；不需要实际 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:115](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L115)。

## TC-028 · TP/DP 下的唯一 HCCL rank

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_tp_dp_ranks[1-0-3-4]`
- 类型：UT（CPU/mock）。
- 输入：`dp=1; dp_rank=0; rank=3; expected=4`
- 步骤：构造 TP4 worker 和给定 DP index/local rank，mock stateless 建组调用。
- 断言：建组 rank 等于 expected；world_size=4*DP+1，device=0；rank_offset=1。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:138](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L138)。

## TC-029 · TP/DP 下的唯一 HCCL rank

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_tp_dp_ranks[2-0-3-4]`
- 类型：UT（CPU/mock）。
- 输入：`dp=2; dp_rank=0; rank=3; expected=4`
- 步骤：构造 TP4 worker 和给定 DP index/local rank，mock stateless 建组调用。
- 断言：建组 rank 等于 expected；world_size=4*DP+1，device=0；rank_offset=1。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:138](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L138)。

## TC-030 · TP/DP 下的唯一 HCCL rank

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_tp_dp_ranks[2-1-0-5]`
- 类型：UT（CPU/mock）。
- 输入：`dp=2; dp_rank=1; rank=0; expected=5`
- 步骤：构造 TP4 worker 和给定 DP index/local rank，mock stateless 建组调用。
- 断言：建组 rank 等于 expected；world_size=4*DP+1，device=0；rank_offset=1。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:138](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L138)。

## TC-031 · TP/DP 下的唯一 HCCL rank

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_tp_dp_ranks[2-1-3-8]`
- 类型：UT（CPU/mock）。
- 输入：`dp=2; dp_rank=1; rank=3; expected=8`
- 步骤：构造 TP4 worker 和给定 DP index/local rank，mock stateless 建组调用。
- 断言：建组 rank 等于 expected；world_size=4*DP+1，device=0；rank_offset=1。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:138](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L138)。

## TC-032 · 错误组规模提前失败

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_invalid_world_size_rejected_before_rendezvous`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：TP4/DP2 实际需要 9 个参与者，但传入 world_size=5。
- 断言：抛 ValueError，stateless rendezvous 未被调用。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:149](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L149)。

## TC-033 · 拒绝不支持的配置

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_unsupported_configuration[enable_eplb-True]`
- 类型：UT（CPU/mock）。
- 输入：`field='enable_eplb'; value=True`
- 步骤：给 worker 设置 EPLB 或 awq quantization 后 start_weight_update。
- 断言：抛 NotImplementedError；不进入权重写入。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:157](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L157)。

## TC-034 · 拒绝不支持的配置

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_unsupported_configuration[quantization-awq]`
- 类型：UT（CPU/mock）。
- 输入：`field='quantization'; value='awq'`
- 步骤：给 worker 设置 EPLB 或 awq quantization 后 start_weight_update。
- 断言：抛 NotImplementedError；不进入权重写入。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:157](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L157)。

## TC-035 · 含 PP 的 rank 展平

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_tp_pp_dp_ranks[1-0-3-4]`
- 类型：UT（CPU/mock）。
- 输入：`dp=1; dp_rank=0; rank=3; expected=4`
- 步骤：构造 TP2/PP2，并设置给定 DP/local rank，拦截建组接口。
- 断言：DP 内 world_size=TP*PP=4；跨 DP 加 rank_offset 后等于 expected。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:165](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L165)。

## TC-036 · 含 PP 的 rank 展平

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_tp_pp_dp_ranks[2-1-0-5]`
- 类型：UT（CPU/mock）。
- 输入：`dp=2; dp_rank=1; rank=0; expected=5`
- 步骤：构造 TP2/PP2，并设置给定 DP/local rank，拦截建组接口。
- 断言：DP 内 world_size=TP*PP=4；跨 DP 加 rank_offset 后等于 expected。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:165](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L165)。

## TC-037 · 含 PP 的 rank 展平

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_tp_pp_dp_ranks[2-1-3-8]`
- 类型：UT（CPU/mock）。
- 输入：`dp=2; dp_rank=1; rank=3; expected=8`
- 步骤：构造 TP2/PP2，并设置给定 DP/local rank，拦截建组接口。
- 断言：DP 内 world_size=TP*PP=4；跨 DP 加 rank_offset 后等于 expected。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:165](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L165)。

## TC-038 · 单次与分块生命周期

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_caller_owned_chunks_and_one_shot_lifecycle`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：mock prepare/broadcast；先 send_weight_chunk，再 send_weights。
- 断言：chunk 不自动 start/finish；单次更新恰好各一次；总广播两次。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:176](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L176)。

## TC-039 · 空列表不启动更新

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_empty_and_non_sender_noop`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：分别以 sender=True/False 调用两个 send 入口，输入均为空列表。
- 断言：start 未被调用。本用例没有覆盖 non-sender 加非空 payload。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:192](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L192)。

## TC-040 · 部分更新失败不宣告完成

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_failure_does_not_finish_partial_update`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：prepare 返回 patch，broadcast 主动抛 RuntimeError，再调用 send_weights。
- 断言：异常向外传播，finish_weight_update 未被调用；不证明自动回滚。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:200](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L200)。

## TC-041 · 接收 payload 的设备与结构

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_receive_on_communicator_device`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：以 CPU mock communicator 及零更新 metadata 调用 receive，拦截 loader。
- 断言：传给 loader 的名称/形状正确，values 位于 communicator 的 CPU device；无广播。不证明真实 NPU 跨设备同步。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:211](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L211)。

## TC-042 · 接收端 int64 与 2**31 边界

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_receive_large_checkpoint_passes_upstream_bounds_validation`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：mock broadcast 填入 [0,2147483647] 和两个值；shape=(65536,32768)；wrap 真实上游 loader，在 torch.full staging 前抛测试专用异常。
- 断言：真实范围校验通过并到达 NaN staging；仅一次 staging 请求；线上 int32/float32；loader 索引为 int64 且数值不变；不分配完整大 tensor。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:229](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L229)。

## TC-043 · 重复关闭幂等

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_shutdown_closes_once`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：给 worker 放入 mock group，连续 shutdown 两次。
- 断言：group.close 仅调用一次，避免重复释放。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:270](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L270)。

## TC-044 · PP 过滤不得改变 collective 序列

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_pp_selection_happens_after_all_broadcasts`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：两 stage 各提供一个 patch；mock missing helper 在被调用时检查广播计数。
- 断言：先完成四次 indices/values 广播，再过滤；loader 只收到本 stage 参数。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:279](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L279)。

## TC-045 · payload 设备与连续性归一化

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_prepare_moves_cpu_views_to_communicator_device`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：以切片制造非连续 indices/values，调用 trainer prepare，communicator mock 为 CPU。
- 断言：输出连续且 values 不变；这是归一化语义测试，不是实测 CPU→NPU 搬运。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:305](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L305)。

## TC-046 · delta backend 不接受 WeightSource

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_source_is_rejected`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：构造 trainer 时显式提供 source。
- 断言：抛包含 WeightSource 的 ValueError，防止稳定参数流与每轮 patch 接口混用。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:318](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L318)。

## TC-047 · 多层 view 清理顺序

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_checkpoint_loader_views_unwind_all_layers[None]`
- 类型：UT（CPU/mock）。
- 输入：`failure=None`
- 步骤：模型中注册 first/second 两个带 view 的层；根据 failure 参数正常退出、第二层进入失败或加载体失败，记录 enter/exit。
- 断言：按已成功进入的层逆序退出；异常保留；entry 失败不会进入加载体。验证资源清理，不验证权重回滚。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:324](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L324)。

## TC-048 · 多层 view 清理顺序

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_checkpoint_loader_views_unwind_all_layers[entry]`
- 类型：UT（CPU/mock）。
- 输入：`failure='entry'`
- 步骤：模型中注册 first/second 两个带 view 的层；根据 failure 参数正常退出、第二层进入失败或加载体失败，记录 enter/exit。
- 断言：按已成功进入的层逆序退出；异常保留；entry 失败不会进入加载体。验证资源清理，不验证权重回滚。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:324](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L324)。

## TC-049 · 多层 view 清理顺序

- Nodeid：`tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py::test_checkpoint_loader_views_unwind_all_layers[loading]`
- 类型：UT（CPU/mock）。
- 输入：`failure='loading'`
- 步骤：模型中注册 first/second 两个带 view 的层；根据 failure 参数正常退出、第二层进入失败或加载体失败，记录 enter/exit。
- 断言：按已成功进入的层逆序退出；异常保留；entry 失败不会进入加载体。验证资源清理，不验证权重回滚。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py:324](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py#L324)。

## TC-050 · 单层 GMM loader view

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_checkpoint_loader_view_preserves_storage_and_restores_layout[False]`
- 类型：UT（CPU/mock）。
- 输入：`fail=False`
- 步骤：在共享底层存储的 transpose view 中写入两个位置，再正常退出或主动抛错。
- 断言：w13/w2 恢复推理 shape，写入值映射正确，parameter data_ptr 不变；异常不会回滚写入值。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:18](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L18)。

## TC-051 · 单层 GMM loader view

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_checkpoint_loader_view_preserves_storage_and_restores_layout[True]`
- 类型：UT（CPU/mock）。
- 输入：`fail=True`
- 步骤：在共享底层存储的 transpose view 中写入两个位置，再正常退出或主动抛错。
- 断言：w13/w2 恢复推理 shape，写入值映射正确，parameter data_ptr 不变；异常不会回滚写入值。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:18](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L18)。

## TC-052 · 不支持的专家存储拒绝进入

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_checkpoint_loader_view_rejects_unsupported_expert_storage[False]`
- 类型：UT（CPU/mock）。
- 输入：`nz=False`
- 步骤：根据 nz 参数构造缺少标准 w13/w2 tensor 参数，或 NPU format=29 的 NZ tensor 场景。
- 断言：进入上下文时抛 NotImplementedError，with 体不执行。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:43](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L43)。

## TC-053 · 不支持的专家存储拒绝进入

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_checkpoint_loader_view_rejects_unsupported_expert_storage[True]`
- 类型：UT（CPU/mock）。
- 输入：`nz=True`
- 步骤：根据 nz 参数构造缺少标准 w13/w2 tensor 参数，或 NPU format=29 的 NZ tensor 场景。
- 断言：进入上下文时抛 NotImplementedError，with 体不执行。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:43](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L43)。

## TC-054 · EPLB 导出视图展平

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_get_expert_weights_flattens_layout_aware_views`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：为两个 local experts 构造 tensor views，调用 get_expert_weights。
- 断言：形状按 expert 维展平，首个 view 与原 tensor 共享存储。属于已有相邻回归。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:92](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L92)。

## TC-055 · 独立 expert tensor 不被强制拼接

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_get_expert_weights_preserves_independent_expert_tensors`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：把每 expert 独立 tensor 列表交给 get_expert_weights，再 empty_like。
- 断言：返回 EplbExpertTensorList，元素对象保持，buffer 类型/shape 正确且 offset=0。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:101](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L101)。

## TC-056 · 逐 expert 保留 NPU 格式

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffers_preserve_each_source_npu_format[nz-float-weight]`
- 类型：UT（CPU/mock）。
- 输入：`dtype=torch.float16; npu_formats=(29, 50)`
- 步骤：使用 fake NPU tensors 和给定 dtype/formats，mock empty_with_format 并调用 empty_like。
- 断言：每个 source 独立查询格式；按其 acl_format 分配，保留 shape/dtype/device，requires_grad=False。不是实际 NZ NPU kernel 验证。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:127](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L127)。

## TC-057 · 逐 expert 保留 NPU 格式

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffers_preserve_each_source_npu_format[nd-float-weight]`
- 类型：UT（CPU/mock）。
- 输入：`dtype=torch.float16; npu_formats=(2, 2)`
- 步骤：使用 fake NPU tensors 和给定 dtype/formats，mock empty_with_format 并调用 empty_like。
- 断言：每个 source 独立查询格式；按其 acl_format 分配，保留 shape/dtype/device，requires_grad=False。不是实际 NZ NPU kernel 验证。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:127](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L127)。

## TC-058 · 逐 expert 保留 NPU 格式

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffers_preserve_each_source_npu_format[nz-int8-weight]`
- 类型：UT（CPU/mock）。
- 输入：`dtype=torch.int8; npu_formats=(29, 50)`
- 步骤：使用 fake NPU tensors 和给定 dtype/formats，mock empty_with_format 并调用 empty_like。
- 断言：每个 source 独立查询格式；按其 acl_format 分配，保留 shape/dtype/device，requires_grad=False。不是实际 NZ NPU kernel 验证。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:127](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L127)。

## TC-059 · 逐 expert 保留 NPU 格式

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffers_preserve_each_source_npu_format[nz-packed-weight]`
- 类型：UT（CPU/mock）。
- 输入：`dtype=torch.int32; npu_formats=(29, 50)`
- 步骤：使用 fake NPU tensors 和给定 dtype/formats，mock empty_with_format 并调用 empty_like。
- 断言：每个 source 独立查询格式；按其 acl_format 分配，保留 shape/dtype/device，requires_grad=False。不是实际 NZ NPU kernel 验证。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:127](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L127)。

## TC-060 · 逐 expert 保留 NPU 格式

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffers_preserve_each_source_npu_format[nd-integer-scale]`
- 类型：UT（CPU/mock）。
- 输入：`dtype=torch.int64; npu_formats=(2, 2)`
- 步骤：使用 fake NPU tensors 和给定 dtype/formats，mock empty_with_format 并调用 empty_like。
- 断言：每个 source 独立查询格式；按其 acl_format 分配，保留 shape/dtype/device，requires_grad=False。不是实际 NZ NPU kernel 验证。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:127](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L127)。

## TC-061 · 禁止改变 expert buffer 属性

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffer_rejects_property_overrides[changed-dtype]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'dtype': torch.float32}`
- 步骤：调用 empty_like 时给出某个属性覆盖。
- 断言：dtype/device/memory_format/requires_grad 的指定覆盖均抛 TypeError。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:187](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L187)。

## TC-062 · 禁止改变 expert buffer 属性

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffer_rejects_property_overrides[changed-device]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'device': 0}`
- 步骤：调用 empty_like 时给出某个属性覆盖。
- 断言：dtype/device/memory_format/requires_grad 的指定覆盖均抛 TypeError。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:187](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L187)。

## TC-063 · 禁止改变 expert buffer 属性

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffer_rejects_property_overrides[changed-memory-format]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'memory_format': torch.contiguous_format}`
- 步骤：调用 empty_like 时给出某个属性覆盖。
- 断言：dtype/device/memory_format/requires_grad 的指定覆盖均抛 TypeError。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:187](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L187)。

## TC-064 · 禁止改变 expert buffer 属性

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffer_rejects_property_overrides[requires-grad]`
- 类型：UT（CPU/mock）。
- 输入：`kwargs={'requires_grad': True}`
- 步骤：调用 empty_like 时给出某个属性覆盖。
- 断言：dtype/device/memory_format/requires_grad 的指定覆盖均抛 TypeError。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:187](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L187)。

## TC-065 · 拒绝非零 storage offset

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffer_rejects_nonzero_storage_offset`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：source offset=1，mock 分配器使其一旦被调用就失败。
- 断言：提前抛包含 storage_offset=1 的 AssertionError，不调用分配器。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:194](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L194)。

## TC-066 · buffer stride 不匹配拒绝

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_eplb_expert_buffer_rejects_stride_mismatch`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：模拟分配器返回相同逻辑 shape 但 stride 不匹配的转置 tensor。
- 断言：断言失败并报告 source/buffer，避免传输依赖错误物理布局。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:213](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L213)。

## TC-067 · 拒绝无专家权重视图的量化路径

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_get_expert_weights_rejects_unsupported_quantization`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：让 quant method 返回空权重 views。
- 断言：抛 NotImplementedError，说明 weight views 未定义。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:242](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L242)。

## TC-068 · 缺少 get_eplb_weight_views 接口

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_get_expert_weights_rejects_missing_weight_view_contract`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：给 quant_method 一个不含协议方法的对象。
- 断言：抛 NotImplementedError，明确要求相应接口。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:247](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L247)。

## TC-069 · expert 维长度错误

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_get_expert_weights_rejects_non_expert_first_dimension`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：local_num_experts=2，却提供首维为 3 的 tensor。
- 断言：抛 ValueError，指出 first dimension。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:256](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L256)。

## TC-070 · 独立专家列表长度错误

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_get_expert_weights_rejects_wrong_expert_tensor_list_length`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：local_num_experts=2，却只提供一个 expert tensor。
- 断言：抛 ValueError，要求列表长度等于 local_num_experts。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:261](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L261)。

## TC-071 · 不能零拷贝展平的布局

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_get_expert_weights_rejects_non_contiguous_view`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：提供 transpose 后不能直接 view 展平的 expert tensor。
- 断言：抛 ValueError，避免偷偷复制后破坏 storage 关联。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:266](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L266)。

## TC-072 · 旧/新 runner 的 expert map 选择

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_ascend_expert_map_follows_model_runner[False]`
- 类型：UT（CPU/mock）。
- 输入：`use_v2_model_runner=False`
- 步骤：同时设置 legacy map 和 upstream map，根据 use_v2_model_runner 切换。
- 断言：旧 runner 返回 legacy map；v2 返回 upstream map，对象身份一致。不是 sparse EPLB 支持声明。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:272](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L272)。

## TC-073 · 旧/新 runner 的 expert map 选择

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_ascend_expert_map_follows_model_runner[True]`
- 类型：UT（CPU/mock）。
- 输入：`use_v2_model_runner=True`
- 步骤：同时设置 legacy map 和 upstream map，根据 use_v2_model_runner 切换。
- 断言：旧 runner 返回 legacy map；v2 返回 upstream map，对象身份一致。不是 sparse EPLB 支持声明。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:272](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L272)。

## TC-074 · 兼容两种 expert map 更新入口

- Nodeid：`tests/ut/ops/test_routed_experts.py::test_update_expert_map_preserves_upstream_and_legacy_contracts`
- 类型：UT（CPU/mock）。
- 输入：`固定输入，见步骤和源码；无 pytest 参数化。`
- 步骤：无参调用 update_expert_map 检查父类入口，再传 legacy map 显式更新。
- 断言：无参委托父类；显式参数更新 ascend map 与 manager map。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/ut/ops/test_routed_experts.py:287](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/ut/ops/test_routed_experts.py#L287)。

## TC-075 · 真实 HCCL 对 dense 参考

- Nodeid：`tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py::test_sparse_checkpoint_matches_native_dense[tp4-ep4]`
- 类型：硬件 E2E。
- 输入：`tp_size=4; pp_size=1; dp_size=1`
- 步骤：生成 num_layers=PP、8 experts 的 BF16 checkpoint；建立 dense 参考并记录全 worker 哈希/生成；恢复 baseline；pause 后每四条记录为一 chunk 经真实 HCCL 更新，finish 后检查 hash，再 resume 生成。
- 断言：所有 worker 的所有 named_parameters 哈希相等；包含 DP 副本；基线与参考确实不同；greedy 4 token 和 finish_reason 相等。重复 q_proj patch 按输入顺序覆盖。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py:192](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py#L192)。

## TC-076 · 真实 HCCL 对 dense 参考

- Nodeid：`tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py::test_sparse_checkpoint_matches_native_dense[tp2-dp2-ep4]`
- 类型：硬件 E2E。
- 输入：`tp_size=2; pp_size=1; dp_size=2`
- 步骤：生成 num_layers=PP、8 experts 的 BF16 checkpoint；建立 dense 参考并记录全 worker 哈希/生成；恢复 baseline；pause 后每四条记录为一 chunk 经真实 HCCL 更新，finish 后检查 hash，再 resume 生成。
- 断言：所有 worker 的所有 named_parameters 哈希相等；包含 DP 副本；基线与参考确实不同；greedy 4 token 和 finish_reason 相等。重复 q_proj patch 按输入顺序覆盖。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py:192](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py#L192)。

## TC-077 · 真实 HCCL 对 dense 参考

- Nodeid：`tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py::test_sparse_checkpoint_matches_native_dense[tp1-pp2-ep1]`
- 类型：硬件 E2E。
- 输入：`tp_size=1; pp_size=2; dp_size=1`
- 步骤：生成 num_layers=PP、8 experts 的 BF16 checkpoint；建立 dense 参考并记录全 worker 哈希/生成；恢复 baseline；pause 后每四条记录为一 chunk 经真实 HCCL 更新，finish 后检查 hash，再 resume 生成。
- 断言：所有 worker 的所有 named_parameters 哈希相等；包含 DP 副本；基线与参考确实不同；greedy 4 token 和 finish_reason 相等。重复 q_proj patch 按输入顺序覆盖。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py:192](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py#L192)。

## TC-078 · 真实 HCCL 对 dense 参考

- Nodeid：`tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py::test_sparse_checkpoint_matches_native_dense[tp2-pp2-ep2]`
- 类型：硬件 E2E。
- 输入：`tp_size=2; pp_size=2; dp_size=1`
- 步骤：生成 num_layers=PP、8 experts 的 BF16 checkpoint；建立 dense 参考并记录全 worker 哈希/生成；恢复 baseline；pause 后每四条记录为一 chunk 经真实 HCCL 更新，finish 后检查 hash，再 resume 生成。
- 断言：所有 worker 的所有 named_parameters 哈希相等；包含 DP 副本；基线与参考确实不同；greedy 4 token 和 finish_reason 相等。重复 q_proj patch 按输入顺序覆盖。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py:192](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py#L192)。

## TC-079 · HF 示例 patch 坐标转换

- Nodeid：`tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py::test_example_patches_both_pipeline_ends_and_experts[False]`
- 类型：CPU 示例转换。
- 输入：`fused=False`
- 步骤：动态加载示例；以四层、四 experts 的小模型模拟 fused 或逐 expert 参数，调用 make_patches。
- 断言：恰好四个 patch，覆盖 layers 0/3 和 experts 0/2；shape/索引/值对应前两行交换；patch 外位置保持原值。本用例不启动 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py:318](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py#L318)。

## TC-080 · HF 示例 patch 坐标转换

- Nodeid：`tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py::test_example_patches_both_pipeline_ends_and_experts[True]`
- 类型：CPU 示例转换。
- 输入：`fused=True`
- 步骤：动态加载示例；以四层、四 experts 的小模型模拟 fused 或逐 expert 参数，调用 make_patches。
- 断言：恰好四个 patch，覆盖 layers 0/3 和 experts 0/2；shape/索引/值对应前两行交换；patch 外位置保持原值。本用例不启动 HCCL。
- 结果：所属批次通过；未单独记录耗时。
- 代码：[tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py:318](https://github.com/wangx700/vllm-ascend/blob/ee1f30111881795103af01a0fe19caf41d20a5ac/tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py#L318)。

## TC-081 · 真实 HCCL 示例 CLI（独立验证）

输入：可复现两层、8 experts 的小型 HF Qwen3 MoE checkpoint；trainer 物理 NPU 8，TP2/PP2/EP2 worker 使用 9–12。执行原样 examples/rl/rlhf_sparse_hccl.py，不以 --help 或 make_patches 单测代替。

步骤：dense 参考记录 → 恢复 baseline → 原样 trainer 生成首尾层的 expert row-swap patch → pause/真实 HCCL/send_weights/finish → resume → 再比较。

断言：8192 sparse values；四个 worker 的完整参数哈希与 dense 参考一致；每个 worker 的权重与 baseline 不同；恢复生成 [365,139,400,77] 与 dense 相等。更新前后 token 恰好相同，不能宣称 token 改变。退出码 0。

## TC-082 · int64 反向回归（独立验证）

在独立源码副本中只移除 receiver 的 int64 转换，运行 TC 对应的 `test_receive_large_checkpoint_passes_upstream_bounds_validation`。结果：上游范围校验抛 ValueError，1 failed / 48 deselected，退出码 1，为预期结果；原始源码 74 UT 全部通过。此失败证明测试能捕获这一行修复的回退，不计入正常通过数。

## 主要未覆盖项

- 不同 ambient NPU 与 communicator NPU 下的 worker 完成同步设备，尚无直接回归。
- 零 nnz patch 列表过滤尚未实施；现有 empty 测试不是这一优化的覆盖。
- 未测试 PP+DP 同时大于 1 的硬件组合，也没有 EP 关闭的硬件矩阵；TP/PP/DP/EP 四行不是全组合穷举。
- 硬件使用 BF16、ND、eager、小模型；没有生产规模性能、真实训练优化器步骤、量化/NZ/EPLB 正向支持或故障回滚验证。
- 两个示例转换用例只检查 patch 构造；真实 CLI 是单独的手工验证，尚未自动包含在六项 pytest 中。
- 上下文清理测试恢复布局/退出顺序，不保证权重数值事务回滚。

## 执行与证据

```bash
python -m pytest tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py tests/ut/ops/test_routed_experts.py -q
VLLM_VERSION=0.30.0 ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python -m pytest -rs tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py
```

需先进入保留的隔离环境并加载 CANN/PYTHONPATH；完整命令、版本纠正过程、原始日志见[验证说明及记录](validation/README.md)。首次 0.31.0 release 标识错误的启动失败未计为通过。本报告没有重新执行硬件测试，仅重新收集 nodeid 并逐项核对代码和既有结果。

验证评论：https://github.com/vllm-project/vllm-ascend/pull/18915#issuecomment-6314522982
