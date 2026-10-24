# VPP：虚拟流水线并行说明

整理日期：2026-10-24。以Qwen3-30B-A3B的48层为例；本说明是均匀分层的逻辑示例，不代表当前作业已开启VPP，也不包含新的多卡性能验证。

## 1. VPP是什么

VPP（Virtual Pipeline Parallelism，虚拟流水线并行）把一个物理PP stage负责的层进一步拆成多个model chunk，配合交错流水线调度。`virtual_pipeline_model_parallel_size=2`通常表示每个物理PP stage有两个模型chunk。

它不会新增设备、进程或权重副本，也不会把单个QKV或专家矩阵再切一刀；切的是层集合和调度单元。训练框架可以在同一个物理rank上持有多个模型chunk。

官方依据：[Megatron Bridge交错流水线说明](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html#interleaved-pipeline-parallel-schedule)。

表注：表中“切分对象”指该并行方式直接组织的对象，不能将各size不加区别地相乘计算卡数。

| 并行方式 | 切分或组织对象 | 本例作用 |
|---|---|---|
| TP | 单个普通层矩阵 | QKV分为两个TP分片 |
| PP | 模型深度，分配到物理stage | 48层分配给两个物理stage |
| VPP | 每个PP stage内的层集合 | 每stage拆成两个chunk，交错执行 |
| EP | 专家集合 | 128专家分为两组 |
| ETP | 单个专家矩阵 | ETP2时每专家矩阵由两rank持有 |
| DP | 同一权重分片的副本 | 普通层与专家层可有不同DP size |

## 2. PP2、不启用VPP

按均匀连续分层：

```text
PP0：layer 0～23
PP1：layer 24～47

单个microbatch逻辑前向：
PP0[0～23] → PP1[24～47]
```

这里layer编号为HF全局编号，从0开始；实际模型的embedding、最终norm、lm_head需另外按框架规则放置。

## 3. PP2 + VPP2

在均匀交错布局下：

```text
逻辑chunk总数 = PP2 × VPP2 = 4
每chunk Transformer层数 = 48 / 4 = 12
每物理PP stage总层数 = 12 × 2 = 24
```

表注：这是默认均匀交错思路的示例，非自定义pipeline layout；局部层号0～11仅表示chunk内编号，不保证某个版本的named_parameters一定这样命名。

| 物理PP stage | 虚拟chunk | HF全局层范围 | chunk内逻辑层号 |
|---|---|---|---|
| PP0 | chunk0 | 0～11 | 0～11 |
| PP1 | chunk0 | 12～23 | 0～11 |
| PP0 | chunk1 | 24～35 | 0～11 |
| PP1 | chunk1 | 36～47 | 0～11 |

单个microbatch的逻辑前向依赖顺序：

```text
PP0.chunk0[0～11]
   → PP1.chunk0[12～23]
   → PP0.chunk1[24～35]
   → PP1.chunk1[36～47]
```

这是一个microbatch的层依赖路径，不是多microbatch完整时间表。实际interleaved调度会交错安排多个microbatch的前向/反向；同一设备上的两个chunk不会因此自动同时计算。

注意与“把PP0原来的0～23简单拆成0～11、12～23”不同：本例的chunk按虚拟流水线顺序重新分配，所以PP0拥有0～11和24～35，PP1拥有12～23和36～47。

## 4. VPP是否影响4卡、8卡结论

不改变卡数公式。在CP1、无其他额外并行维度时仍是：

```text
world_size = PP × TP × dense-DP
world_size = PP × ETP × EP × expert-DP
```

不额外乘VPP。以下与[TP/EP/ETP主说明](README.md)对应。

表注：卡数是拓扑规模，不是显存可运行保证；VPP改变每个PP stage的chunk划分，不改变表内DP结果。

| 训练配置 | 卡数 | dense-DP | expert-DP | 开启VPP2后 |
|---|---:|---:|---:|---|
| TP2、PP2、EP2、ETP1 | 4 | 1 | 1 | 每卡2个chunk，卡数仍4 |
| TP2、PP2、EP2、ETP2 | 8 | 2 | 1 | 每卡2个chunk，卡数仍8 |

在4卡示例里，A/B仍属PP0，但层集合变成0～11和24～35；C/D仍属PP1，层集合变成12～23和36～47。各层内部TP/EP归属继续按原表解释。

在8卡示例里，A～D仍属PP0，E～H仍属PP1；dense副本、TP、EP、ETP分组不因VPP本身变化，只是每个rank持有的层被组织成两个chunk。

## 5. QKV、MoE shape和全局名称

VPP不改变单层数学维度或TP/ETP切分，因此本例shape保持：

表注：QKV为每个TP rank的本地逻辑shape；fc1/fc2为每个本地专家、每个ETP rank的shape，未包含物理格式/padding。

| 参数 | 分片设置 | 本地shape |
|---|---|---|
| linear_qkv.weight | TP2 | [2560,2048] |
| 专家linear_fc1 | ETP1 | [1536,2048] |
| 专家linear_fc2 | ETP1 | [2048,768] |
| 专家linear_fc1 | ETP2 | [768,2048] |
| 专家linear_fc2 | ETP2 | [2048,384] |

变化的是参数所属物理PP stage与chunk。例如：

```text
HF全局layer24：
  PP2、无VPP → PP1
  PP2、VPP2  → PP0.chunk1

HF name仍然是：
  model.layers.24.self_attn.q_proj.weight
```

HF全局名称不增加`chunk1`前缀，也不能把chunk内layer0直接当成HF全局layer0。

## 6. 对Bridge、snapshot和NaN probe导出的影响

应使用Bridge转换任务提供的全局参数身份，而非手工拼接“本地layer号”。本地导出路径为：

```text
训练module（包含本物理rank上的所有model chunks）
  → bridge.get_conversion_tasks(megatron_model)
  → task.global_param_name / task.param_weight / task.mapping / task.megatron_module
  → build_export_index()
  → 本rank参数snapshot与diff
  → probe.megatron_to_hf(local_nan_buffer, module)
  → HF全局name、shape、indices、values
  → slot合并、发送
```

接口摘录：

```python
tasks = bridge.get_conversion_tasks(megatron_model)
for task in tasks:
    name = task.global_param_name
    param = task.param_weight
    mapping = task.mapping
    module = task.megatron_module
    # 后续按param的归属和mapping建立导出记录。
```

代码证据：[本地VIME导出摘录](../qwen3-30b-megatron-hf-sparse-sync/evidence/vime-export.txt)；该摘录保留源文件行号、HEAD、工作树状态和SHA256。重点查看`build_export_index`与PP/VPP范围说明。它表示代码的设计范围，不等于本次已验证实际VPP训练作业。

正确适配需要保证：

- 遍历所有本地chunk，不能只导出`model[0]`。
- 不同chunk的同名局部参数不能覆盖同一个snapshot键；应使用框架/Bridge提供的唯一全局身份。
- layer24在PP0.chunk1拥有，PP1对同一全局目录行应按导出实现贡献零count。
- slot仍描述HF name和完整shape，不需要把训练VPP直接变成推理端的分片方式。
- 训练VPP与推理PP独立。正确导出到HF坐标后，推理端按自己的布局加载；训练开启VPP不会自动让#18915获得推理PP2支持。

这里NaN probe只处理权重布局映射，不负责流水线microbatch调度。已有“遍历所有chunk”的设计仍需用具体Bridge版本和模型配置检查全局名称唯一性与导出完整性。

## 7. 配置接口和取舍

Megatron Bridge配置示意：

```python
pipeline_model_parallel_size = 2
virtual_pipeline_model_parallel_size = 2
```

这是模型provider参数示意，不是可直接执行的完整训练命令；CLI层有些版本使用“每虚拟stage层数”等选项，不能不核对版本就互换参数含义。

VPP用于减少流水线空泡，但不保证端到端更快：更多chunk边界可能增加通信和调度开销，激活峰值也取决于microbatch、重计算和调度方式。它不会自动将每卡参数量再除以VPP。层数、microbatch数量和schedule的合法性要按所用版本检查。

官方入口：[Bridge Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)、[Megatron Core流水线调度源码](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)。文档链接随上游更新，本次核对日期为2026-10-24。
