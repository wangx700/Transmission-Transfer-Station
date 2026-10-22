# Qwen3-30B-A3B：Megatron → HF → 推理侧稀疏权重同步

资料日期：2026-10-22。本文基于本地 verl/SGLang、VIME、vLLM、vLLM-Ascend 分支代码阅读，shape 和坐标为指定配置下的推导，不代表本文新增完成了多卡真机验证。

- [NaN probe 逐步操作示例](nan-probe-example.md)
- [可运行的缩小版 NaN probe 示例](nan_probe_demo.py)
- [代码证据与来源版本](evidence/README.md)

## 1. 模型和并行假设

模型为非量化 BF16 的 [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。

表注：以下是模型逻辑维度，不包含设备物理布局、额外 padding；Q 输出维度由 heads×head_dim 计算，不等于 hidden size。

| 配置 | 数值 | 说明 |
|---|---:|---|
| 层数 | 48 | 示例 PP0 放全局层0～23，PP1 放24～47 |
| hidden size H | 2048 | 投影输入隐藏维度 |
| Q heads | 32 | Q 输出4096维 |
| KV heads | 4 | K/V 各输出512维 |
| head_dim | 128 | 不使用2048÷32推导 |
| 专家数 E | 128 | 示例连续 EP2 分组，每组64个专家 |
| 单专家 intermediate I | 768 | gate/up 输出维度、down 输入维度 |
| 每 token 激活专家 | 8 | 不改变权重 shape |

训练配置固定：**TP2、PP2、EP2、ETP2、CP1、无VPP，8卡**。

```text
普通参数：world = TP2 × PP2 × dense-DP2 = 8
专家参数：world = ETP2 × EP2 × PP2 × expert-DP1 = 8
```

EP 在 dense-DP 维度上划分专家，不能在 `TP×PP×DP` 后再额外乘一次 EP。只说 TP2/PP2/EP2 还不能唯一确定专家 shape，必须说明 ETP。

推理配置固定：**vLLM TP2、PP2、DP1，开启 EP，得到 EP2，4卡**。Attention 使用 TP2；本地 vLLM 配置逻辑在开启 EP 后将专家内部 TP 设为1，每个 EP rank 持有完整专家。因此训练8卡与推理4卡无需逐 rank 对应。

证据：[vLLM MoE并行配置](evidence/vllm-moe-parallel.txt)。

## 2. 训练 rank 的并行说明

采用逻辑坐标 `(p,d,t)`，分别表示 PP、dense-DP、TP；不是实际部署的 global rank 编号。在本例 ETP=TP=2 下，d 区分两个 EP 分区。

表注：每行一张训练卡；专家连续分组仅作示例，实际以 Megatron 并行组和 Bridge task 为准。dense 副本0/1拥有相同 Attention 参数。

| 逻辑坐标 | 全局层范围 | Attention | 专家 |
|---|---|---|---|
| (0,0,0) | 0～23 | TP0，副本0 | 专家0～63，ETP0 |
| (0,0,1) | 0～23 | TP1，副本0 | 专家0～63，ETP1 |
| (0,1,0) | 0～23 | TP0，副本1 | 专家64～127，ETP0 |
| (0,1,1) | 0～23 | TP1，副本1 | 专家64～127，ETP1 |
| (1,0,0) | 24～47 | TP0，副本0 | 专家0～63，ETP0 |
| (1,0,1) | 24～47 | TP1，副本0 | 专家0～63，ETP1 |
| (1,1,0) | 24～47 | TP0，副本1 | 专家64～127，ETP0 |
| (1,1,1) | 24～47 | TP1，副本1 | 专家64～127，ETP1 |

导出第0层时：PP0 的 Attention 副本0贡献两个 TP 分片，副本1去重；PP0 的两个 EP 分区各两个 ETP rank 都贡献专家分片；PP1 没有第0层参数，但保留全局目录行及零 count，保证集合操作顺序一致。PP>1 时本地导出代码使用 WORLD 合并目录行。

证据：[VIME build_export_index 与 contributes](evidence/vime-export.txt)。

## 3. QKV：三套 name、shape、坐标

表注：统一举全局第0层。Megatron 使用常见 Bridge 全局转换名示例，实际前缀和本地层号以 task 为准；区间左闭右开。Megatron 与推理 shape 相同不代表行排列相同。

| 阶段 | name | shape | 分片解释 |
|---|---|---|---|
| Megatron TP0 | `decoder.layers.0.self_attention.linear_qkv.weight` | [2560,2048] | Q前16 heads、K/V前2 heads，Megatron分组排列 |
| Megatron TP1 | 同上 | [2560,2048] | Q后16 heads、K/V后2 heads |
| HF | `model.layers.0.self_attn.q_proj.weight` | [4096,2048] | TP0贡献行[0,2048)，TP1贡献[2048,4096) |
| HF | `model.layers.0.self_attn.k_proj.weight` | [512,2048] | TP0贡献行[0,256)，TP1贡献[256,512) |
| HF | `model.layers.0.self_attn.v_proj.weight` | [512,2048] | 同K |
| 推理 TP0 | `model.layers.0.self_attn.qkv_proj.weight` | [2560,2048] | 本地Q2048行、K256行、V256行融合 |
| 推理 TP1 | 同上 | [2560,2048] | 同样融合自己负责的Q/K/V |

```text
全量QKV行数 = 32×128 + 4×128 + 4×128 = 5120
TP2后每卡 = 2560行

典型 Megatron GQA排列：
每组 [8个Q heads, 1个K head, 1个V head] = 1280行
每个TP rank有2组，共2560行

vLLM本地排列：[Q_local; K_local; V_local]
```

例如训练TP1的 Megatron 本地 `[0,7]` 更新为0.5，经 probe 得到：

```text
HF q_proj[2048,7] = 0.5
HF flat index = 2048×2048+7 = 4194311
推理TP1 qkv_proj[0,7] = 0.5
```

证据：[Qwen3 MoE的QKV构造与HF名称映射](evidence/vllm-qwen3-moe.txt)。

## 4. MoE：专家编号与 ETP 分片

举全局第0层、全局专家70。连续分组下它位于 EP1，本地专家编号 `70−64=6`。

表注：Megatron grouped-linear 使用常见 `weight6` 命名示例，6是本地专家号，不是HF全局专家号；训练ETP2，fc1按逻辑 `[gate_local;up_local]` 表示。

| 阶段 | name | shape | 内容 |
|---|---|---|---|
| Megatron ETP0 | `decoder.layers.0.mlp.experts.linear_fc1.weight6` | [768,2048] | gate前384行 + up前384行 |
| Megatron ETP1 | 同上 | [768,2048] | gate后384行 + up后384行 |
| Megatron ETP0 | `decoder.layers.0.mlp.experts.linear_fc2.weight6` | [2048,384] | down前384列 |
| Megatron ETP1 | 同上 | [2048,384] | down后384列 |
| HF | `model.layers.0.mlp.experts.70.gate_proj.weight` | [768,2048] | 完整gate |
| HF | `model.layers.0.mlp.experts.70.up_proj.weight` | [768,2048] | 完整up |
| HF | `model.layers.0.mlp.experts.70.down_proj.weight` | [2048,768] | 完整down |

推理侧 EP2、专家内部 TP1，每卡堆叠64个完整专家。

表注：以下是本地非量化 Ascend 分支的无额外 padding 示例。部分版本 module 路径会多出 `routed_experts`；name及实际shape以运行时枚举为准。loader视图与运行时布局是同一权重的不同解释。

| 布局 | name | shape |
|---|---|---|
| checkpoint loader视图 | `model.layers.0.mlp.experts.w13_weight` | [64,1536,2048] |
| checkpoint loader视图 | `model.layers.0.mlp.experts.w2_weight` | [64,2048,768] |
| Ascend转置后运行时 | `model.layers.0.mlp.experts.w13_weight` | [64,2048,1536] |
| Ascend转置后运行时 | `model.layers.0.mlp.experts.w2_weight` | [64,768,2048] |

例如 EP1、ETP1 的 fc1 本地 `[0,7]=0.75`：

```text
HF experts.70.gate_proj[384,7]
HF flat index = 384×2048+7 = 786439
推理EP1，本地expert6
loader视图：w13_weight[6,384,7]
Ascend运行时：w13_weight[6,7,384]
```

证据：[Ascend loader视图与转置](evidence/ascend-layout.txt)、[专家映射和直接稀疏写入](evidence/ascend-moe-direct.txt)。

## 5. slot 和 delta entry 六元组

单个 slot 是 `(hf_name, hf_full_shape)`，完整 delta entry 才是六元组：

```python
(slots, dtype_str, counts, hf_idx, hf_val, gather_group)
```

表注：entry描述一条Megatron参数目录行的HF输出；一条目录行可能拆成多个HF参数，也可能包含多个EP rank的不同专家。

| 字段 | 内容 | 作用 |
|---|---|---|
| slots | 有序的(HF name, HF完整shape)列表 | 指明每个payload分段属于谁 |
| dtype_str | 如bfloat16 | values的类型 |
| counts | 每slot的变化数量 | 切分拼接的indices和values |
| hf_idx | 各slot的flat indices顺序拼接 | 每个index相对于自己所属HF张量 |
| hf_val | 对应位置的当前绝对值 | 覆盖，不是新旧值的差值 |
| gather_group | 合并进程组 | 本地PP>1实现使用WORLD |

`mcore_hf_delta_entry()`返回前五项；外层`iter_delta_entries()`追加第六项。第六项是训练侧合并控制信息，不是要序列化给推理侧的 ProcessGroup 对象。

### QKV示例

以下人为固定Q、K、V顺序用于解释，真实顺序取slot缓存：

```python
slots = [
    ('model.layers.0.self_attn.q_proj.weight', (4096, 2048)),
    ('model.layers.0.self_attn.k_proj.weight', (512, 2048)),
    ('model.layers.0.self_attn.v_proj.weight', (512, 2048)),
]
entry = (
    slots, 'bfloat16',
    [2, 1, 0],
    [4194311, 4194312, 524297],
    [0.5, 0.6, 0.7],
    WORLD,
)
```

前两项index归Q，第三项归K，V为空。`524297`是K自身的flat index，不是Q/K/V拼接大张量的偏移。

### MoE示例

同一目录行的本地专家6在EP0/EP1分别映射到全局专家6/70。交换后的slot表示意如下，实际顺序以交换结果为准：

```python
slots = [
    ('model.layers.0.mlp.experts.6.gate_proj.weight',  (768,2048)),
    ('model.layers.0.mlp.experts.6.up_proj.weight',    (768,2048)),
    ('model.layers.0.mlp.experts.70.gate_proj.weight', (768,2048)),
    ('model.layers.0.mlp.experts.70.up_proj.weight',   (768,2048)),
]
# PP0、EP1、ETP1贡献expert70的一项gate变化
entry = (slots, 'bfloat16', [0,0,1,0], [786439], [0.75], WORLD)
# PP1不拥有第0层，但保持同一slot表和目录顺序
empty = (slots, 'bfloat16', [0,0,0,0], [], [], WORLD)
```

slot表只缓存名称和完整shape，不缓存变化位置。初始化会交换元数据；probe转换本身不gather其他rank的真实权重。证据：[VIME导出与slot交换](evidence/vime-export.txt)。

## 6. 全量基线、diff、发送与应用接口

```text
首次：
训练暂停更新 → Bridge正常导出全量HF → 推理加载
  → 每个参与导出的训练rank保存自己的Megatron shard CPU snapshot

后续：
Bridge.get_conversion_tasks → build_export_index缓存mapping/probe/slots
CPU snapshot H2D → NPU上与当前Megatron shard按位比较
  → local_indices + current_values
  → mcore_hf_delta_entry：本地NaN底板 → probe.megatron_to_hf
  → 提取非NaN的HF全局indices/values → delta entry六元组
  → 按slot和group合并 → 编码元数据及payload → 发送
推理：校验 → PP过滤 → HF完整shape的NaN底板
  → index_copy_填入变化值 → model.load_weights
  → 原生loader完成TP切片、参数融合、专家选择
  → masked copy仅覆盖非NaN位置；Ascend loader view处理布局
```

CPU snapshot是本rank Megatron分片，不是每rank一份完整HF模型。先diff再probe，避免每轮先导出完整HF模型。values是当前绝对值。

verl/SGLang使用`__delta_spec__`张量承载JSON，加`__positions__`、`__values__`，不要求真的写一个`manifest.json`文件。协议还需描述encoding、各参数payload偏移。manifest与六元组是不同阶段的结构。

推理masked copy语义：`destination = where(isnan(source), destination, source)`。上下文必须通过finally恢复原始copy_；全局替换Tensor.copy_期间应隔离并发加载操作。

证据：[verl导出/CPU基线](evidence/verl-diff.txt)、[SGLang解码应用](evidence/sglang-loader.txt)、[vLLM加载契约](evidence/vllm-patch.txt)、[Ascend接收流程](evidence/ascend-receiver.txt)。

## 7. 适配边界和优化

1. **转换必须验证。** probe假设mapping主要做重排、拆分、拼接；跨分片算术混合会传播NaN，不能直接套用。Bridge升级后应比较“真实全量转换”与“各rank probe结果合并”的逐位一致性。
2. **加载契约有限。** 通用NaN路径要求最终同shape浮点copy；多阶段原地变换、量化和自定义写入要单独适配。本地Ascend分支要求非量化、关闭EPLB、ND格式（weight_nz_mode=0）。
3. **基线版本必须一致。** 当前代码可能在准备diff时已刷新snapshot；部分应用失败不能直接继续下一轮。现有VIME路径保持推理暂停并禁止复用，重连后全量seed。增强重试可加入base_version、target_version、update_id和成功确认后提交的snapshot变更。
4. **减少snapshot D2H。** 保留CPU基线H2D比较，但只回写变化indices/values；本地VIME已有稀疏批量刷新。
5. **减少probe开销。** 已知纯重排mapping直接转换稀疏坐标，未知mapping走NaN probe；本地VIME已有快路径。
6. **减少接收端稠密临时量。** NaN通用路径节约网络流量，但仍需填充、扫描完整HF张量；已验证的QKV/MoE布局可直接转换运行时坐标后index_copy_。本地Ascend已有受保护的MoE direct入口。
7. **校验不止checksum。** 还应验证name、shape、dtype、count、payload边界、indices越界/重复、真实NaN及版本连续性。NaN在此作为协议sentinel，不能同时承载真实NaN更新值。

说明：本目录仅新增文档、教学脚本和证据摘录，不修改训练/推理实现。教学脚本验证的是缩小版索引与NaN操作，不替代真实Bridge和多卡NPU测试。
