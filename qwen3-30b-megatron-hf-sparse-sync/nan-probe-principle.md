# NaN probe实现原理：复用真实转换，模拟跨卡取数

整理日期：2026-10-24。本文总结本地VIME/verl的Megatron-Bridge probe实现。依据代码阅读，不表示新增完成真实Bridge、多卡或NPU验证。

## 1. 一句话解释

**保留Megatron→HF的真实转换算法，在mapping副本上把真实gather替换为本地NaN占位模拟；让变化值经过原来的拆分、拼接、重排，自动得到它们在HF张量中的位置。**

`megatron_to_hf()`本来就存在，不需要重新实现。需要改造的是它内部“去其他rank取数据”的动作，而不是外部接口名称。

NaN在这里表示“本rank没有提供这个位置的更新”，不是权重变化值。真实更新values是当前权重绝对值，推理侧执行覆盖，不执行加法。

## 2. 为什么需要这层转换

训练侧拿到的是Megatron本地分片的坐标，但推理侧需要HF/checkpoint坐标。两者可能有QKV融合/分组排列、gate/up融合、专家本地编号、TP/ETP切分及PP/VPP层归属等差别。

直接把本地flat index发过去通常不够；自己重写所有坐标映射又容易与Bridge实现不一致。probe利用Bridge现成转换代码完成布局变换。

```text
Megatron本地分片坐标
  → 现成Bridge mapping的转换算法
  → HF全局参数名、完整shape、flat index
```

## 3. 全流程和接口

```text
初始化：
bridge.get_conversion_tasks(model)
  → 参数、全局任务名、mapping、module
  → make_probe(mapping, module)
      → 提前初始化延迟mapping
      → 递归复制mapping树
      → 替换副本中的gather和组引用
  → 建立并交换HF slot元数据

每轮更新：
当前Megatron shard + CPU snapshot H2D
  → 设备上diff
  → local_indices + current_values
  → 本地Megatron shape的NaN底板
  → 填入变化值
  → probe.megatron_to_hf(buf, module)
  → 扫描HF输出中的非NaN项
  → slots、dtype、counts、hf_idx、hf_val
  → 外层追加gather_group形成六元组
  → 合并稀疏贡献、发送
```

CPU snapshot是本rank的Megatron分片；顺序为先diff再probe，不是先导出完整HF模型再做HF diff。snapshot更新和失败恢复是同步协议的职责，probe自身不提供事务提交或回滚。

## 4. 方法具体如何替换

本地`make_probe`中的关键操作是：

```python
def _gather_tp(tensor, _c=c):
    nan = _nan_block(tensor.shape, tensor.dtype, tensor.device, nan_pool)
    out = [nan] * _c.tp_size
    out[_c.tp_rank] = tensor
    return out

c.gather_from_tp_ranks = _gather_tp
```

这里`c`是mapping副本。给实例属性赋一个同名函数后，原转换方法里的`self.gather_from_tp_ranks(weight)`就会调用这个替换函数。没有修改原mapping类的源码，也没有在此执行真实all-gather。

解释用的调用示意：

```python
# 原转换方法继续复用；以下简化代码不代表所有mapping内部完全相同。
def megatron_to_hf(self, weight, module):
    shards = self.gather_from_tp_ranks(weight)
    return convert_layout(shards, module)

original.megatron_to_hf(weight, module)  # 原始对象 → 真实gather
probe.megatron_to_hf(buf, module)       # 副本对象 → NaN模拟gather
```

`_c=c`让模拟函数记住当前副本，从它读取TP size/rank。函数直接赋在实例上，不会像类中定义的方法那样自动绑定self，因此首个参数直接是tensor。

## 5. TP2的占位为什么必须保留rank位置

正常gather返回`[TP0真实分片, TP1真实分片]`。probe分别在两个rank上本地构造：

```text
TP0：[本rank变化底板, NaN占位]
TP1：[NaN占位, 本rank变化底板]
```

列表长度、块shape及本rank位置与真实gather一致，但其他rank的真实权重没有取回来。NaN块可按shape/dtype/device缓存复用，应按实现要求保持只读。

不能把TP1的分片放到列表第0项，否则可能映射到TP0的HF位置。也不能随意把tp_size改为1，否则mapping可能按单rank布局计算reshape、head/group维度或反交错。

“保留真实几何关系”指保留原TP/ETP的size/rank及相应布局，不表示所有通信组都原样保留。本地代码将probe的PP组设为size1/rank0，让拥有参数stage上的PP广播走本地恒等路径；不拥有参数的PP rank按目录输出零count。

## 6. AutoMapping和delegate的白话解释

AutoMapping是“分派员”，delegate是它选中的“实际执行者”，通常保存在`_mapping`属性中。delegate只是Python对象，不是额外进程或rank。

```text
QKVMapping
  → AutoMapping：判断参数按哪种方式切分
      → Column/Row/Replicated等具体mapping：执行转换及可能的gather
```

`megatron_to_hf`接口一直存在，但具体执行对象可能首次调用时才创建。若复制时内部对象还是None，随后新创建的对象可能拿到真实进程组，绕过外层的通信替换。

因此先调用`_warm_lazy_mappings`提前创建可识别的delegate，再逐层复制。warm是初始化对象，不是先执行真实权重gather；它可能初始化原对象的延迟缓存，但通信方法替换应作用于副本。

只浅复制外层也不够：父副本可能仍引用原来的子对象。实际递归核心为：

```python
def _inject(m):
    c = _stub(copy.copy(m))
    for attr, value in list(vars(c).items()):
        if isinstance(value, MegatronParamMapping):
            _warm_lazy_mappings(value, module)
            setattr(c, attr, _inject(value))
    return c
```

`_stub`替换当前副本的通信，`_inject`继续复制子mapping并替换父副本的引用。最终原始mapping树正常通信，probe副本树本地模拟取数。

副本的TP/ETP等组引用还被替换成`_ProbeGroup`：它保留size/rank查询，但不支持的通信访问会报错。该边界能暴露未适配的新通信路径，不能理解为对任意mapping的自动兼容保证。

## 7. Qwen3-30B-A3B具体坐标示例

配置TP2、H2048、Q heads32、KV heads4、head_dim128。每rank合计16Q、2K、2V。典型Megatron GQA排列包含两个group，每group为8Q、1K、1V，不能直接当成连续的[16Q;2K;2V]。

TP1的Megatron本地QKV shape为[2560,2048]。假设只有本地[0,7]更新成0.5：

```python
buf = torch.full((2560, 2048), float('nan'), dtype=dtype, device=device)
buf.view(-1)[local_indices] = local_values  # 本例indices=[7]，values=[0.5]
outs = probe.megatron_to_hf(buf, module)
```

TP1的内部模拟gather为`[NaN分片, buf]`，随后真实mapping执行GQA拆分/重排。

表注：以下为本例逻辑HF输出，未变化和其他rank贡献位置均为NaN；NaN底板不作为全量权重发送。

| HF输出 | 完整shape | 非NaN位置 |
|---|---|---|
| q_proj.weight | [4096,2048] | [2048,7]=0.5 |
| k_proj.weight | [512,2048] | 无 |
| v_proj.weight | [512,2048] | 无 |

提取Q的变化：

```python
flat = outs[q_name].reshape(-1)
hf_idx = (~torch.isnan(flat)).nonzero().view(-1)
hf_val = flat[hf_idx]
# HF index = 2048*2048+7 = 4194311
# values = [0.5]
```

得到的index相对于HF Q张量，不是Megatron分片或推理本地融合权重的index。推理loader再按自己的TP、专家归属和运行时布局加载。

## 8. 与index_copy和推理NaN底板的区别

训练probe本地代码使用`buf.view(-1)[lidx] = lval`，并非直接调用`index_copy_`。在indices唯一有效等前提下，可用`buf.view(-1).index_copy_(0, lidx, lval)`表达相近的覆盖语义。

推理端重建的底板是HF完整shape，再用index_copy_填入HF indices/values，并在masked copy上下文中调用load_weights。两边底板坐标不同：

```text
训练：Megatron本地shape底板 → 为了转换坐标
推理：HF完整shape底板       → 为了复用原生loader且保留未变化值
```

VIME训练侧还会在稀疏snapshot刷新中直接使用`index_copy_`；这是更新CPU基线的操作，与probe转换是另一件事。

## 9. 适用边界与核验

- 转换主要应是拆分、拼接、重排等能保留sentinel意义的操作；跨分片算术混合可能传播NaN，不能直接套用。
- 真实NaN权重不能与“未变化”共用一个标记，须按协议拒绝或另设显式mask方案。
- probe不做真实权重gather，不等于整个同步流程无集合通信：slot元数据交换及后续稀疏合并仍会通信。
- 未知mapping、延迟初始化失败及新的通信模式需要单独适配，不能只因已有megatron_to_hf接口就认为可用。
- Bridge升级或新增模型后，应比较真实完整转换与各rank probe贡献合并的结果，验证名称、shape、覆盖范围和数值逐位一致性。
- 当前VIME对部分已知mapping有直接稀疏坐标快路径，本说明描述通用NaN probe机制，不表示每轮每个参数必定创建完整NaN输出。

## 10. 代码依据与后续阅读

- [VIME导出源码摘录](evidence/vime-export.txt)：`_ProbeGroup`、`_warm_lazy_mappings`、`make_probe`中的`_stub/_inject`、`build_export_index`、`mcore_hf_delta_entry`和snapshot刷新；摘录保留源行号、HEAD、工作树状态及SHA256。
- [verl probe源码摘录](evidence/verl-probe.txt)：相同职责的本地verl实现，具体缓存和快路径与VIME不同。
- [逐步操作示例](nan-probe-example.md)：步骤B展开TP模拟和AutoMapping，步骤C～E连接HF坐标及推理加载。
- [可运行教学脚本](nan_probe_demo.py)：标准库缩小版，验证占位、重排、slot分段及masked copy，不调用真实Bridge。
- [同步流程与接口](verl-sglang-flow-and-interfaces.md)：基线、传输、校验、加载及失败语义。
