# NaN probe：从一个本地变化值到HF全局坐标

## 1. Qwen3-30B-A3B真实维度例子

配置与[主文档](README.md)一致：TP2、Q heads32、KV heads4、head_dim128、H2048。以下以典型Megatron GQA分组排列说明，不是把vLLM的融合布局直接当作Megatron布局。

目标：只有训练TP1的 `linear_qkv.weight[0,7]` 从旧值变为 **0.5**。diff阶段已经得到本地flat index和当前绝对值：

```python
local_shape = (2560, 2048)
local_indices = tensor([7], dtype=int64, device='npu')
local_values = tensor([0.5], dtype=bfloat16, device='npu')
```

### 步骤A：创建本地NaN底板，写入变化值

```python
buf = torch.full(local_shape, float('nan'), dtype=torch.bfloat16, device=device)
buf.view(-1).index_copy_(0, local_indices, local_values)
# 只有buf[0,7]=0.5，其余全NaN
```

底板shape是本地Megatron分片shape，不是全量QKV shape。

### 步骤B：probe替换通信，保留真实TP几何关系

真实mapping如果调用`gather_from_tp_ranks(buf)`，probe不发集合通信，而在TP1本地返回：

```python
[
    nan_block(shape=(2560,2048)),  # TP0未知贡献全部NaN
    buf,                          # 本rank位于真实TP rank=1的位置
]
```

TP0运行自己的probe时则返回`[buf_tp0, nan_block]`。两者分别导出自己的贡献；没有把其他rank的真实权重拉过来。

mapping仍看到`tp_size=2`、`tp_rank=1`。不能简单把TP size改为1，否则某些转换的维度计算和反交错会错误。递归复制mapping还需要处理AutoMapping的内部delegate。

证据：[make_probe通信stub与递归复制](evidence/vime-export.txt)。

#### B.1 正常gather与本地模拟gather

正常mapping调用`gather_from_tp_ranks`会获取真实分片列表：

```text
[TP0真实分片, TP1真实分片]
```

probe替换后的函数则只在本设备构造列表。下面为等价示意，不是生产代码的逐行摘录：

```python
def gather_from_tp_ranks(tensor):
    nan = nan_block(tensor.shape, tensor.dtype, tensor.device)
    out = [nan] * tp_size
    out[tp_rank] = tensor
    return out
```

列表长度、每块shape和本rank所在的位置与正常gather保持一致，但没有取回其他rank的真实数据。NaN占位同时表示其他rank的未知贡献；本地tensor中的NaN则表示本rank未变化的位置。

#### B.2 为什么TP1必须放在列表位置1

在本例中，TP0负责HF Q行[0,2048)，TP1负责[2048,4096)。TP1的模拟结果必须是`[NaN, local]`；如果写成`[local, NaN]`，mapping可能把该贡献映射到TP0负责的HF位置。

```text
正确：TP1本地Q位置 → HF Q全局行2048以后
错误：把TP1分片放到位置0 → 可能错误落到HF Q前半部分
```

因此NaN块不仅补齐shape，也保留了各rank在全局张量中的位置关系。实际Megatron QKV还包含query-group排列，具体全局坐标继续由真实mapping计算，不能简单把所有本地flat index统一加一个常数。

#### B.3 为什么不能将tp_size改成1

`tp_rank=1`表示TP组中第二个rank；`tp_size=1`表示整个TP组只有一个rank，两者含义不同。

若将size改为1，mapping会按单rank布局计算，可能认为本地分片已经是完整输入，从而改变拼接、reshape、head/group维度和反交错规则。

probe要做到的是：**保留TP2的转换规则，只将其他rank的数据替换为NaN**。这里也不要求每种mapping都显式读取rank字段；即便它只读取size或gather列表，仍需要保留真实布局信息。

```text
tp_size=2、tp_rank=1
  → 模拟[NaN分片, 本rank底板]
  → 按真实TP2规则转换
  → 输出完整HF shape，其中只有本rank变化位置非NaN
```

#### B.4 为什么要处理AutoMapping内部delegate

**白话理解：AutoMapping是“分派员”，delegate是它选中的“实际执行者”。** 分派员先看这个参数怎么切分，再选一个具体转换对象执行。delegate不是额外的rank、线程或通信进程，只是保存在AutoMapping内部的一个Python对象，通常放在`_mapping`属性里。

例如，参数按输出维切分，就选择ColumnParallelMapping；按输入维切分，选择RowParallelMapping；每个rank都有完整副本，选择ReplicatedMapping。具体识别规则以所用Bridge版本和module属性为准。

下面为解释用的伪代码，不是完整接口实现：

```python
class AutoMapping:
    def __init__(self):
        self._mapping = None  # 暂时还没选具体执行者

    def megatron_to_hf(self, weight, module):
        if self._mapping is None:
            kind = self._detect_parallelism_type(module)
            self._mapping = self._get_or_create_mapping(kind)

        # AutoMapping做选择，具体转换交给选中的对象。
        return self._mapping.megatron_to_hf(weight, module)
```

第一次调用才选择、创建执行对象，之后继续复用，这就是“延迟初始化”。`delegate`只是“被委托执行工作的对象”的叫法，不需要额外理解成一种新的并行机制。

一个mapping可能包含多层委托，例如：

```text
QKVMapping
  → _tp_mapping：AutoMapping
      → _mapping：Column/Row/Replicated等实际实现
          → gather_from_tp_ranks(...)
```

只替换最外层对象的方法不够，真实通信可能由内部实现执行。若内部对象仍使用真实进程组，probe就可能意外触发集合通信。

仍用白话说：只告诉分派员“不要去其他卡取数据”不够；真正出去取数据的执行者也必须改成“在本卡拿NaN占位”。Python对象的方法替换不会自动传递给它引用的其他对象。

此外AutoMapping可能在第一次调用时才创建内部实现。若递归复制时delegate还不存在，复制过程就无法覆盖它；随后延迟创建的delegate可能重新取得真实进程组。

```text
复制时：_mapping还是None → 没有内部对象可以复制、替换
运行时：第一次转换才创建实际mapping → 新对象可能仍走真实通信
```

因此需要先让执行者“就位”，再给它制作probe版本。`_warm_lazy_mappings`的warm在这里是提前初始化内部对象，不是先跑一次真实权重gather。

本地实现的处理顺序是：

```text
_warm_lazy_mappings(mapping, module)
  → 确保可识别的AutoMapping实际delegate已创建
  → copy.copy(mapping)
  → 替换当前副本的进程组引用和gather方法
  → 遍历内部MegatronParamMapping对象
  → 对子mapping重复warm、复制、替换
```

这些替换作用于probe副本，正常导出的mapping通信语义仍需保留。组替身`_ProbeGroup`提供真实size/rank；若某种未处理的通信操作直接访问该替身，应报错而不是悄悄按错误布局转换。warm失败会产生警告，不能把它理解成任意新mapping都能自动安全兼容。

还有一个容易遗漏的点：`copy.copy()`只做浅复制。复制了外面的对象，不等于里面引用的对象也自动复制了一份。

```text
只浅复制外层时，可能是：
原始AutoMapping ─┐
                 ├→ 同一个内部ColumnParallelMapping
probe外层副本 ──┘

逐层复制并替换引用后，才是：
原始AutoMapping → 原始ColumnParallelMapping（正常通信）
probe副本      → ColumnParallelMapping副本（NaN占位模拟）
```

所以“递归复制”就是：顺着内部mapping一层层找下去，每层都复制一份，把父副本指向对应的子副本，并替换各副本的通信方法。不要直接修改原来的子对象，否则正常全量导出也可能被影响。提前warm会初始化原对象的延迟缓存，但通信替换应作用在probe副本上。

总结这段接口逻辑：**先知道谁真正干活，让它提前创建出来，再逐层制作probe副本；保留TP布局规则，将副本的跨卡取数替换成本地NaN占位。**

证据文件中对应`_ProbeGroup`、`_warm_lazy_mappings`、`make_probe`以及内部`_stub`、`_inject`函数。

#### B.5 完整接口关系与适用边界

```text
local_indices + current_values
  → 填入本地Megatron shape的NaN底板
  → rec.probe.megatron_to_hf(buf, rec.module)
  → 内部gather调用被替换为本地列表合成
  → 原mapping继续执行真实TP布局的拆分/重排
  → 提取HF输出中非NaN的位置和值
```

probe本身不取回其他rank的真实权重，但初始化slot元数据交换和后续稀疏条目合并仍有通信。该方法依赖转换主要是拆分、拼接、重排；若跨分片做算术混合，NaN可能传播，需另行适配并做真实转换与probe合并的差分验证。

### 步骤C：运行真实mapping的拆分/重排

```python
outs = rec.probe.megatron_to_hf(buf, rec.module)
```

本例全局有4个query group，每组8个Q heads、1个K head、1个V head：

```text
每组行数 = (8+1+1)×128 = 1280
TP0负责全局group0、1
TP1负责全局group2、3

TP1的本地row0 → group2的第一个Q行
              → HF Q全局row = 2×8×128 = 2048
```

输出是逻辑完整HF shape，但仅有本rank变化位置非NaN：

表注：这里只放入一个Q变化；K/V没有变化，因此全NaN。probe输出shape不是“本rank HF小张量shape”。

| HF输出 | shape | 非NaN位置 |
|---|---|---|
| q_proj.weight | [4096,2048] | [2048,7]=0.5 |
| k_proj.weight | [512,2048] | 无 |
| v_proj.weight | [512,2048] | 无 |

### 步骤D：扫描非NaN，生成五元组，再追加group

```python
flat = outs[q_name].reshape(-1)
hf_idx = (~torch.isnan(flat)).nonzero().view(-1)
hf_val = flat[hf_idx]
# hf_idx=[4194311]，hf_val=[0.5]
# 4194311 = 2048*2048+7

entry = (
    [(q_name,(4096,2048)), (k_name,(512,2048)), (v_name,(512,2048))],
    'bfloat16', [1,0,0], [4194311], [0.5], WORLD,
)
```

slot顺序仅为解释固定成Q/K/V，生产代码按缓存目录排序。合并阶段只传稀疏结果，NaN底板不作为完整张量发送。

### 步骤E：推理侧重建HF patch并加载

```python
patch = torch.full((4096,2048), float('nan'), dtype=dtype, device=device)
patch.view(-1).index_copy_(0, hf_idx, hf_val)
with masked_copy_context:
    model.load_weights([(q_name, patch)])
```

推理TP1的loader切取HF Q行`[2048,4096)`，将HF `[2048,7]`转换为本地`qkv_proj[0,7]`。最终masked copy只覆盖该元素。TP0的对应Q切片没有非NaN元素，保持原权重。

## 2. MoE补充：EP编号和ETP偏移一起转换

同样先diff再probe。EP1的本地专家6对应全局专家70，ETP1的fc1包含gate后384行和up后384行。

```text
Megatron fc1 local shape = [768,2048]
local[0,7] = 0.75
  → 本地NaN底板只有该项非NaN
  → ETP1 synthetic gather：[NaN shard, local shard]
  → mapping拆分gate/up，并绑定全局专家号70
  → HF experts.70.gate_proj[384,7] = 0.75
  → flat index = 384×2048+7 = 786439
```

EP stub只返回本rank拥有的专家名，不获取其他EP rank的真实专家权重。slot表通过元数据交换得到全局一致目录；不属于本rank的slot count为0。

## 3. 可运行缩小版

运行：

```bash
python nan_probe_demo.py
```

只依赖Python标准库。为了直接观察，缩小为H4、Q heads4、KV heads2、head_dim2、TP2，每rank一个query group。本例TP1放入两个变化：本地Q `[0,1]=0.5`、本地K `[4,2]=0.7`。

预期输出：

```text
q_proj: shape=(8,4), entries=[(17, 0.5)]
k_proj: shape=(4,4), entries=[(10, 0.7)]
v_proj: shape=(4,4), entries=[]
counts=[1, 1, 0], hf_idx=[17, 10], hf_val=[0.5, 0.7]
PASS: probe坐标、slot分段、masked copy保留未变化值
```

它验证NaN占位、真实rank位置、GQA重排、slot分段及masked copy；不调用Megatron-Bridge，也不执行多卡collective。生产路径的精确性仍需真实mapping差分测试。

## 4. 不能省略的边界

- 真实权重NaN必须拒绝，因为它与“未更新”的sentinel冲突。
- mapping如果对不同分片做算术混合，NaN会传播，不能依靠这个probe自动恢复贡献。
- probe不发真实权重gather，不等于整个同步流程无通信：目录初始化和稀疏条目合并仍有通信。
- 当前VIME对已知mapping可用稀疏坐标快路径；此示例解释的是通用NaN probe路径。
