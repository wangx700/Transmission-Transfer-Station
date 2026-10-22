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
