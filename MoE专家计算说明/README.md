# Qwen3-30B-A3B：MoE gate/up/down计算与专家权重命名

整理日期：2026-10-24。以非量化Qwen3-30B-A3B为例，说明单专家计算、ETP切分、MoE路由和Megatron grouped-linear的weight编号。本目录仅记录计算和代码阅读，不代表本次完成多卡训练验证。

模型维度：hidden size H=2048，单专家中间维度I=768，共128个专家，每token选择8个专家。依据：[Qwen官方配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。

## 1. 两种gate不是同一模块

- **router gate**：为token选择专家，并计算专家输出的组合权重，HF通常名为`model.layers.L.mlp.gate.weight`，逻辑shape为[128,2048]。
- **专家gate_proj**：单个专家内部SwiGLU的一个投影分支，HF名为`model.layers.L.mlp.experts.E.gate_proj.weight`，shape为[768,2048]。

router负责“交给哪些专家”，gate_proj参与“这个专家内部怎么算”。

## 2. 单专家gate/up/down计算

设送到专家e的token矩阵为`X:[T,2048]`，T表示本次分发给该专家的token数，不是固定batch size。以下按PyTorch权重[out_features,in_features]约定，忽略bias。

表注：shape为完整单专家的逻辑权重shape，未做ETP切分，不包含设备物理格式或padding。

| 权重 | shape | 计算 | 输出shape |
|---|---|---|---|
| gate_proj.weight | [768,2048] | G = X @ W_gate.T | [T,768] |
| up_proj.weight | [768,2048] | U = X @ W_up.T | [T,768] |
| down_proj.weight | [2048,768] | Y = Z @ W_down.T | [T,2048] |

```python
G = X @ W_gate.T
U = X @ W_up.T
Z = silu(G) * U       # 逐元素相乘，不是矩阵乘法
Y = Z @ W_down.T

# silu(g) = g * sigmoid(g)
```

```text
                  gate_proj → SiLU ──┐
输入X ────────────                    × → down_proj → 输出Y
                  up_proj ───────────┘
```

gate分支对up分支的中间特征进行调制，down再映射回hidden维度。`up`是投影名称，不能理解为必然升维；这个模型的单专家是2048→768。

本地代码`_swiglu_forward_chunked`明确执行`F.silu(gate_f) * up_f`，在FP32临时量中计算后写回输入dtype。见[代码证据](evidence.md#swiglu计算)。

## 3. Megatron的fc1、fc2与HF的三个投影

```python
W_fc1 = torch.cat([W_gate, W_up], dim=0)  # [1536,2048]
W_fc2 = W_down                          # [2048,768]

GU = X @ W_fc1.T                        # [T,1536]
G, U = GU.chunk(2, dim=-1)               # 各[T,768]
Y = (silu(G) * U) @ W_fc2.T              # [T,2048]
```

因此：

```text
linear_fc1 = gate_proj + up_proj（沿输出维融合）
linear_fc2 = down_proj
```

gate/up融合使一次矩阵乘法同时计算两个分支，不是将两份权重相加。以上为逻辑排列，具体内核的打包、转置或交错应通过mapping/loader处理。

HF参数举例：

```text
model.layers.0.mlp.experts.70.gate_proj.weight  [768,2048]
model.layers.0.mlp.experts.70.up_proj.weight    [768,2048]
model.layers.0.mlp.experts.70.down_proj.weight  [2048,768]
```

## 4. ETP2如何计算同一个专家

ETP把专家的768维中间特征切为两份，每份384维。每个rank都计算gate和up，不能理解成rank0只算gate、rank1只算up。

表注：前/后半指gate与up各自的输出行、down的输入列；T按逻辑表示对应token集合，实际token/序列布局及通信随实现变化。

| rank | 本地fc1 | 本地fc2 | 中间特征 |
|---|---|---|---|
| ETP0 | gate前384行+up前384行，[768,2048] | down前384列，[2048,384] | Z0:[T,384] |
| ETP1 | gate后384行+up后384行，[768,2048] | down后384列，[2048,384] | Z1:[T,384] |

```python
Z0 = silu(X @ W_gate0.T) * (X @ W_up0.T)
Y0 = Z0 @ W_down0.T  # [T,2048]，部分和

Z1 = silu(X @ W_gate1.T) * (X @ W_up1.T)
Y1 = Z1 @ W_down1.T  # [T,2048]，部分和

Y = Y0 + Y1
```

输出需归约部分和。具体是all-reduce还是配合序列并行的reduce-scatter等，由所用实现决定；上面是数学计算关系，不是完整通信调度代码。

ETP1则单卡持有该专家完整fc1 [1536,2048]、fc2 [2048,768]。EP划分专家集合，ETP划分每个专家内部的矩阵；卡数关系见[并行策略说明](../并行策略说明/README.md)。

## 5. 整个MoE如何汇总

```text
输入token x
 → router计算128个专家的路由分数
 → top-8选择及路由权重计算
 → token分发到被选专家
 → 每个专家执行gate/up/down
 → 专家输出按路由权重加权求和
```

对一个token：

```text
y(x) = Σ p_e(x) · Expert_e(x)，e属于该token选出的8个专家
```

`p_e`表示最终路由权重。这里没有将专家计算中的逐元素门控与router概率混为一谈。

- ETP归约：合并同一个专家的矩阵分片结果。
- MoE加权求和：合并不同专家的输出。

两者是不同层次，不能相互替代。

## 6. weight6这个名字是否正确

**在本地TE grouped-linear参数实现中，weight6这种名字存在，表示该grouped-linear模块的本地专家索引6。** 本地代码通过`getattr(grouped_linear, f"weight{expert_index}")`获取权重，且检查[out_features,in_features] shape。

```text
decoder.layers.0.mlp.experts.linear_fc1.weight6
  → 对应本地专家6的gate+up融合权重

decoder.layers.0.mlp.experts.linear_fc2.weight6
  → 对应同一本地专家6的down权重
```

这两个名字的完整层号/前缀仅为示例，实际以运行时`named_parameters()`和Bridge任务为准；不能将它推广为所有Megatron MoE实现的统一命名。有些实现用local_experts子模块或堆叠参数。

### 本地编号与全局编号

连续划分128个专家、EP2时：

表注：全局专家号属于单个Transformer层；本表假设每EP rank连续拥有64个专家，非任意专家放置策略的通式。

| EP rank | 全局专家集合 | 本地expert6对应HF全局专家 |
|---|---|---:|
| EP0 | 0～63 | 6 |
| EP1 | 64～127 | 70 |

**本地参数名后缀、Bridge规范化后的全局转换名和HF专家号不能混同。** 例如某些HF读取转换器直接把传入规范化名字的weight后缀当成HF全局expert号；这不证明原始本地weight6在EP1也应读取HF expert6。进入该转换器前是否做全局编号转换，必须沿调用链核对。

在Bridge导出路径中应使用：

```text
bridge.get_conversion_tasks(model)
  → task.param_weight：实际本地参数
  → task.global_param_name：转换任务的全局身份
  → task.mapping + task.megatron_module
  → probe.megatron_to_hf(...)
  → HF全局专家name和坐标
```

不要仅凭字符串weight6手工拼接HF experts.6。EP1本地专家6映射到HF专家70的示例，以Bridge实际任务输出和专家分配为最终依据。

更多HF坐标、slot与NaN probe示例见[稀疏同步说明](../qwen3-30b-megatron-hf-sparse-sync/README.md)。
