# Megatron并行策略：TP、PP、EP、ETP与DP

整理日期：2026-10-24。以Qwen3-30B-A3B为例，解释TP2、PP2、EP2在ETP1与ETP2下的卡数和权重分布。本文是拓扑说明，不表示已验证当前训练环境配置，也不表示指定卡数一定有足够显存。

## 1. 先确定启动进程数，再推导DP

假设一张设备对应一个训练进程，CP1、无VPP及其他额外并行维度，使用支持独立专家TP的Megatron版本。

```text
普通层：world_size = PP × TP × dense-DP
专家层：world_size = PP × ETP × EP × expert-DP
```

TP切分普通层矩阵，EP划分专家集合，ETP切分单个专家的矩阵。普通层和专家层在同一批设备上建立不同的分组，不应直接用TP×PP×EP×ETP计算总卡数。

**ETP2不会自动增加卡数。** 启动命令确定world_size，再按上述公式计算两种DP并检查整除约束。

## 2. 4卡：TP2、PP2、EP2、显式ETP1

```text
普通层：4 = PP2 × TP2  × dense-DP1
专家层：4 = PP2 × ETP1 × EP2 × expert-DP1
```

每个PP stage两张卡。Attention把一个投影矩阵分到两卡；MoE把不同专家集合分到两卡，每个专家自身不切分。

表注：A～D为示意卡号，不代表实际global rank；专家按连续编号分组；PP0示例放全局层0～23，PP1放24～47。

| 卡 | PP | Attention层角色 | MoE层角色 |
|---|---|---|---|
| A | PP0 | TP0 | EP0，完整专家0～63 |
| B | PP0 | TP1 | EP1，完整专家64～127 |
| C | PP1 | TP0 | EP0，完整专家0～63 |
| D | PP1 | TP1 | EP1，完整专家64～127 |

TP2与EP2复用同两张卡，是对不同层的两种分组视角，不是额外两套设备。

## 3. 8卡：TP2、PP2、EP2、ETP2

每个PP stage需要EP2×ETP2=4张卡；两个stage共8张卡。

```text
普通层：8 = PP2 × TP2  × dense-DP2
专家层：8 = PP2 × ETP2 × EP2 × expert-DP1
```

表注：副本0/1表示dense-DP的两个权重副本；它们拥有相同的Attention参数，却拥有不同专家集合。专家0～63与64～127是每一层内部的专家编号，不是跨PP共享同一组参数。卡号仅示意，实际分组取决于rank排列。

| 卡 | PP | Attention层角色 | MoE层角色 |
|---|---|---|---|
| A | PP0 | 副本0、TP0 | EP0、ETP0：专家0～63的前半分片 |
| B | PP0 | 副本0、TP1 | EP0、ETP1：专家0～63的后半分片 |
| C | PP0 | 副本1、TP0 | EP1、ETP0：专家64～127的前半分片 |
| D | PP0 | 副本1、TP1 | EP1、ETP1：专家64～127的后半分片 |
| E | PP1 | 副本0、TP0 | EP0、ETP0：专家0～63的前半分片 |
| F | PP1 | 副本0、TP1 | EP0、ETP1：专家0～63的后半分片 |
| G | PP1 | 副本1、TP0 | EP1、ETP0：专家64～127的前半分片 |
| H | PP1 | 副本1、TP1 | EP1、ETP1：专家64～127的后半分片 |

以PP0为例，示意分组如下：

```text
Attention TP组：{A,B}、{C,D}
Attention DP组：{A,C}、{B,D}
专家 ETP组：   {A,B}、{C,D}
专家 EP组：    {A,C}、{B,D}
专家 DP：      1，无额外权重副本
```

同一对设备可以在Attention层执行DP梯度同步，在MoE层承担EP专家路由；这不意味着它们的专家权重也互为副本。

## 4. “自动得到DP2”的准确含义

```text
启动8个进程，指定TP2、PP2、CP1：
  dense-DP = 8 / (2×2×1) = 2
指定EP2、ETP2：
  expert-DP = 8 / (2×2×2) = 1
```

普通层DP2来自world_size与TP/PP的商，不是设置ETP后系统自动扩容。

如果只启动4个进程并指定ETP2、EP2、PP2，则4不能被8整除，配置不合法；不会自动补成8个进程。ETP1与ETP2也不是简单的“关闭/开启”开关，而是单专家张量并行度的不同数值。

## 5. Qwen3-30B-A3B单专家权重shape

H=2048，单专家intermediate=768。fc1融合gate与up，完整shape为[1536,2048]；fc2为down，完整shape为[2048,768]。

表注：以下是每卡、每个本地专家的逻辑权重shape，不包含物理布局或padding；fc1切分gate/up各自的输出行，fc2切分输入列。

| ETP | 单专家fc1 | 单专家fc2 | 分片解释 |
|---|---|---|---|
| 1 | [1536,2048] | [2048,768] | 单卡持有完整专家 |
| 2 | [768,2048] | [2048,384] | 每卡gate384行+up384行，down384列 |

EP2决定每卡负责哪64个专家；ETP决定这些专家的每个矩阵是否进一步切分。Attention在这两个示例中都是TP2，因此QKV本地shape仍为[2560,2048]。

模型维度依据：[Qwen官方配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。更多name/shape与坐标示例见[稀疏同步文档](../qwen3-30b-megatron-hf-sparse-sync/README.md)。

## 6. 默认值及接口代码依据

官方Megatron `initialize_model_parallel`先读取分布式world_size，建立普通层rank生成器，再建立专家rank生成器。专家生成器的tp取ETP、ep取EP；两者的PP分组需要一致。

官方代码中，`expert_tensor_parallel_size=None`时默认继承普通TP。因此不能将“未显式配置ETP”当成ETP1。以下是去除其他并行维度后用于解释的等价逻辑，非原文件逐行摘录：

```python
world_size = torch.distributed.get_world_size()
dense_dp = world_size // (tp * pp)  # 本文CP=1
if etp is None:
    etp = tp
expert_model_size = etp * ep * pp
if world_size % expert_model_size != 0:
    raise RuntimeError('world_size cannot form expert parallel groups')
expert_dp = world_size // expert_model_size
# 建立普通层和专家层各自的RankGenerator，再创建TP/DP/EP/ETP组。
```

代码依据：[NVIDIA/Megatron-LM parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py)，重点查看`initialize_model_parallel`中的普通层rank生成器、`expert_tensor_parallel_size is None`分支、专家整除校验、专家rank生成器及PP分组一致性断言。链接main会变化，以上说明核对日期为2026-10-24，不代表用户部署版本已经核对。

使用4卡方案应显式设置ETP1，并确认Megatron、Bridge及Ascend适配版本支持TP与ETP不同。上述整除关系是基础拓扑约束，不替代具体版本的rank排列、通信后端、模型配置和显存检查。本文未更改运行配置或进行多卡训练测试。
