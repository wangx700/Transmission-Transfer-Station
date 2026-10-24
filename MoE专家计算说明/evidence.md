# 本地代码依据

摘录日期：2026-10-24。下列摘录来自本地`pr447_sync`工作树的`vime/backends/megatron_utils/alignment/deepgemm_moe_forward.py`；这里只证明该实现的计算和命名方式，不据此声称其DeepGEMM专用执行路径支持ETP2。ETP2章节给出数学分解。

## SwiGLU计算

函数：`_swiglu_forward_chunked(gate, up)`，源文件约201～219行。关键代码：

```python
gate_f = gate[start:end].float()
up_f = up[start:end].float()
silu_gate = F.silu(gate_f)
down_input[start:end].copy_((silu_gate * up_f).to(dtype=gate.dtype))
```

含义：对gate应用SiLU，再与up逐元素乘，作为down投影的输入。

## grouped-linear本地权重命名

函数：`_get_expert_weights(grouped_linear, *, num_local_experts, expected_shape, module_name)`，源文件约1163～1185行。关键代码：

```python
weights = []
for expert_index in range(num_local_experts):
    attr_name = f"weight{expert_index}"
    weight = getattr(grouped_linear, attr_name, None)
    if not isinstance(weight, torch.Tensor):
        raise RuntimeError(f"{module_name} is missing tensor parameter {attr_name}")
    if tuple(weight.shape) != expected_shape:
        raise RuntimeError(...)
    weights.append(weight)
return weights
```

含义：weight后缀是该模块的本地expert_index，weight6是合法的本地第6号专家参数命名。省略号表示删除了错误消息细节，非可独立运行代码。

同文件约1272～1273行还构造：

```python
named_weights = [(f"linear_fc1.weight{i}", weight) for i, weight in enumerate(fc1_weights)]
named_weights.extend((f"linear_fc2.weight{i}", weight) for i, weight in enumerate(fc2_weights))
```

## Bridge导出接口

已有[导出证据摘录](../qwen3-30b-megatron-hf-sparse-sync/evidence/vime-export.txt)记录`build_export_index`、`get_conversion_tasks`和`probe.megatron_to_hf`路径，并含该次摘录的HEAD及源文件SHA256。参照其接口区分本地参数与全局转换任务。

实际参数前缀、专家编号规范化行为应在当前运行模型中读取确认；本文不将某个旧转换器按weight后缀拼HF名称的规则，作为所有EP场景的本地编号映射保证。
