# verl + SGLang：修正版同步流程与接口代码

资料日期：2026-10-22。依据本地 verl 工作树；这里只记录代码行为，不把设计建议当成已有能力。本文对应 CUDA 路径，NPU/H2D 是迁移至 Ascend 后的对应实现。

## 1）先全量同步HF权重，建立共有基线

首次通过Megatron-Bridge正常导出接口，将训练权重转换成完整HF权重，发送至推理侧加载。训练后端建立并缓存转换任务、NaN probe和HF slot目录；缓存可能按需创建，不要求与首次网络发送同时执行。

NaN probe保留真实TP size/rank，将其他rank的贡献替换为NaN占位，独立把本rank的变化映射到HF全局坐标，无需先gather完整Megatron权重。probe本身不进行真实权重gather；初始化元数据交换和后续稀疏条目合并仍有集合通信。

全量同步后，各训练rank在CPU保存自己的 **Megatron本地分片snapshot**，不是每个rank一份完整HF权重。参数在全量导出、发送和snapshot初始化期间需要保持不变，才能构成同一基线。

接口调用关系：

```text
DeltaCheckpointEngine.send_weights(engine)
  → engine.get_per_tensor_param()
  → _send_full_seed(full, global_steps)
  → engine.prime_delta_snapshots()
      → get_per_tensor_param_shard()
      → utils.prime_delta_snapshots(gen, snaps, pin)
```

以下为`send_weights`首次分支的核心摘录，删除了日志、注释等非关键部分；完整接口见[checkpoint engine证据](evidence/verl-checkpoint-interface.txt)。

```python
async def send_weights(self, engine, global_steps: int | None = None):
    if not self._shard_seeded:
        full, _ = engine.get_per_tensor_param()
        metrics = self._send_full_seed(full, global_steps)
        engine.prime_delta_snapshots()
        self._shard_seeded = True
        return metrics
    # 后续分支继续处理delta，不是这里直接返回。
```

CPU snapshot创建接口，见[完整摘录](evidence/verl-diff.txt)：

```python
def prime_delta_snapshots(gen, snaps: dict, pin: bool) -> None:
    for name, local, _spec in gen:
        local = local.detach().contiguous().view(-1)
        snap = snaps.get(name)
        if snap is None or snap.numel() != local.numel():
            snap = torch.empty_like(local, device='cpu', pin_memory=pin)
            snaps[name] = snap
        snap.copy_(local, non_blocking=True)
```

`pin`由训练后端决定，不应统一写成“所有snapshot都是pinned CPU内存”；本地Megatron后端设置`delta_pin_snapshots=False`。

## 2）先Megatron shard diff，再probe，最后HF patch

### 2.1 diff及snapshot更新

CPU snapshot传到当前训练设备，与Megatron本地分片按位比较，返回本地flat indices及当前权重绝对值。随后才调用probe转换。

```text
get_per_tensor_param_delta_shard()
  → hf_delta_export(gen, self._delta_shard_snap, self._hf_delta_entry)
      → CPU snapshot H2D
      → shard_delta_indices(local, base, 0)
      → snapshot刷新
      → _hf_delta_entry(...)
          → mcore_hf_delta_entry(...)
```

下面为`hf_delta_export`核心摘录；省略了placement推导和snapshot存在性断言，不能将其单独当成完整实现复制使用。

```python
def hf_delta_export(gen, snaps: dict, entry_fn):
    for name, local, spec in gen:
        local = local.detach().contiguous().view(-1)
        snap = snaps.get(name)
        # 原实现先根据spec推导place、contributes、pg。
        if contributes:
            base = snap.to(local.device, non_blocking=True)
            lidx, lval = shard_delta_indices(local, base, 0)
        else:
            lidx = torch.empty(0, dtype=torch.int64, device=local.device)
            lval = torch.empty(0, dtype=local.dtype, device=local.device)
        snap.copy_(local, non_blocking=True)
        yield (*entry_fn(name, spec, place, lidx, lval), pg)
```

**现有代码在生成增量时就刷新snapshot，并非收到推理侧应用成功确认后刷新。** `snap.copy_(local)`在此是完整本地分片回写，不是只写变化元素。不能把VIME的稀疏snapshot刷新优化当成这份verl已有实现。

因此不具备“失败后直接用原基线重试”的事务语义。延迟提交snapshot、增加base_version/target_version/update_id及接收端确认，是可增加的设计，不是上述接口已提供的保证。

证据：[diff及snapshot接口](evidence/verl-diff.txt)、[Megatron后端接线](evidence/verl-megatron-interface.txt)。

### 2.2 NaN probe接口

核心接口签名：

```python
make_probe(mapping, module)
build_export_index(bridge, megatron_model, slot_cache: dict | None = None)
mcore_hf_delta_entry(rec, _place, lidx, lval, slot_cache)
```

`make_probe`复制mapping树并替换通信；`build_export_index`枚举Bridge转换任务、构建probe和slot目录；`mcore_hf_delta_entry`执行本地稀疏变化到HF坐标的转换。

以下为`mcore_hf_delta_entry`核心操作摘录，省略空delta、slot缓存和校验分支：

```python
buf = torch.full(tuple(rec.param.shape), float('nan'),
                 dtype=lval.dtype, device=lval.device)
if lidx.numel():
    buf.view(-1)[lidx] = lval
outs = rec.probe.megatron_to_hf(buf, rec.module)

# 每个HF输出slot分别提取坐标和值。
fl = out.reshape(-1)
p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
counts[s_i] = p_.numel()
idx_pieces.append(p_.to(torch.int32))
val_pieces.append(fl[p_])
# 实际实现还处理空输出、拼接等，最终返回：
# slots, dtype_str, counts, hf_idx, hf_val
```

外层再追加进程组形成六元组：

```python
(slots, dtype_str, counts, hf_idx, hf_val, gather_group)
```

顺序是 **Megatron shard diff → probe → HF全局坐标**，不是先转完整HF权重再做HF diff。`values`是当前绝对值，接收端覆盖而不做加法。

证据：[verl原版probe代码](evidence/verl-probe.txt)。操作细节见[NaN probe示例](nan-probe-example.md)，Qwen3 name/shape与六元组见[主文档](README.md)。

### 2.3 合并和wire接口

`send_weights`后续分支从后端消费六元组，经gather队列、bucket编码后发布flush：

```python
weights, _ = engine.get_per_tensor_param_delta_shard()
# 原代码先建立bucket和_GatherQueue。
for slots, dtype_str, counts, hf_idx, hf_val, pg in weights:
    gq.put(pg, slots, dtype_str, counts, hf_idx, hf_val)
gq.flush_all()
# 随后seal/emit尾bucket，处理可选verify sweep和终止标记。
```

表注：这些名称属于SGLang自定义loader接收的named tensors；JSON以tensor传输，不要求生成落盘manifest.json。positions偏移按字节，values偏移按元素。

| tensor名称 | 内容 |
|---|---|
| `__delta_spec__` | uint8承载JSON：encoding、params及checksum等 |
| `__positions__` | uint8位置blob，indices编码为int32位置；dense seed不需要 |
| `__values__` | 当前flush统一dtype的权重值 |

params包含name、shape、dtype、count及pos_start/pos_end、val_start/val_end等字段。group对象只在训练侧控制合并，不发送到推理端。

证据：[checkpoint engine发布接口](evidence/verl-checkpoint-interface.txt)、[SGLang wire契约](evidence/sglang-loader.txt)。

## 3）SGLang自定义loader应用patch

入口注册方式：

```text
custom_weight_loader =
  ["verl.workers.rollout.sglang_rollout.delta_loader.apply_delta"]
```

训练发送时使用对应的load_format导入路径，SGLang在各TP worker内调用：

```python
apply_delta(model: torch.nn.Module,
            named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None
```

### 3.1 解析和checksum校验

核心摘录如下，见[完整loader证据](evidence/sglang-loader.txt)：

```python
tensors = dict(named_tensors)
spec = json.loads(bytes(tensors['__delta_spec__'].cpu().numpy().tobytes()).decode())
values = tensors['__values__']
positions = tensors.get('__positions__')
if positions is None:
    positions = torch.empty(0, dtype=torch.uint8, device=values.device)
got = _checksum(positions, values)
if got != int(spec['checksum']):
    raise RuntimeError('delta checksum mismatch ...')
```

准确结论是“已有positions和values的checksum校验”。不能据此声称所有shape/dtype/count、索引边界、版本关系都已经完整校验，也不能把payload checksum描述为manifest全部字段的完整性保护。

### 3.2 dense和sparse分流

首次dense全量seed走`_apply_dense`，直接从values切片、reshape后调用`model.load_weights`，不需要NaN底板和masked copy。开启verify的dense sweep另走`_verify_dense`，不应混同普通首次seed。

sparse路径调用关系：

```text
apply_delta(model, named_tensors)
  → _masked_copy()上下文
  → 逐参数_decode_one(encoding, positions, values, p)
  → 按CHUNK_BYTES累计
  → model.load_weights(chunk)
  → finally恢复Tensor.copy_
```

`CHUNK_BYTES = 512 << 20`是分批加载目标，不是硬内存上限；单个HF张量仍完整展开，而且当前实现先decode下一张量再判断是否flush已有chunk，峰值还受已有chunk、当前张量和临时量影响。

### 3.3 NaN底板和index_copy_

下面为`_decode_one`的indices路径核心摘录，省略零元素和不支持encoding分支：

```python
numel = math.prod(p['shape'])
dtype = getattr(torch, p['dtype'])
flat = torch.full((numel,), float('nan'), dtype=dtype, device=values.device)
vals = values[p['val_start']:p['val_end']]
pos_b = positions[p['pos_start']:p['pos_end']]
idx = pos_b.view(torch.int32).to(torch.int64)
flat.index_copy_(0, idx, vals.to(dtype))
return flat.view(p['shape'])
```

### 3.4 masked copy上下文

下面保留原实现的关键逻辑：

```python
@contextmanager
def _masked_copy() -> Iterator[None]:
    orig_copy = torch.Tensor.copy_

    def masked_copy_(self, src, *args, **kwargs):
        if isinstance(src, torch.Tensor) and src.is_floating_point() and self.shape == src.shape:
            cast = src.to(self.dtype)
            return orig_copy(self, torch.where(torch.isnan(cast), self, cast))
        return orig_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = masked_copy_
    try:
        yield
    finally:
        torch.Tensor.copy_ = orig_copy
```

原生loader负责名称映射、融合参数处理、TP切片；最终同shape的copy只覆盖来源非NaN位置。这个上下文替换的是进程级Tensor.copy_，不是单个model方法；需隔离并发加载。自定义loader、多阶段原地变换和量化路径是否兼容，仍应按模型验证。

## 4）与VIME + vLLM + vLLM-Ascend的对应关系

本节为迁移对照，不将两份实现混为同一套接口。

表注：左列是本文verl/SGLang CUDA接口，右列是本目录本地VIME/Ascend证据中对应职责；不表示名称或失败恢复行为完全相同。

| 职责 | verl + SGLang | VIME + vLLM + Ascend |
|---|---|---|
| diff基线 | Megatron shard CPU snapshot | 同样本地Megatron shard CPU snapshot |
| diff设备 | CUDA | NPU |
| 坐标转换 | mcore_hf_delta_entry + probe | 相同职责，增加部分直接稀疏映射快路径 |
| snapshot回写 | hf_delta_export完整shard回写 | 本地分支有变化项批量回写优化 |
| 传输 | delta checkpoint engine wire | sparse HCCL |
| 接收入口 | apply_delta | SparseHCCLWeightTransferEngine.receive_weights |
| 应用 | model.load_weights + _masked_copy | load_checkpoint_weight_patches + loader view |

详细边界、代码证据和优化建议见[主文档第6～7节](README.md#6-全量基线diff发送与应用接口)。本次仅新增说明和证据，没有修改训练或推理实现。
