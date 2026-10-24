# 优化候选：接口、时序与可行性

候选基于 #492 `b7a462f5`，复用 #480 `1e46fc61` 的 sparse 相关依赖段。下图是候选代码对应的逻辑顺序；没有绘制时间比例，也不表示候选已完成多卡实测。

## 整体时序

```mermaid
sequenceDiagram
    autonumber
    participant A as VIME Actor
    participant U as Sparse Updater
    participant M as All Megatron Ranks
    participant T as HCCL Sender rank0
    participant W as All Inference Workers
    participant L as Checkpoint Loader
    A->>U: connect_rollout_engines
    par 初始化控制面
        T->>W: init_weight_transfer_engine(rank_offset=1)
    and 初始化数据面
        T->>T: HCCL trainer_init(rank=0)
    end
    Note over T,W: 单推理组为 1+TP*PP*DP；EP不额外相乘
    A->>A: offload_train 时唤醒训练权重
    A->>U: update_weights
    U->>W: pause_generation / flush_cache / start_weight_update
    U->>M: build_export_index / validate_export_weights
    alt 首次或重连后的 dense seed
        U->>M: HfWeightIteratorDirect 全量导出
        M-->>T: 所有rank参与导出，仅rank0发送
        loop 每个全量chunk
            par 元数据RPC
                T->>W: update_weights(dense=true)
            and HCCL广播
                T->>W: 完整values，无indices
            end
            W->>W: 完整接收后过滤非本PP参数
            W->>L: dense CheckpointWeightPatch
            L->>L: loader view / native load / restore view
            W-->>T: 应用与设备同步完成
        end
        U->>M: prime_delta_snapshots
    else 后续 sparse steady update
        M->>M: diff / compact / HF checkpoint坐标映射
        M->>T: 一致collective顺序批量gather
        T->>T: 按dtype与字节预算分桶
        loop 每个sparse chunk
            T->>T: 结构检查 / 按设备批量数值校验 / 规范化
            par 元数据RPC
                T->>W: update_weights(names/shapes/counts)
            and HCCL广播
                T->>W: 每patch的int32 indices
                T->>W: 每patch的values
            end
            W->>W: 完整接收后再做PP过滤
            W->>L: load_checkpoint_weight_patches
            L->>L: int64边界比较 / 重复索引 / NaN校验
            opt MoE direct开启且满足BF16 ND静态EP等条件
                L->>L: TP/EP映射 / 按w13,w2分组 / 原地覆盖
            end
            L->>L: 剩余patch NaN staging / loader view / native load
            W-->>T: 应用与设备同步完成
            T->>T: 等待RPC / 同步发送stream
        end
    end
    U->>M: 汇总错误状态
    U->>W: finish_weight_update(version) / continue_generation
    U-->>A: 同步完成
    A->>A: offload_train 时休眠训练权重
```

## 核心修改与代码证据

| 修改位置 | 修改逻辑 | 解决的问题 |
| --- | --- | --- |
| Ascend `sparse_hccl_engine.py::send_dense_weight_chunk` | 同一HCCL组发送dense元数据和完整values，等待应用完成 | 补齐VIME首轮调用；不发送全量indices，不做NaN staging |
| `SparseHCCLWeightTransferUpdateInfo` / `_receive_patches` | dense模式的indices为None；sparse模式为int32 | 让元数据、发送和接收广播次数一致 |
| `receive_weights` | 先收完整wire，再按PP参数存在性过滤 | 保持所有worker collective顺序一致 |
| `_prepare_patches` / `sparse_weight_patch.py` | 结构检查后按设备汇总数值标志，通信前拒绝错误 | 减少逐patch主机同步，保留bounds/duplicate/NaN检查 |
| `sparse_moe_direct.py` | 校验后按runtime参数分组写入，未处理patch保序回退 | 避免支持的MoE patch完整形状staging及重复loader调用 |
| 公共 `checkpoint_weight_patch.py` | 边界比较显式使用int64；wire不变 | 避免不同后端对大Python整数与int32比较的差异 |

表注：位置均在 VIME 提交的 dependency patch 内；可读源码见下面链接。最后一项是本候选相对复用 #480 流程的新增逻辑，其他为恢复已有协议能力。

- [完整Ascend引擎](core-source/sparse_hccl_engine.py)：`receive_weights`、`_prepare_patches`、`send_dense_weight_chunk`、`_broadcast_chunk`。
- [发送校验](core-source/sparse_weight_patch.py)、[MoE映射](core-source/sparse_moe_direct.py)。
- [公共loader](core-source/checkpoint_weight_patch.py)、[边界回归用例](core-source/test_checkpoint_weight_patch.py)。
- [VIME updater](core-source/update_weight_from_sparse_hccl.py)：沿用原有调度、diff/gather和失败状态机。

MoE direct在推理runtime视图下执行，临时checkpoint视图仅包围回退loader，退出时恢复。HF flat index不能直接当作runtime flat index。patch values是目标值覆盖，0值也必须写入。

## 失败处理

```mermaid
sequenceDiagram
    participant U as Updater
    participant M as Training Ranks
    participant W as Inference Workers
    U->>M: 汇总error与payload_started
    alt payload尚未开始且清理成功
        U->>W: 结束旧会话并恢复生成
        U->>U: 清快照 / needs_reconnect / 下次dense seed
    else payload可能已开始或finish失败
        U->>U: 标记_failed，拒绝继续复用
        Note over U,W: 不主动恢复生成；排查并重连/重启后dense seed
    end
```

更新原地执行，不能回滚已写参数。RPC与broadcast并发是必要的，不能先等待接收RPC结束再开始broadcast。通信链路故障可能令collective等待；当前没有宣称事务恢复或所有故障下的有限时退出保证。

## 可行性与延迟预估边界

接口/补丁层面可行：两份patch均能在固定原始源码上直接 `git apply --check` 和应用，生成源码与被测候选一致；96项CPU回归通过；真实单NPU的BF16 dense→sparse加载及大张量边界校验通过。新候选的多卡HCCL、TP2/PP2/EP2 oracle、30B完整同步仍需实测。

首轮成本包括全量导出、dense HCCL、原生加载及建立快照，不能保证与full耗时相同。稳态关键路径包含diff/export、gather、发送校验、广播和接收应用；部分阶段包含等待或互相重叠，不能简单把profile数值相加。

BF16 payload模型：full约 `2*N` 字节；sparse约 `(4+2)*K=6*K` 字节。仅在变化位置比例 `K/N<1/3` 时，sparse tensor payload小于full。这个阈值只比较payload，忽略元数据、fan-out、分桶、loader和diff/gather，不能当作端到端加速阈值。

#480历史参考：直写关闭的稳态均值78.73秒；直写开启51.20秒；full为59.73秒。它们不是本候选的预估秒数或SLA。需要同配置、同payload复测后才能给出新候选实际性能判断。
