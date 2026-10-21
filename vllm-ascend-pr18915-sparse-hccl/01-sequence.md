# 整体 UML 时序流程

PR [#18915](https://github.com/vllm-project/vllm-ascend/pull/18915)，Ascend 提交 `ee1f30111881795103af01a0fe19caf41d20a5ac`；上游 vLLM 固定 `c4df37dfcc7f72ca41cd43801a468526ecf99a2b`。本材料按 2026-10-21 验证版本整理，不代表自动更新后的其他上游版本。

## 阅读说明

实线箭头表示调用或发送消息，虚线箭头表示返回；`par` 表示双方必须并发进入 collective。图按当前 HTTP 示例的单次 send_weights 路径组织，不表示 HCCL 数据经 HTTP 传输：HTTP 是控制面，HCCL 是数据面。

独立 trainer rank=0；worker rank=1+dp_rank×(TP×PP)+rank_within_dp；总参与者=1+TP×PP×DP，EP 不再额外乘一次。每个参与者独占 NPU。

## 1. 建组、稀疏更新与恢复推理

```mermaid
sequenceDiagram
    autonumber
    participant C as 调用方/示例
    participant T as HCCL Trainer
    participant R as RPC辅助线程/HTTP
    participant W as 全部TP/PP/DP Workers
    participant H as HCCL组
    participant V as Loader Views
    participant L as vLLM Checkpoint Loader
    participant M as 模型参数
    C->>C: 同一baseline，生成checkpoint坐标patch
    C->>R: pause generation
    C->>T: Factory.trainer_init(init_info)
    T->>R: 辅助线程：init_weight_transfer_engine
    R->>W: init_transfer_engine(rank_offset=1)
    W->>W: 校验配置、world_size、rank
    par Trainer rank 0
        T->>H: 加入stateless group
    and 所有worker
        W->>H: rank=1+dp_rank*(TP*PP)+local_rank
    end
    H-->>T: rendezvous完成，等待worker初始化RPC
    T-->>C: trainer engine
    C->>T: send_weights(patches)
    T->>T: prepare：校验、设备搬运、contiguous
    Note over T,W: 空列表返回；列表内零nnz patch目前未过滤
    T->>R: start_weight_update
    R->>W: 配置/布局guard，不重新初始化权重
    T->>R: 辅助线程：update_weights(metadata)
    R->>W: 基类update_weights → receive_weights
    par trainer主线程
        T->>H: 每patch广播int32 indices及values
    and 每一个worker
        W->>H: 消费相同次序的全部广播
    end
    W->>W: indices转int64，然后按PP归属过滤
    W->>V: enter checkpoint_loader_views
    V->>M: 临时transpose视图，共享storage
    W->>L: load_checkpoint_weight_patches
    L->>L: 范围校验、NaN staging、按顺序分块
    L->>M: model.load_weights：TP/EP/packed映射
    M->>M: 最终copy仅替换非NaN元素
    L-->>W: 应用完成
    W->>V: 退出ExitStack，逆序关闭view
    V->>M: 恢复原推理data视图
    W->>W: 基类torch.accelerator.synchronize()
    Note over W: 同步调用存在；跨ambient device目标尚无直接测试
    W-->>R: update RPC返回
    R-->>T: future.result完成
    T->>T: 发送stream同步
    T->>R: finish_weight_update
    R->>W: sparse finish为no-op，框架结束session
    T-->>C: send_weights返回
    C->>T: shutdown communicator
    C->>R: resume generation，仅成功路径
    C->>R: completions请求
    R->>M: 正常推理
    M-->>R: tokens
    R-->>C: HTTP response
```

图注：recv 的 NPU 上下文在返回前退出，随后基类做同步；图只标出真实调用，未对尚未直接测试的跨设备同步作保证。只有成功路径才 finish/resume。send_weights 失败会跳过 finish；示例 finally 关闭 engine，异常继续传播而不执行 resume。

## 2. 多层 view 的进入与异常清理

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker
    participant S as ExitStack
    participant A as 第一层view
    participant B as 第二层view
    participant L as 原生loader
    W->>S: enter checkpoint_loader_views
    S->>A: enter_context
    A-->>S: 已进入并注册退出
    alt 第二层进入失败
        S->>B: enter_context抛异常
        S->>A: 退出已进入view
        S-->>W: 原异常传播，loader不执行
    else 所有层成功进入
        S->>B: enter_context成功
        S-->>W: yield
        W->>L: 加载patch
        alt 加载成功
            L-->>W: 正常返回
        else 加载失败
            L-->>W: 抛异常
        end
        W->>S: 退出上下文
        S->>B: 逆序退出
        S->>A: 逆序退出
        S-->>W: 正常返回或传播异常
    end
    Note over W,L: 恢复视图不等于恢复权重数值；失败后不得自动resume
```

图注：单层 view 用 finally 恢复 parameter.data；外层 ExitStack 管理已进入 view 的逆序退出。失败不提供权重事务回滚。

## 3. E2E 的 dense 参考对比

```mermaid
sequenceDiagram
    autonumber
    participant P as pytest/验证脚本
    participant W as 全部Workers
    participant T as HCCL Trainer
    P->>P: 生成确定性小checkpoint和patch记录
    P->>W: 启动服务并capture baseline
    P->>W: 原生dense加载同一更新
    W-->>P: 全worker哈希及生成参考
    P->>W: restore baseline，验证恢复
    P->>W: pause
    P->>T: 初始化并发送相同patch
    T->>W: 真实HCCL更新
    W-->>P: 全worker参数哈希
    P->>P: 断言哈希完全等于dense参考
    P->>W: resume并生成
    W-->>P: greedy token及finish_reason
    P->>P: 断言生成相同，写validation.json
    Note over P,W: DP通过all_gather_object核对所有副本；不只检查HTTP返回的DP0
```

## 分块更新与示例差别

示例用 send_weights：prepare → start → broadcast → finish。checkpoint E2E 用调用方持有的 session：start → 多次 send_weight_chunk → finish。chunk 入口本身不自动 start/finish；不能把两种生命周期嵌套。

E2E 每四条记录构成一个 chunk；重复 q_proj 名称用于验证按输入顺序覆盖。PP=2 时创建两层模型，分别覆盖两个 stage。真实示例单独生成首尾层的 expert gate row-swap patch。

`01-sequence.puml` 是可编辑的标准 PlantUML 主时序图；`01-sequence.mmd` 是 Mermaid 主图源文件；`01-sequence.svg` 是可直接查看的概览。细节及异常路径以本文和核心代码为准。
