# HiddenStates 回放进展、extract_hidden_states 与 OPD 的关系及 Ascend 映射

核对日期：2026-10-23（北京时间）。上游基准：本地 main `19c0e170d2`（2026-10-23，较 10-23 主体报告基准 `432b1003f8ea` 新 14 提交）；Ascend 基准：本地 main `db8836752`（2026-10-23）。PR 状态经 GitHub API（#62411、#59543）与本地 git 历史 grep 双通道核对；未执行 GPU/NPU 数值验证。

- 上游主报告：[特性流程说明.md](特性流程说明.md)（H 节）、[README.md](README.md) 六项特性表
- 本文回答三个问题：`extract_hidden_states` 与 OPD hidden states 的区别；HiddenStates 回放当前全部进展（含新提案 #62411、#59543）；这些在 vllm-ascend 上的映射与缺口。

## 1. `extract_hidden_states` 与 OPD hidden states 的区别

OPD（Online Predictive Decoding）指 vLLM 在线投机解码框架（EAGLE/DFlash/DSpark 等 drafter + target 验证）。两者**捕获机制完全同源**，用途相反：

| | OPD（真投机）的 hidden states | `extract_hidden_states`（伪投机方法） |
| --- | --- | --- |
| hidden states 身份 | drafter 的输入（燃料） | 导出产物（数据集） |
| 捕获机制 | H1–H3：forward 在 `eagle_aux_hidden_state_layer_ids` 层边界收集 aux states | **同一套**，零新增捕获 |
| 消费者 | 引擎内 drafter head，进程内即用即弃 | ExtractHiddenStatesModel（cache-only attention）→ HiddenStateCache 池 → KV connector → 每请求 safetensors → 外部 trainer |
| 对生成的影响 | 加速；token 经 draft+verify 管线 | 零影响：propose 直接返回 target 采样 token、"always verify"（`vllm/v1/spec_decode/extract_hidden_states.py:98-106,172-177`） |
| 成本 | drafter 权重 + 每步 draft forward + verify | 无 attention 计算；代价是 hidden-state 池内存 + D2H + 磁盘 |
| 产物 | 无 | safetensors：hidden states + 对齐 token_ids（`include_output_tokens` 控制末 token） |

机制要点：`extract_hidden_states` 注册在 `speculative_config.method` 槽位但不投机（`num_speculative_tokens` 恒为 1）。真正的动作是把 aux hidden states stack 成 `[num_tokens, num_aux_layers, hidden_size]`，经一层 **CacheOnlyAttentionLayer**（"Cache hidden states as KV pairs without computing attention"，`vllm/model_executor/models/extract_hidden_states.py:279-301`）写入专门的 `HiddenStateCacheSpec` 缓存池（num_heads←aux 层数、head_size←hidden_size，:342-350,306-313）。**借 KV-cache 形态完整复用 KV connector 的传输/保存生命周期**（`@maybe_transfer_kv_layer` → connector `save_kv_layer` → 异步 D2H → 文件），这是 connector 路线不需要动 scheduler/engine 核心的根本原因。

推论：**两者抢同一个 method 槽位，互斥**——配 `extract_hidden_states` 就不能同时跑 EAGLE3/MTP 真 drafter。

## 2. HiddenStates 回放当前进展全景（截至 2026-10-23）

**已合入主线（衬底）**：H1–H4 内部提取（`SupportsEagle3`/`EagleModelMixin`、`models/extract_hidden_states.py`、`v1/spec_decode/extract_hidden_states.py`、`example_hidden_states_connector.py` 的请求级文件导出含 `hidden_states_path`/lock file/cleanup）。OPD 消费与 connector 导出共享该衬底。

**待合入提案（均 Open，git/API 核对未合入）**：

| PR | 创建 | 路线 | 形态 | 关键约束 |
| --- | --- | --- | --- | --- |
| [#59543](https://github.com/vllm-project/vllm/pull/59543) | 10-01 | Core 通道 | `SamplingParams.return_last_hidden_states` + `--enable-return-last-hidden-states`，每个生成 token 的 `compute_logits` 输入行内联进 `CompletionOutput.last_hidden_states`（经 `vllm/v1/token_outputs.py` 类型化通道） | 237 行/12 文件，任意 MRV2 模型；MRV2-only，启动即拒 spec decode/PP/CP；CPU 367 通过，GPU 测试收集待跑（作者早期 transport 有 RTX 4090/PRO 6000 证据） |
| [#57736](https://github.com/vllm-project/vllm/pull/57736) | 早 | Runner 级 | 窗口捕获（capture plan + 可选 aux 层）→ in-process 返回 | 改 scheduler/request/engine 协议/output processor；**可与真 drafter 共存** |
| [#62411](https://github.com/vllm-project/vllm/pull/62411) | 10-19 | Connector 级 | `ExampleHiddenStatesConnector` 窗口化导出（start/end/min_rows/layer layout；窗口外行不写缓存；有界 pinned D2H） | 几乎不动核心（+424 行集中 connector）；**不与真 drafter 共存**（占 method 槽）；V1 runner/PP=1/无 async scheduling/禁 prefix cache/仅 CUDA |
| [#37002](https://github.com/vllm-project/vllm/pull/37002) | 早 | Observation | Phase 1 抽象/config | 不足以自动完成 Phase 2 runner hook |

相关但未入 main：#48743（extraction connector 状态返回）、#57185（n=1 最后 prompt 位状态经 kv_transfer_params）、#62145/#62272（命名 readouts，将作为 #59543 通道的后续 entry）、RFC #59542/#56916/#47280。

**在线可用性判定（#62411）**：机制上**可以在线**——主线协议管道已打通：请求 `kv_transfer_params`（`entrypoints/openai/completion/protocol.py:248`，透传 `:401`）→ connector `build_connector_meta` 按请求预分配 `{storage_path}/{req_id}.safetensors` 并预建 lock file（`example_hidden_states_connector.py:494-516`）→ `request_finished` 返回路径（`:559`）→ 响应回传（`completion/serving.py:694,704`）。四个折扣：①未合入（跑 PR 分支）；②文件形态非内联（client 需共享文件系统）；③文件生命周期自理（持续负载下磁盘累积）；④资源代价（禁 prefix cache；request_finished 延迟释放 KV blocks 至导出完成，高 QPS 有 block 压力）。若要"响应内联张量"形态，等 #59543。

## 3. vllm-ascend 映射（`db8836752`）

**MRV1（V1 runner，Ascend 传统默认）**：原生 NPU 适配 `vllm_ascend/worker/v1/spec_decode/extract_hidden_states_proposer.py` 的 `AscendExtractHiddenStatesProposer`（继承上游 `ExtractHiddenStatesProposer`）。差异三点（类 docstring 自述）：ACL graphs 替代 CUDA graphs；`dummy_run` 按 Ascend 签名实现 ACL capture；`prepare_next_token_ids_padded` 适配 Ascend indices/count 模式。另有一处关键 DP 修复：上游 `coordinate_batch_across_dp` 向同一 DP cpu_group 投递不同形状 tensor 会破坏 gloo 集合通信，Ascend 改为 SP-pad 后复用 `runner._sync_metadata_across_dp`。

**MRV2（V2 runner，现为 Ascend 默认）**：`worker/v2/README.md:53` 明确勾选 `[x] extract_hidden_states (MRV2)`——直接复用上游 proposer（`worker/v2/spec_decode/__init__.py:38-41`）；`worker/v2/attn_utils.py` 特判 hidden-state 池："extract_hidden_states dumps are live at the same time as the target"、单 tensor 无 K/V split。

**平台 patch**：`patch/platform/patch_mamba_config.py:155-166` 对 hybrid+KV store 特判 `method=="extract_hidden_states"`；patch 注册表（`patch/__init__.py:404-409`）放宽上游 eagle3/extract_hidden_states 的模型类型限制（MiniMax）。

**e2e 矩阵**（`tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py`）：Qwen3-8B 真权重（aux 层 [2,18,34]）× eager/ACL graph × MRV1/MRV2 + Qwen3.5-0.8B hybrid dummy 权重（token_ids round-trip 校验）。

**缺口与移植清单**：

1. **在线 HTTP 形态未在 NPU 验证**——现有 e2e 全部为离线 `LLM` 形态；`vllm serve` + kv_transfer_params 回传路径未测。
2. **#62411 无法按原样上 Ascend**：CUDA-only/V1 runner 限定；移植四关：V2 runner 兼容（Ascend 默认已是 V2）、pinned D2H 的 NPU 等价（NPU 无 UVA，`UvaBuffer` 被 patch 成 dummy）、Ascend scheduler patch 体系下 finished_req_ids 完成语义回归、禁 prefix cache 与 RL rollout 形态的相容性。
3. **#59543 若合入**：Ascend 需补 MRV2 的 NPU gather/输出通道验证（token_outputs 通道本身是设备无关的，主要风险在 runner gather 点与 ACL graph 组合）。
4. 互斥约束与上游一致：Ascend 上同样不能与 MTP/EAGLE 真 drafter 并存（method 槽位冲突）。

## 4. vllm-ascend 现在可以做的事情（已验证可用）

以下均有当前 main 的代码/测试证据，今天即可使用：

1. **离线 hidden states 提取与文件导出全链路**：`speculative_config.method=extract_hidden_states` + `ExampleHiddenStatesConnector`。MRV1 走原生 `AscendExtractHiddenStatesProposer`（ACL graph capture、DP 集合通信修复），MRV2 复用上游 proposer；dense（Qwen3-8B，aux 层 [2,18,34]）与 hybrid（Qwen3.5-0.8B）均有 e2e。**用途**：EAGLE3/DFlash/DSpark 草稿训练数据离线采集、蒸馏/激活诊断数据集构建（文件形态，token_ids 已对齐）。
2. **OPD 推理正常线**：DSpark/MTP/EAGLE/DFlash 投机解码 patch 体系齐全。与第 1 条共享 method 槽位——**采集与加速可按部署分时切换**（同一次部署内互斥）。
3. **扩展模型面的现成范式**：`patch_mamba_config.py` 的 hybrid+KV store 特判、patch 注册表的模型类型放宽（MiniMax）——新模型接入提取链路时有先例可循。
4. **hidden states 未定型期间的替代诊断通道**：R3 路由回放（`--enable-return-routed-experts`，TP+EP e2e）、L 的 logprobs + Trace Replay（MRV2）、M sampling mask（NPU Triton kernel）、B batch invariance（beta）——训推一致性诊断不必然等 hidden states。

## 5. 待做事项

### 5.1 不依赖上游合入，可立即启动

| # | 事项 | 要点与验收 |
| --- | --- | --- |
| T1 | **在线 HTTP 形态 e2e** | `vllm serve` + extract_hidden_states + 请求 `kv_transfer_params` 回传路径；覆盖 MRV1/MRV2 × eager/ACL、并发下 lock file → `get_finished` 完成语义、取消/失败清理、`cleanup_hidden_states` 磁盘策略。验收：响应回传 `hidden_states_path` 且文件完整可 load |
| T2 | **#62411 分支 PoC（移植四关）** | ① V2 runner 兼容改造点盘点；② pinned D2H 的 NPU 等价（NPU 无 UVA）；③ Ascend scheduler patch 体系下 finished_req_ids 语义回归（#62411 的 completion tracking 修复恰好动这块）；④ 禁 prefix cache 对 RL rollout 成本的量化评估 |
| T3 | **成本基准** | hidden-state 池内存模型（`num_aux_layers × hidden_size × tokens × dtype`）、D2H/文件 IO 吞吐、磁盘增长曲线；`request_finished` 延迟释放 KV blocks 的 QPS 压力点定位 |
| T4 | **六特性 NPU 实测最小闭环** | M+L（sampling mask + processed_logprobs）、R（TP/EP 行对齐）、B（A2/A3/950 混批）——现有结论均为代码/测试存在性证据，补真实硬件数据 |

### 5.2 依赖上游合入后的对齐

| # | 触发条件 | Ascend 动作 |
| --- | --- | --- |
| T5 | #59543 合入 | NPU 等价 hook（上游 hook 在 `gpu_model_runner.py` 仅 +7 行，Ascend V2 runner 需对应 gather 点）；token_outputs 通道 × ACL graph 组合验证；注意其 PP/CP 拒绝约束与 Ascend RL 部署形态的相容性 |
| T6 | #62411 合入 | 按 T2 清单转正；窗口解析/掩码写缓存不动，平台差异集中于 D2H 与 runner 挂点 |
| T7 | #57736 合入 | runner 级 capture plan 的 NPU 适配；若主线选择该方向（与 drafter 共存），优先级最高 |
| T8 | 持续 | 三线任一定型即触发对齐评估；为每条线锁定上游锚点 commit 再升级 |

### 5.3 需要内部决策的点

- **D1 互斥取舍**：rollout 是否需要 MTP 加速与 hidden 导出并存？并存 → 只能等 #57736 线；分时 → `extract_hidden_states` 路线今天可用。
- **D2 格式未定型前的内部 schema**：hidden states 文件先带 `schema_version`/窗口与层映射元数据（对齐主报告表 9 建议），不冒充 vLLM 公共格式，上游定型后迁移。
- **D3 向上游反哺**：Ascend 侧 DP gloo 修复、ACL graph capture 模式经验，可在 #62411/#57736/RFC 下补跨硬件视角（#59543 正在征集 GPU 测试运行，Ascend 数据点可作参考但不替代其 CUDA 验证要求）。

## 6. 结论

1. `extract_hidden_states` 是 OPD 框架的"无 drafter 复用"：捕获同源、用途从加速改为导出；生成结果与不开它严格一致。
2. H 对外回放当前有两条主提案线（runner 级 #57736 可共存 vs connector 级 #62411 低侵入文件导出）+ 一条最小内联线（#59543），均未合入；**跟踪清单从 2 项扩为 4 项**（加 #62411、#59543）。
3. vllm-ascend 对 `extract_hidden_states` 衬底的支持是**全量且原生**的（V1 NPU proposer + V2 复用 + 平台 patch + e2e 矩阵），后续提案合入后的 Ascend 适配成本可控；在线 HTTP 形态是当前最大未验证缺口。

## 证据命令

```bash
git -C vllm log origin/main --oneline --grep="(#59543)"        # 空 => 未合入
git -C vllm-ascend grep -n "AscendExtractHiddenStatesProposer" -- "vllm_ascend/worker/v1/model_runner.py"
git -C vllm-ascend grep -n "extract_hidden_states" -- "vllm_ascend/worker/v2/README.md"
```
