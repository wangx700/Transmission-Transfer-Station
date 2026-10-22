# 代码证据摘录

摘录日期：2026-10-22。所有摘录来自本地工作树，前缀数字为原文件行号。HEAD仅标识工作树起点；存在本地修改时不能将内容等同于该commit。SHA256标识完整源文件的本次内容。

表注：状态为git status对该源文件的结果，clean表示未跟踪到修改；untracked表示源文件未被Git跟踪。摘录仅用于解释接口，不是可单独导入的源码包。

| 摘录 | 源路径 | HEAD | 工作树状态 | 源文件SHA256 |
|---|---|---|---|---|
| [vime-export](vime-export.txt) | pr447_sync/vime/backends/megatron_utils/update_weight/megatron_delta_export.py | f9a9ed96f79cd65a28d6bb63f3945b13898d44af | clean | 1A70BC4C5C11C11EFD2C7EED2A1C17EB7F99016C83203BF98B70825831B300B1 |
| [verl-diff](verl-diff.txt) | verl-sglang/verl/verl/workers/engine/utils.py | 5026915b60fb0a99a0d6e929b4b5af9a4ceae1f8 |  M verl/workers/engine/utils.py | 7CA782878BBC1A1D923C73ABE3567B23121D379CC35DAFF633009F75FD38580F |
| [sglang-loader](sglang-loader.txt) | verl-sglang/verl/verl/workers/rollout/sglang_rollout/delta_loader.py | 5026915b60fb0a99a0d6e929b4b5af9a4ceae1f8 |  M verl/workers/rollout/sglang_rollout/delta_loader.py | 5EACE9E158FED7BD49DCA79429FEA2BE04F38BBAD60A7D33D2F287ACC97C931D |
| [vllm-moe-parallel](vllm-moe-parallel.txt) | vllm-pr18915-backport/vllm/model_executor/layers/fused_moe/config.py | e6bfe03ad73a3330cb427885aa90d97a12e1c704 | clean | 5520E19A6E329E75D69E24744024460649A9EEE4471F0A54C02A74DBF3B48288 |
| [vllm-qwen3-moe](vllm-qwen3-moe.txt) | vllm-pr18915-backport/vllm/model_executor/models/qwen3_moe.py | e6bfe03ad73a3330cb427885aa90d97a12e1c704 | clean | 3A0DAE7A1B0140BCBF1F7903135C72E5E44A8174C7FADE026B77659C8377D7EE |
| [vllm-patch](vllm-patch.txt) | vllm-pr18915-backport/vllm/model_executor/model_loader/checkpoint_weight_patch.py | e6bfe03ad73a3330cb427885aa90d97a12e1c704 |  M vllm/model_executor/model_loader/checkpoint_weight_patch.py | C7C7CB622D58B2976C37FD33A941DFE06426DF3F1DAE2394683D0D67991F7B83 |
| [ascend-layout](ascend-layout.txt) | ascend-pr18915-backport/vllm_ascend/ops/fused_moe/routed_experts.py | fd815467c221ee600137f6bdd53fe354d5e7c999 |  M vllm_ascend/ops/fused_moe/routed_experts.py | AF3D98A20041635AEEAEDD0117D59BDD61CE82E2325EF942CAEE16CBD34B8547 |
| [ascend-moe-direct](ascend-moe-direct.txt) | ascend-pr18915-backport/vllm_ascend/distributed/weight_transfer/sparse_moe_direct.py | fd815467c221ee600137f6bdd53fe354d5e7c999 | ?? vllm_ascend/distributed/weight_transfer/sparse_moe_direct.py | 44F630FB1EB45910EA12869D843A2568EDC76C06568CCDFFC2E0EABAEF1E1EFD |
| [ascend-receiver](ascend-receiver.txt) | ascend-pr18915-backport/vllm_ascend/distributed/weight_transfer/sparse_hccl_engine.py | fd815467c221ee600137f6bdd53fe354d5e7c999 | ?? vllm_ascend/distributed/weight_transfer/sparse_hccl_engine.py | D91345D5E6A83CB83EA2F5EC3887E8F27E87C44A7D41A70DEE1A5825B228AFCE |
