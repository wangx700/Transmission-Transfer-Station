#!/bin/bash
# The retained remote validation environment; use fresh output paths for reruns.
run_root=/home/w00899129/pr18915-oct21.AxPrLW
source /home/w00899129/vllm-code/vllm_vime/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
export LD_LIBRARY_PATH="/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$run_root/source:$run_root/vllm:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export VLLM_VERSION=0.30.0
export OMP_NUM_THREADS=1
export HCCL_IF_BASE_PORT=58400 HCCL_HOST_SOCKET_PORT_RANGE=63200-63250 HCCL_NPU_SOCKET_PORT_RANGE=63300-63350
cd "$run_root/source"
python -m pytest tests/ut/distributed/weight_transfer/test_sparse_hccl_engine.py tests/ut/ops/test_routed_experts.py -q
python -m pytest -rs tests/e2e/pull_request/eight_card/test_sparse_hccl_checkpoint.py --basetemp="$run_root/e2e-repeat"
# For a fresh real-example run, reuse the exact source with a new output root.
example_root=$(mktemp -d /home/w00899129/pr18915-example-repeat.XXXXXX)
ln -s "$run_root/source" "$example_root/source"
export PR18915_0330_ROOT="$example_root"
python "$run_root/example-driver.py"
