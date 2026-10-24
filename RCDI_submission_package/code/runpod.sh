#!/usr/bin/env bash
# One-command experiment on a single 80 GB+ GPU (H100, A100 80GB or H200).
#   bash runpod.sh smoke   -> 5 queries per workload (~10 min) to check that everything works
#   bash runpod.sh full    -> paper run (800 queries per workload, 3 workloads, 3 model tiers)
# Output: rcdi_<mode>.tar.gz (download it before terminating the pod).
set -euo pipefail
MODE=${1:-smoke}
if [ "$MODE" = "full" ]; then N=800; OUT=results; else N=5; OUT=results_smoke; fi
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p logs "$OUT"

echo ">> installing packages"
pip install -q "vllm==0.6.6.post1" "transformers==4.47.1" "tokenizers<0.22" "huggingface-hub<1.0" "datasets<4" "pyarrow<19" -r requirements.txt

start () {  # model port gpu_fraction
  if curl -s "localhost:$2/v1/models" >/dev/null 2>&1; then echo ">> $1 already up"; return; fi
  echo ">> starting $1 on :$2"
  nohup vllm serve "$1" --port "$2" --gpu-memory-utilization "$3" --max-model-len 6144 \
        --seed 0 --dtype half --disable-log-requests > "logs/vllm_$2.log" 2>&1 &
  for i in $(seq 1 120); do
    if curl -s "localhost:$2/v1/models" >/dev/null 2>&1; then echo ">> $1 ready"; return; fi
    sleep 10
  done
  echo "!! $1 failed to start; see logs/vllm_$2.log"; tail -n 40 "logs/vllm_$2.log"; exit 1
}
# largest first; the three servers share one GPU (0.50 + 0.25 + 0.10 of memory). Port 8001 is taken by the pod's nginx.
start Qwen/Qwen2.5-32B-Instruct-AWQ 8003 0.50
start Qwen/Qwen2.5-7B-Instruct      8002 0.25
start Qwen/Qwen2.5-1.5B-Instruct    8011 0.10

echo ">> collecting records ($MODE)"
python collect.py --n "$N" --out "$OUT" 2>&1 | tee "logs/collect_$MODE.log"
python analyze.py --res "$OUT" --reps 200 | tee "logs/analysis_$MODE.log"
tar czf "rcdi_${MODE}.tar.gz" "$OUT" logs
echo ">> DONE. Download rcdi_${MODE}.tar.gz, then TERMINATE the pod."
