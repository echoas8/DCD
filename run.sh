#!/bin/bash
set -e

PY=./frame_v2/main.py
GPU=0
BATCH=1
LOGDIR=logs_flops
PLAN=plan.txt

mkdir -p ${LOGDIR}

echo "==============================="
echo "Start running by FLOPs plan..."
echo "PLAN=${PLAN}"
echo "LOGDIR=${LOGDIR}"
echo "==============================="

group_id=0

while IFS= read -r line; do
  # 跳过空行
  [[ -z "$line" ]] && continue

  # 注释行：打印组信息
  if [[ "$line" == \#* ]]; then
    echo ""
    echo ">>> ${line}"
    group_id=$((group_id + 1))
    continue
  fi

  # 一行就是一组：组内并发
  IFS='|' read -ra models <<< "$line"

  pids=()
  echo "[GROUP ${group_id}] Launching ${#models[@]} jobs..."

  for m in "${models[@]}"; do
    m=$(echo "$m" | xargs)  # trim
    model=$(echo "$m" | awk -F',' '{print $1}')
    pretrained=$(echo "$m" | awk -F',' '{print $2}')
    flops=$(echo "$m" | awk -F',' '{print $3}')

    log="${LOGDIR}/${model}__${pretrained}.log"

    echo "  -> RUN model=${model}, pretrained=${pretrained}, FLOPs=${flops}, log=${log}"

    python ${PY} \
      --batch_size ${BATCH} \
      --model ${model} \
      --pretrained ${pretrained} \
      --gpu ${GPU} \
      > "${log}" 2>&1 &

    pids+=($!)
  done

  # 等待组内全部结束
  echo "[GROUP ${group_id}] Waiting for all jobs to finish..."
  for pid in "${pids[@]}"; do
    wait $pid
  done
  echo "[GROUP ${group_id}] ✅ All done."

done < ${PLAN}

echo ""
echo "🎉 All groups finished."
